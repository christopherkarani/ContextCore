# Phase 4: Consolidation Engine

You are implementing Phase 4 of ContextCore. Phases 1–3 are complete. You have data models, memory stores (`EpisodicStore`, `SemanticStore`, `ProceduralStore`), Metal scoring engines (relevance, recency, attention, sentence importance), `WindowPacker`, and `ChunkOrderer`. All compile and pass tests.

Phase 4 builds the background memory maintenance system. Over long agent sessions, episodic memory fills with redundant entries — the user restates preferences, the same facts appear across turns, similar tool call patterns repeat. The consolidation engine runs asynchronously in the background, merging duplicates, promoting repeated facts to semantic memory, and flagging contradictions for the consumer to resolve.

This phase adds 2 new Metal kernels, one new Swift actor (`ConsolidationEngine`), and an internal scheduler. It never blocks the main agent loop.

---

## Why Consolidation Matters

Without consolidation, a 500-turn session produces ~500 episodic memory chunks. Most agent turns contain overlapping information — the user's name, their project context, established preferences. Retrieval degrades because the ANNS index returns 5 near-duplicate chunks instead of 5 diverse, relevant ones. Token budget is wasted on redundancy.

Consolidation fixes this by:
1. **Merging duplicates** — two chunks saying the same thing become one.
2. **Promoting facts** — information repeated across episodes is durable knowledge → moves to semantic memory with higher retention.
3. **Detecting contradictions** — "user prefers dark mode" vs "user switched to light mode" → flagged for the consumer to resolve.

The result: a smaller, higher-quality episodic store and a growing semantic store of verified facts.

---

## The Consolidation Pipeline

```
episodicStore (500 chunks)
    │
    ▼
pairwise_similarity (Metal)          ← O(n²), GPU-only feasible
    │
    ▼
find_merge_candidates (Metal)        ← threshold > 0.92
    │
    ▼
For each duplicate pair:
    ├─ shorter chunk → SemanticStore.upsert()    ← promoted to fact
    ├─ both originals: retentionScore -= 0.2     ← demoted
    └─ if retentionScore < 0.1 → evict           ← garbage collected
    │
    ▼
antipodal_test (Metal)               ← contradiction detection on SemanticStore
    │
    ▼
contradictionCandidates()            ← returned to consumer for LLM resolution
```

---

## Phase 1–3 Types You Depend On

Already exist — import, do not redefine:

```swift
// Phase 1
public actor EpisodicStore {
    func insert(turn: Turn) async throws
    func retrieve(query: [Float], k: Int) async throws -> [MemoryChunk]
    var count: Int { get }
    // You will need to ADD these methods in this phase:
    // func allChunks() async -> [MemoryChunk]
    // func updateRetentionScore(id: UUID, delta: Float) async throws
    // func evict(id: UUID) async throws
}

public actor SemanticStore {
    func upsert(fact: String, embedding: [Float]) async throws
    func retrieve(query: [Float], k: Int) async throws -> [MemoryChunk]
    var count: Int { get }
    // You will need to ADD:
    // func allChunks() async -> [MemoryChunk]
}

public struct MemoryChunk { id, content, embedding, type, createdAt, lastAccessedAt, accessCount, retentionScore, sourceSessionID, metadata }

// Phase 2
// CPUReference — add new reference implementations here
// MetalContext — shared device/library loading
```

**Important**: EpisodicStore and SemanticStore need new methods (`allChunks`, `updateRetentionScore`, `evict`) that weren't part of Phase 1. Add them to the existing actors. Write tests for the new methods alongside the consolidation tests.

---

## File Map (Phase 4 additions)

```
Sources/
  ContextCoreShaders/
    Consolidation.metal         # NEW — pairwise_similarity, find_merge_candidates, antipodal_test
  ContextCoreEngine/
    ConsolidationEngine.swift   # NEW — consolidation actor + scheduler
    CPUReference.swift          # MODIFIED — add pairwise/antipodal references
  ContextCore/
    Memory/
      EpisodicStore.swift       # MODIFIED — add allChunks, updateRetentionScore, evict
      SemanticStore.swift       # MODIFIED — add allChunks
Tests/
  ContextCoreTests/
    ConsolidationTests.swift    # NEW
    ContradictionTests.swift    # NEW
```

---

## Execution Plan

Three sub-tasks. TDD for each.

---

### 4.1 — Pairwise Similarity Kernel

The foundational GPU operation for consolidation. Computes cosine similarity between every pair of chunks in the episodic store.

#### The Math

For n chunks with dim-dimensional embeddings, the pairwise similarity matrix is n×n, symmetric, with diagonal = 1.0. We only compute the upper triangle — (n × (n-1)) / 2 unique pairs.

**Triangular indexing**: Given a linear thread ID `tid` in range [0, n*(n-1)/2), map to matrix coordinates (i, j) where j > i:

```
i = n - 2 - floor((sqrt(4*(n*(n-1)/2 - 1 - tid) + 1) - 1) / 2)
j = tid - i*(2*n - i - 3)/2 + i + 1  -- but this is error-prone
```

Simpler alternative for the kernel: use a 2D dispatch grid where `gid.x` = i, `gid.y` = j, and skip cells where `gid.y <= gid.x`. This wastes half the threads but is trivial to implement and debug. For n <= 4096, the waste is acceptable.

#### Metal Kernels

**`pairwise_similarity`** (`Consolidation.metal`):

```metal
kernel void pairwise_similarity(
    device const float* embeddings   [[buffer(0)]],  // [n * dim]  flat
    device float* similarity         [[buffer(1)]],  // [n * n]    upper triangle written, rest = 0
    constant uint& dim               [[buffer(2)]],
    constant uint& n                 [[buffer(3)]],
    uint2 gid                        [[thread_position_in_grid]]   // 2D dispatch
)
```

Per-thread logic:
1. `uint i = gid.x; uint j = gid.y;`
2. `if (i >= n || j >= n || j <= i) return;` — skip diagonal and lower triangle.
3. Compute cosine similarity between `embeddings[i * dim]` and `embeddings[j * dim]`.
4. `similarity[i * n + j] = cosine;`

**Tiled version** for n > 2048:

The output buffer `similarity[n * n]` would be `2048² * 4 = 16MB` — fine. But at n = 4096, it's `64MB`. At n = 8192, it's `256MB` — too large for a single `MTLBuffer` on many devices (max is typically 256MB).

Solution: tile the computation. Process 512×512 sub-blocks of the matrix. Each tile reads 512 embeddings for rows and 512 for columns from the full embedding buffer, computes 512×512 similarities, and writes to the appropriate region of the output buffer.

```swift
// Pseudocode for tiled dispatch
let tileSize = 512
for rowTile in stride(from: 0, to: n, by: tileSize) {
    for colTile in stride(from: rowTile, to: n, by: tileSize) {  // upper triangle tiles only
        // Dispatch kernel for this tile
        // Pass rowTile, colTile as offsets
    }
}
```

For Phase 4, implement the simple 2D dispatch first. Add tiling only if n > 2048 at runtime:

```swift
if n <= 2048 {
    dispatchSimple(n: n, dim: dim, ...)
} else {
    dispatchTiled(n: n, dim: dim, tileSize: 512, ...)
}
```

**`find_merge_candidates`** (`Consolidation.metal`):

```metal
kernel void find_merge_candidates(
    device const float* similarity   [[buffer(0)]],  // [n * n]
    device uint2* candidates         [[buffer(1)]],  // output pairs (i, j)
    device atomic_uint* candidateCount [[buffer(2)]], // atomic counter
    constant float& threshold        [[buffer(3)]],  // default 0.92
    constant uint& n                 [[buffer(4)]],
    uint2 gid                        [[thread_position_in_grid]]
)
```

Per-thread logic:
1. `uint i = gid.x; uint j = gid.y;`
2. `if (i >= n || j >= n || j <= i) return;`
3. `if (similarity[i * n + j] > threshold)`:
   - `uint idx = atomic_fetch_add_explicit(candidateCount, 1, memory_order_relaxed);`
   - `candidates[idx] = uint2(i, j);`

Pre-allocate the candidates buffer to a reasonable max size (e.g. `n * 10` — if more than 10 candidates per chunk, something is very wrong). Check `candidateCount` after dispatch and warn if it hit the buffer limit.

#### CPU Reference

Add to `CPUReference.swift`:

```swift
/// Full n×n pairwise cosine similarity matrix.
static func pairwiseSimilarity(embeddings: [[Float]]) -> [[Float]]

/// Pairs (i, j) where similarity[i][j] > threshold.
static func findMergeCandidates(
    similarities: [[Float]],
    threshold: Float
) -> [(Int, Int)]
```

#### Swift Wrapper

```swift
public actor ConsolidationEngine {
    private let device: MTLDevice
    private let pairwisePipeline: MTLComputePipelineState
    private let mergeCandidatePipeline: MTLComputePipelineState
    private let antipodalPipeline: MTLComputePipelineState  // added in 4.3

    private let embeddingProvider: any EmbeddingProvider

    public init(
        device: MTLDevice? = nil,
        embeddingProvider: any EmbeddingProvider
    ) throws

    /// Find near-duplicate pairs in the episodic store.
    /// Returns pairs of chunk UUIDs whose similarity exceeds the threshold.
    public func findDuplicates(
        in store: EpisodicStore,
        threshold: Float = 0.92
    ) async throws -> [(UUID, UUID)]
}
```

`findDuplicates` implementation:
1. `let chunks = await store.allChunks()` — get all chunks.
2. Flatten embeddings into a single `[Float]` buffer (n × dim).
3. Dispatch `pairwise_similarity`.
4. Dispatch `find_merge_candidates`.
5. Read back candidate pairs, map indices to UUIDs.

#### Store Extensions

Add to `EpisodicStore`:

```swift
/// Return all stored chunks. Used by consolidation — not for regular retrieval.
public func allChunks() async -> [MemoryChunk]

/// Adjust a chunk's retention score by delta. Clamps to [0, 1].
public func updateRetentionScore(id: UUID, delta: Float) async throws

/// Remove a chunk permanently.
public func evict(id: UUID) async throws
```

Add to `SemanticStore`:

```swift
/// Return all stored facts.
public func allChunks() async -> [MemoryChunk]
```

#### Tests (`ConsolidationTests.swift`)

| # | Test | Setup | Assertion |
|---|---|---|---|
| 1 | CPU pairwise — symmetry | 20 random chunks dim=384 | `sim[i][j] == sim[j][i]` for all pairs |
| 2 | CPU pairwise — diagonal | 20 chunks | `sim[i][i] == 1.0` for all i |
| 3 | GPU vs CPU parity | 100 random chunks dim=384 | Max absolute error < 1e-4 on upper triangle |
| 4 | Merge candidates — true positives | 100 chunks: 90 unique + 10 near-duplicate pairs (create by adding small noise to 10 base vectors) | All 10 pairs detected |
| 5 | Merge candidates — no false positives | 100 fully unique chunks (orthogonal or widely spaced) | Zero candidates at threshold=0.92 |
| 6 | Threshold sensitivity | Same 100 chunks from test 4 | threshold=0.80 returns >= 10 pairs, threshold=0.99 returns fewer |
| 7 | n=1 | 1 chunk | Zero pairs, no crash |
| 8 | n=2 identical | 2 copies of same vector | Exactly 1 pair |
| 9 | Store extension — allChunks | Insert 15 chunks into EpisodicStore | allChunks() returns 15 items |
| 10 | Store extension — updateRetentionScore | Insert chunk with retentionScore=0.5, update by -0.2 | retentionScore == 0.3 |
| 11 | Store extension — evict | Insert 5 chunks, evict 1 | count == 4, evicted chunk not in allChunks |

**Generating near-duplicate test data**:

```swift
func makeNearDuplicate(of base: [Float], noise: Float = 0.05, seed: UInt64) -> [Float] {
    // Add small random noise to each dimension, then L2-normalize
    var rng = SomeSeededRNG(seed: seed)
    var noisy = base.map { $0 + Float.random(in: -noise...noise, using: &rng) }
    // L2-normalize
    let norm = sqrt(noisy.reduce(0) { $0 + $1 * $1 })
    return noisy.map { $0 / norm }
}
```

Near-duplicates created this way will have cosine similarity > 0.95, comfortably above the 0.92 threshold.

**Commit:** `feat(phase4): 4.1 — Pairwise similarity kernel with tiled dispatch`

---

### 4.2 — Semantic Extraction & Scheduling

The CPU logic that uses `findDuplicates` results to promote facts and clean up the episodic store.

#### Consolidation Pipeline

Add to `ConsolidationEngine`:

```swift
/// Run the full consolidation pipeline for a session.
/// 1. Find duplicates in episodic store.
/// 2. Promote shorter chunk of each pair to semantic memory.
/// 3. Demote originals (retentionScore -= 0.2).
/// 4. Evict episodic chunks with retentionScore < 0.1.
public func consolidate(
    session: UUID,
    episodicStore: EpisodicStore,
    semanticStore: SemanticStore,
    threshold: Float = 0.92
) async throws -> ConsolidationResult
```

**`ConsolidationResult`** (new type for observability):

```swift
public struct ConsolidationResult: Sendable {
    public let duplicatePairsFound: Int
    public let factsPromoted: Int
    public let chunksEvicted: Int
    public let durationMs: Double
}
```

#### Step-by-step implementation

```
func consolidate(session, episodicStore, semanticStore, threshold):

    let startTime = ContinuousClock.now

    // 1. Find duplicates
    let pairs = try await findDuplicates(in: episodicStore, threshold: threshold)

    // 2. For each pair, promote the shorter chunk
    var promotedIDs: Set<UUID> = []
    let allChunks = await episodicStore.allChunks()
    let chunkMap = Dictionary(uniqueKeysWithValues: allChunks.map { ($0.id, $0) })

    for (idA, idB) in pairs:
        guard let a = chunkMap[idA], let b = chunkMap[idB] else { continue }

        // Skip if either chunk was already promoted by a previous pair
        if promotedIDs.contains(a.id) || promotedIDs.contains(b.id) { continue }

        // Shorter chunk = more general = better fact candidate
        let fact = a.content.count <= b.content.count ? a : b
        promotedIDs.insert(fact.id)

        // 3. Promote to semantic store
        try await semanticStore.upsert(fact: fact.content, embedding: fact.embedding)

        // 4. Demote both originals
        try await episodicStore.updateRetentionScore(id: a.id, delta: -0.2)
        try await episodicStore.updateRetentionScore(id: b.id, delta: -0.2)

    // 5. Evict low-retention chunks
    let updatedChunks = await episodicStore.allChunks()
    for chunk in updatedChunks where chunk.retentionScore < 0.1:
        try await episodicStore.evict(id: chunk.id)

    let duration = ContinuousClock.now - startTime
    return ConsolidationResult(
        duplicatePairsFound: pairs.count,
        factsPromoted: promotedIDs.count,
        chunksEvicted: updatedChunks.filter { $0.retentionScore < 0.1 }.count,
        durationMs: duration.milliseconds
    )
```

#### ConsolidationScheduler

An internal type that decides *when* to trigger consolidation. It does not own the stores — it receives them.

```swift
internal actor ConsolidationScheduler {
    private let engine: ConsolidationEngine
    private let threshold: Int          // episodicStore.count threshold (default 200)
    private let insertionThreshold: Int  // insertions since last consolidation (default 50)

    private var insertionsSinceLastConsolidation: Int = 0
    private var isConsolidating: Bool = false

    init(
        engine: ConsolidationEngine,
        countThreshold: Int = 200,
        insertionThreshold: Int = 50
    )

    /// Called after every episodicStore.insert().
    /// Checks thresholds and triggers consolidation if needed.
    func notifyInsertion(
        episodicCount: Int,
        session: UUID,
        episodicStore: EpisodicStore,
        semanticStore: SemanticStore
    ) async

    /// Reset insertion counter (called after successful consolidation).
    private func resetCounter()
}
```

`notifyInsertion` logic:
1. Increment `insertionsSinceLastConsolidation`.
2. Check: `episodicCount > threshold` OR `insertionsSinceLastConsolidation > insertionThreshold`.
3. If either true AND `!isConsolidating`:
   - Set `isConsolidating = true`.
   - Fire `Task.detached(priority: .background)` that calls `engine.consolidate(...)`.
   - On completion: set `isConsolidating = false`, call `resetCounter()`.
4. If `isConsolidating` already, skip — don't queue multiple consolidations.

The key constraint: `notifyInsertion` must return immediately. The actual consolidation runs on a detached background task. The main agent loop is never blocked.

#### Tests (`ConsolidationTests.swift` — continued)

| # | Test | Setup | Assertion |
|---|---|---|---|
| 12 | Consolidate — facts promoted | 60 chunks (50 unique + 10 duplicate pairs), consolidate | SemanticStore.count >= 8 (some pairs may share a chunk) |
| 13 | Consolidate — episodic shrinks | Same 60 chunks | EpisodicStore.count < 60 after consolidation |
| 14 | Consolidate — shorter chunk promoted | Pair: chunk A (20 words), chunk B (50 words) | Promoted fact content == chunk A's content |
| 15 | Consolidate — retention decrement | Pair detected, both originals checked | Both chunks' retentionScore decreased by 0.2 |
| 16 | Consolidate — eviction at low retention | Chunk with initial retentionScore = 0.15, after -0.2 = -0.05 (clamped to 0.0) | Chunk evicted from EpisodicStore |
| 17 | Consolidate — idempotent | Run consolidate twice on same data | Second run finds 0 duplicates (already merged) |
| 18 | Consolidate — empty store | 0 chunks | Returns ConsolidationResult with all zeros, no crash |
| 19 | Scheduler — count threshold | Insert 201 chunks into EpisodicStore, call notifyInsertion each time | Consolidation triggered (verify via ConsolidationResult or SemanticStore.count > 0) |
| 20 | Scheduler — insertion threshold | Insert 51 chunks (below count threshold of 200) | Consolidation triggered after 51st insertion |
| 21 | Scheduler — no double trigger | Trigger consolidation, immediately call notifyInsertion again | Second call does not start a second consolidation (isConsolidating guard) |
| 22 | Scheduler — non-blocking | Call notifyInsertion, measure return time | Returns in < 1ms (does not wait for consolidation to finish) |
| 23 | ConsolidationResult — all fields | Run consolidation on known data | duplicatePairsFound, factsPromoted, chunksEvicted all match expected values |

**Testing the scheduler's non-blocking property (test 22)**:

```swift
func testSchedulerNonBlocking() async {
    let clock = ContinuousClock()
    let elapsed = await clock.measure {
        await scheduler.notifyInsertion(
            episodicCount: 250,
            session: sessionID,
            episodicStore: episodicStore,
            semanticStore: semanticStore
        )
    }
    XCTAssertLessThan(elapsed, .milliseconds(10), "notifyInsertion must return immediately")
}
```

**Commit:** `feat(phase4): 4.2 — Semantic extraction with auto-scheduling`

---

### 4.3 — Contradiction Detection

A heuristic for finding semantic facts that contradict each other. This is *not* semantic understanding — it's a signal that two facts are suspiciously similar yet directionally opposed.

#### The Intuition

Contradictory statements are semantically similar (both about the same topic) but encode opposite meanings. In embedding space, this can manifest as two vectors that are close in cosine similarity but have opposite signs on a significant fraction of their dimensions.

Example:
- "The user prefers dark mode" → embedding A
- "The user switched to light mode" → embedding B

These are similar (both about user display preferences) but many embedding dimensions flip sign between "dark" and "light" contexts.

**This is a heuristic**, not ground truth. It produces false positives and false negatives. The output is *candidates* for the consumer to resolve (typically by passing both facts to the LLM and asking which is current).

#### Detection Criteria

A pair of semantic facts is a contradiction candidate when BOTH:
1. **Cosine similarity > 0.75** — they're about the same topic.
2. **Antipodal fraction > 0.30** — more than 30% of embedding dimensions have opposite signs.

#### Metal Kernel

**`antipodal_test`** (add to `Consolidation.metal`):

```metal
kernel void antipodal_test(
    device const float* embeddingsA   [[buffer(0)]],  // [pairCount * dim]  first element of each pair
    device const float* embeddingsB   [[buffer(1)]],  // [pairCount * dim]  second element of each pair
    device float* antipodalFraction   [[buffer(2)]],  // [pairCount] output
    constant uint& dim                [[buffer(3)]],
    constant uint& pairCount          [[buffer(4)]],
    uint gid                          [[thread_position_in_grid]]
)
```

Per-thread logic (one thread per pair):
1. `if (gid >= pairCount) return;`
2. Count dimensions where `sign(embA[gid * dim + d]) != sign(embB[gid * dim + d])` for d in [0, dim).
3. `antipodalFraction[gid] = (float)signDiffCount / (float)dim;`

Sign comparison: `(embA[d] >= 0.0f) != (embB[d] >= 0.0f)`. Treat zero as positive.

#### Swift Wrapper

Add to `ConsolidationEngine`:

```swift
/// Find semantic facts that may contradict each other.
/// Returns pairs where similarity > 0.75 AND antipodal fraction > 0.30.
/// These are candidates for the consumer to resolve — not confirmed contradictions.
public func contradictionCandidates(
    in store: SemanticStore,
    similarityThreshold: Float = 0.75,
    antipodalThreshold: Float = 0.30
) async throws -> [(MemoryChunk, MemoryChunk)]
```

Implementation:
1. `let facts = await store.allChunks()`
2. Run `pairwise_similarity` on all fact embeddings.
3. Find pairs where similarity > `similarityThreshold`.
4. For those pairs, flatten their embeddings into paired buffers and dispatch `antipodal_test`.
5. Filter pairs where `antipodalFraction > antipodalThreshold`.
6. Map indices back to `MemoryChunk` pairs and return.

#### CPU Reference

Add to `CPUReference.swift`:

```swift
/// Fraction of dimensions where sign(a[d]) != sign(b[d]).
static func antipodalFraction(_ a: [Float], _ b: [Float]) -> Float
```

#### Tests (`ContradictionTests.swift`)

| # | Test | Setup | Assertion |
|---|---|---|---|
| 1 | Exact negation | embedding B = embedding A negated (all signs flipped) | antipodalFraction ≈ 1.0 |
| 2 | Identical embeddings | B = A | antipodalFraction ≈ 0.0 |
| 3 | Partial negation | Flip 50% of dimensions | antipodalFraction ≈ 0.5 |
| 4 | GPU vs CPU parity | 50 pairs dim=384 | Fractions match exactly (integer math, no floating point error expected) |
| 5 | Contradiction candidates — found | 20 semantic facts, 2 are contradictory (manually crafted with negated partial dimensions + high similarity) | Both pairs returned |
| 6 | Contradiction candidates — none | 20 facts, all consistent | Empty result |
| 7 | Similarity filter | Pair with antipodal > 0.3 but similarity < 0.75 | Not returned (fails similarity gate) |
| 8 | Antipodal filter | Pair with similarity > 0.75 but antipodal < 0.3 | Not returned (fails antipodal gate) |
| 9 | Empty store | 0 facts | Returns empty, no crash |
| 10 | Single fact | 1 fact | Returns empty (no pairs possible) |

**Crafting test contradictions**:

For tests 5–6, you need pairs that pass BOTH gates. Create them manually:

```swift
func makeContradictoryPair(dim: Int = 384) -> ([Float], [Float]) {
    // Start with a base vector
    var base = TestHelpers.randomVector(dim: dim, seed: 42)

    // Create "contradiction" by flipping ~40% of dimensions
    var contra = base
    for d in stride(from: 0, to: dim, by: 2) {  // flip every other = 50%
        if d < Int(Float(dim) * 0.4) {
            contra[d] = -contra[d]
        }
    }
    // Normalize
    contra = l2Normalize(contra)

    // These will have similarity ~0.8 (still high) and antipodal ~0.4 (above 0.3)
    return (base, contra)
}
```

Verify the test vectors pass both gates before using them in tests.

**Commit:** `feat(phase4): 4.3 — Contradiction detection with antipodal heuristic`

---

## Design Decisions to Be Aware Of

### Why Not Use the LLM for Contradiction Detection?

ContextCore never calls the LLM directly. It prepares context for the consumer's model. Contradiction detection is a *signal* — the consumer decides whether to resolve it via LLM, manual review, or just logging. This keeps the framework model-agnostic.

### Why O(n²) Pairwise Similarity?

Consolidation runs on the *entire* episodic store for a session. ANNS gives approximate nearest neighbors — it might miss near-duplicates due to graph construction artifacts. Exact pairwise comparison catches everything. The O(n²) cost is acceptable because:
- It runs in the background at low priority.
- n is bounded by session length (typically < 500 chunks).
- GPU parallelism makes n=500 → 125K pairs trivial (~1ms on M2).
- The tiled path handles up to n=8192 (33M pairs, still feasible on GPU).

### Retention Score Lifecycle

A chunk starts with `retentionScore = 0.5` (episodic) or `1.0` (semantic). Each consolidation pass that finds the chunk in a duplicate pair decrements by 0.2. A chunk must be part of 3+ duplicate pairs before it's evicted (0.5 → 0.3 → 0.1 → evicted). This prevents aggressive pruning from a single consolidation run.

---

## Final Verification

After all 3 sub-tasks:

```bash
swift build 2>&1      # zero errors, zero warnings
swift test 2>&1       # all Phase 1–4 tests green
```

Report: total test count, pass/fail, consolidation timing on test data.

## Phase 4 Is Done When

- `pairwise_similarity` kernel matches CPU reference within 1e-4 for 100 chunks
- `find_merge_candidates` detects all true duplicate pairs with zero false positives at threshold=0.92
- `antipodal_test` kernel produces exact integer-math results matching CPU reference
- `consolidate()` promotes facts, decrements retention, and evicts correctly
- `ConsolidationScheduler` fires at both thresholds (count and insertion) without blocking the caller
- No double-triggering when consolidation is already in progress
- Tiled dispatch path compiles and works for n > 2048
- `EpisodicStore` and `SemanticStore` have `allChunks`, `updateRetentionScore`, `evict` methods
- Phase 1–3 tests still pass (no regressions)
- 3 clean atomic commits in git history
