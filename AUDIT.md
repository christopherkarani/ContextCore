# ContextCore Production-Readiness Audit

**Auditor**: Principal Engineer (automated)
**Date**: 2026-03-07
**Scope**: Full repository — Sources, Tests, Shaders, Package configuration
**Swift Version**: 6.2 | **Platforms**: iOS 17+, macOS 14+, visionOS 1+

---

## 1. Executive Summary

### Production Readiness Score: **6.5 / 10**

ContextCore is a well-architected Swift concurrency library implementing a multi-tier memory system (episodic, semantic, procedural) with GPU-accelerated scoring, consolidation, and progressive compression. The architecture is clean, actor isolation is largely correct, and the API surface is thoughtful.

However, several correctness issues, missing input validation, concurrency gaps, and insufficient test coverage prevent a confident production release.

### Top 5 Critical Risks

| # | Risk | Severity | Section |
|---|------|----------|---------|
| 1 | `SemanticStore.bestMatchingChunkID` performs O(n) linear scan on every `upsert`, bypassing the ANN index entirely | Blocker | §2.1 |
| 2 | `CPUReference` uses `precondition()` for public API validation — silently undefined in release builds | Major | §2.4 |
| 3 | `ConsolidationScheduler` spawns `Task.detached` with no cancellation tracking — fire-and-forget GPU work survives actor deallocation | Major | §4.1 |
| 4 | Metal shader `find_merge_candidates` silently discards candidate pairs on buffer overflow with no error signaling | Major | §6.1 |
| 5 | Concurrency test coverage is near-zero — no race condition, cancellation, or stress tests exist | Blocker | §7.1 |

### Release Blockers

1. **SemanticStore linear scan**: Will degrade to O(n²) wall-clock time at scale during consolidation. Must use ANN index for deduplication.
2. **Concurrency testing gap**: Zero tests for data races, cancellation behavior, or concurrent buildWindow + consolidation. Cannot certify thread safety without them.
3. **CPUReference precondition crash risk**: Public API will crash (or silently produce garbage) in release builds on invalid input.

---

## 2. Correctness Issues

### 2.1 SemanticStore Linear Scan Deduplication — **Blocker**

**File**: `Sources/ContextCore/Memory/SemanticStore.swift:160-176`

```swift
private func bestMatchingChunkID(for embedding: [Float], threshold: Float) -> String? {
    var bestID: String?
    var bestSimilarity: Float = -1
    for (id, chunk) in chunksByID {            // O(n) full scan
        let similarity = cosineSimilarity(embedding, chunk.embedding)
        // ...
    }
}
```

The `upsert(fact:embedding:)` method calls `bestMatchingChunkID` which iterates **every stored chunk** to find duplicates via cosine similarity. This is O(n·d) per upsert, making consolidation O(n²·d) overall. The store already has a `MetalANNS.StreamingIndex` available — the ANN index should be used for the similarity search with a high-recall search parameter.

**Impact**: With 1000+ semantic facts, upsert latency will dominate consolidation time. At 10,000 facts, this becomes a user-visible hang.

### 2.2 Redundant Buffer Size Calculation — **Minor**

**File**: `Sources/ContextCore/AgentContext.swift:202`

```swift
let maxRecentBuffer = max(configuration.recentTurnsGuaranteed * 2, configuration.recentTurnsGuaranteed)
```

`max(x * 2, x)` for non-negative x always equals `x * 2`. This is either a logic error (the second operand should be a different floor value) or dead code. If `recentTurnsGuaranteed` is 0, both arms are 0. If negative (should be impossible), the multiplication could underflow.

### 2.3 Procedural Embedding Reuse — **Major**

**File**: `Sources/ContextCore/AgentContext.swift:578-595`

```swift
func makeProceduralCandidates(...) -> [(chunk: MemoryChunk, score: Float)] {
    // ...
    let chunk = MemoryChunk(
        content: content,
        embedding: taskEmbedding,  // ← uses the TASK query embedding, not the tool's own embedding
        // ...
    )
}
```

Procedural memory chunks are assigned the **task query embedding** rather than embeddings derived from their own content. This means every procedural chunk in the window has an identical embedding equal to the query. Any downstream embedding-based operation (attention reranking, consolidation similarity) will treat all procedural chunks as identical, defeating semantic differentiation.

### 2.4 CPUReference Precondition Abuse — **Major**

**File**: `Sources/ContextCoreEngine/CPUReference.swift` (lines 18, 19, 25, 53-55, 96, 145, 174, 198, 212)

```swift
precondition(query.count == chunks[0].count, "Query dimension must match chunk dimension")
precondition(chunks.count == recencyWeights.count, "...")
```

`CPUReference` is a `public enum` with all-public static methods. In `-O` (release) builds, `precondition` calls are **removed by the compiler**. This means:
- Mismatched dimensions produce garbage output instead of crashing.
- Out-of-bounds array access can occur on `embeddings[0].count` if the precondition guard is elided.

**Fix**: Replace `precondition` with `guard ... else { throw }` for all public API entry points.

### 2.5 Consolidation Partial Processing — **Minor**

**File**: `Sources/ContextCoreEngine/ConsolidationEngine.swift:208-241`

The consolidation loop processes duplicate pairs sequentially. If chunk A appears in pairs (A,B) and (A,C), only the first pair is processed — the second is skipped via `processedChunkIDs.contains`. This is intentional greedy behavior, but it means a single consolidation pass may leave undetected duplicates. There is no mechanism to re-run until convergence.

### 2.6 CoreML Model Loading Swallows Errors — **Major**

**File**: `Sources/ContextCore/EmbeddingProviders.swift:10-23`

```swift
func embed(_ text: String) async throws -> [Float] {
#if targetEnvironment(simulator)
    return Self.deterministicVector(for: text, dimension: dimension)
#else
    do {
        let model = try Self.loadModel()
        // ...
    } catch {
        return Self.deterministicVector(for: text, dimension: dimension)  // ← silent fallback
    }
#endif
}
```

On device, if the CoreML model fails to load (missing from bundle, compilation error, Neural Engine unavailable), the provider **silently falls back to deterministic hash vectors**. These are pseudo-random and semantically meaningless. The caller has no indication that embeddings are degraded. This can silently destroy retrieval quality in production with no visible error.

### 2.7 Checkpoint Non-Atomic Write Race — **Minor**

**File**: `Sources/ContextCore/AgentContext.swift:488-494`

```swift
let tempURL = parent.appendingPathComponent(".\(url.lastPathComponent).tmp-\(UUID().uuidString)")
try data.write(to: tempURL, options: .atomic)
if fileManager.fileExists(atPath: url.path) {
    try fileManager.removeItem(at: url)  // ← window where file is deleted but not replaced
}
try fileManager.moveItem(at: tempURL, to: url)
```

Between `removeItem` and `moveItem`, there is a window where the checkpoint file does not exist. A crash during this window loses the checkpoint. The `data.write(to: tempURL, options: .atomic)` is already atomic to the temp file, but the subsequent remove+move is not. On APFS, `moveItem` to an existing destination would fail — but the code removes first. Consider using `replaceItemAt(_:withItemAt:)` for a single atomic operation.

---

## 3. Architecture & Design Gaps

### 3.1 Tight Coupling Between AgentContext and Concrete Stores — **Minor**

`AgentContext` directly instantiates `EpisodicStore()`, `SemanticStore()`, and `ProceduralStore()` in its `init`. There is no protocol abstraction for the top-level store layer (distinct from `ConsolidationEpisodicStore`). This makes it impossible to:
- Inject a persistent-backed store (SQLite, Core Data) without modifying AgentContext.
- Test AgentContext with mock stores.

The consolidation protocols (`ConsolidationEpisodicStore`, `ConsolidationSemanticStore`) exist but cover only the consolidation slice of store behavior.

### 3.2 Duplicate Cosine Similarity Implementations — **Minor**

Cosine similarity is implemented independently in:
1. `AgentContext.cosineSimilarity` (line 697)
2. `SemanticStore.cosineSimilarity` (line 178)
3. `CPUReference.cosineSimilarity` (line 226)

All three are identical scalar loops. This should be a single shared utility.

### 3.3 Duplicate Sentence Splitting — **Minor**

Sentence splitting via `NLTokenizer(unit: .sentence)` is implemented in:
1. `ExtractiveFallbackDelegate.compress` (line 30-88)
2. `ExtractiveFallbackDelegate.extractFacts` (line 94-108)
3. `CompressionEngine.rankSentences` (line 50-112 — uses same pattern)

### 3.4 Type Re-export Architecture — **Good**

The `ContextCore` module re-exports all `ContextCoreTypes` types via typealiases. This is a clean pattern that maintains a single import for consumers while keeping the dependency graph layered. No issues here.

### 3.5 ContextConfiguration Lacks Validation — **Minor**

`ContextConfiguration` accepts arbitrary values with no validation:
- `maxTokens` can be 0 or negative
- `tokenBudgetSafetyMargin` can be > 1.0 (yielding negative effective budget)
- `relevanceWeight` + `centralityWeight` can sum to any value
- `consolidationThreshold` can be 0 (triggering consolidation on every insertion)

While `buildWindow` has runtime guards, invalid configs create confusing errors far from the configuration site.

---

## 4. Concurrency & Safety

### 4.1 ConsolidationScheduler Fire-and-Forget Task — **Major**

**File**: `Sources/ContextCoreEngine/ConsolidationEngine.swift:767-779`

```swift
Task.detached(priority: .background) {
    do {
        let result = try await engine.consolidate(...)
        await self.finish(result: result)
    } catch {
        await self.finish(result: nil)
    }
}
```

The detached task captures `self` (the scheduler) and `engine` strongly. If the `AgentContext` that owns the scheduler is deallocated, the detached task continues running — performing GPU work, store mutations, and ANN index updates on deallocated or replaced stores. There is no `Task` handle stored, so cancellation is impossible.

**Fix**: Store the `Task` handle, cancel it in a `deinit` or explicit shutdown method, and check `Task.isCancelled` within the consolidation loop.

### 4.2 @unchecked Sendable on Metal Buffers — **Minor**

**File**: `Sources/ContextCoreEngine/ScoringEngine.swift:7`

```swift
package final class PreparedScoringInputs: @unchecked Sendable {
    fileprivate let queryBuffer: MTLBuffer
    // ...
}
```

`MTLBuffer` is not inherently thread-safe. The `@unchecked Sendable` annotation suppresses compiler verification. In practice, these buffers are only accessed within the `ScoringEngine` actor, but the type can escape the actor boundary. The risk is contained but the annotation weakens compiler guarantees.

### 4.3 OSAllocatedUnfairLock for Stats — **Good**

**File**: `Sources/ContextCore/AgentContext.swift:37`

```swift
private nonisolated let statsLock = OSAllocatedUnfairLock(initialState: ContextStats())
```

Using `OSAllocatedUnfairLock` for the `nonisolated` stats property is correct and efficient. The lock protects a value type (`ContextStats`) with `withLock` closures. This avoids actor hop overhead for frequent stats reads.

### 4.4 Warmup Task Fire-and-Forget — **Minor**

**File**: `Sources/ContextCore/AgentContext.swift:96-98`

```swift
Task.detached(priority: .background) {
    _ = try? await provider.embed("warmup")
}
```

The warmup embedding task is fire-and-forget. If the embedding provider is slow or fails, this task runs indefinitely with no tracking. Low risk but inconsistent with structured concurrency principles.

### 4.5 Actor Re-entrancy in buildWindow — **Observation**

`buildWindow` calls multiple `async` methods on child actors (`scoringEngine`, `attentionEngine`, `episodicStore`, etc.). While awaiting these calls, the `AgentContext` actor is suspended and can process other messages (e.g., `append`, `remember`). This means stores can be mutated between the `allChunks()` read and the scoring step. The impact is benign (slightly stale data) but worth documenting.

---

## 5. Performance Bottlenecks

### 5.1 SemanticStore O(n) Deduplication — **Blocker**

See §2.1. Every `upsert` performs a full linear scan of all stored chunks. At 5,000 semantic facts with 384-dimension embeddings, this is ~1.9M floating-point multiplications per upsert.

### 5.2 EmbeddingCache O(n) LRU Maintenance — **Minor**

**File**: `Sources/ContextCoreEngine/EmbeddingCache.swift:58-68`

```swift
private func touch(_ digest: String) {
    guard let index = lruOrder.firstIndex(of: digest) else { ... }  // O(n)
    lruOrder.remove(at: index)  // O(n)
    lruOrder.append(digest)
}
```

Every cache hit triggers two O(n) operations on the LRU array. With the default capacity of 512 entries, this averages ~256 comparisons + ~256 element shifts per hit. For a hot cache in a tight loop, this adds measurable overhead. A `Dictionary<String, LinkedListNode>` would provide O(1) LRU updates.

### 5.3 SHA-256 Hashing for Cache Keys — **Minor**

**File**: `Sources/ContextCoreEngine/EmbeddingCache.swift:70-73`

```swift
private static func sha256Hex(_ key: String) -> String {
    let digest = SHA256.hash(data: Data(key.utf8))
    return digest.map { String(format: "%02x", $0) }.joined()
}
```

Every cache `get`/`set` computes SHA-256 and creates 32 intermediate `String` objects for hex encoding. For short keys (typical embedding inputs), a simpler hash (FNV-1a, or just using the String directly) would be far cheaper.

### 5.4 Double-Copy of Embeddings to GPU — **Minor**

Across all engine files, the pattern is:
```swift
let flattened = embeddings.flatMap { $0 }   // Copy 1: Array<Array<Float>> → Array<Float>
device.makeBuffer(bytes: flattened, ...)     // Copy 2: Array<Float> → MTLBuffer
```

For large embedding sets, this doubles peak memory usage temporarily. A single-pass approach writing directly into a pre-allocated buffer would halve memory.

### 5.5 allChunks() Sorting on Every buildWindow — **Minor**

**Files**: `EpisodicStore.allChunks()`, `SemanticStore.allChunks()`

Both stores sort all chunks by `createdAt` on every call. During `buildWindow`, both are called. For 10,000 episodic chunks, this is O(n log n) work that could be avoided by maintaining an ordered data structure or caching the sorted order.

### 5.6 N² Pairwise Similarity Threshold — **Minor**

**File**: `Sources/ContextCoreEngine/ConsolidationEngine.swift:127`

The simple (non-tiled) pairwise similarity path is used for n ≤ 2048. At n=2048, this allocates a 2048×2048×4 = 16MB dense matrix in addition to GPU buffers. The threshold could be lowered to ~512 for better memory behavior.

---

## 6. Security Risks

### 6.1 Metal Shader Silent Data Loss — **Major**

**File**: `Sources/ContextCoreShaders/Shaders/Consolidation.metal:63-66`

```metal
uint idx = atomic_fetch_add_explicit(candidateCount, 1u, memory_order_relaxed);
if (idx < maxCandidates) {
    candidates[idx] = uint2(i, j);
}
// No else — pair is silently lost
```

When the candidate buffer overflows, merge candidate pairs are silently discarded. The caller receives a truncated list with no indication of data loss. This can cause consolidation to miss genuine duplicates, leading to unbounded memory growth.

**Fix**: Return the final `candidateCount` value to the CPU and compare against `maxCandidates`. If overflow occurred, increase buffer and re-run.

### 6.2 Antipodal Test Division by Zero — **Minor**

**File**: `Sources/ContextCoreShaders/Shaders/Consolidation.metal:139`

```metal
fractions[gid] = float(signDiffCount) / float(dim);
```

If `dim == 0`, this produces NaN/Inf. The CPU side should validate `dim > 0` before dispatch.

### 6.3 Custom Format Template Injection — **Minor**

**File**: `Sources/ContextCore/ContextWindow.swift:172-179`

```swift
case .custom(let template):
    return chunks.map { chunk in
        template
            .replacingOccurrences(of: "{role}", with: chunk.role.rawValue)
            .replacingOccurrences(of: "{content}", with: chunk.content)
    }
```

User-controlled `chunk.content` is injected into the template without sanitization. If the template is used in a context where `{role}` or `{content}` substitutions have syntactic meaning (e.g., HTML, XML, or prompt injection), content could escape the intended structure. This is a design-level concern — the library doesn't know the downstream format.

### 6.4 Checkpoint File Path Not Sanitized — **Minor**

**File**: `Sources/ContextCore/AgentContext.swift:482-494`

The `checkpoint(to:)` method accepts an arbitrary `URL` and creates directories at the parent path. It does not validate that the URL points to a safe location. A malicious or misconfigured URL could write to sensitive paths. This is standard for a library API, but worth noting.

---

## 7. Testing Review

### 7.1 Concurrency Testing — **Blocker**

**Coverage**: Near zero.

The only concurrency-adjacent test is `ContextStatsTests.statsRapidUpdates` which spawns 100 concurrent tasks and checks `stats.episodicCount >= 1 && <= 100` — an extremely loose assertion that cannot detect races.

**Missing tests**:
- Concurrent `append` + `buildWindow`
- Concurrent `consolidate` + `append`
- Concurrent `remember` + `recall`
- Cancellation of in-flight `buildWindow`
- `ConsolidationScheduler` task lifecycle (start, deduplicate, dealloc)
- Actor re-entrancy effects

### 7.2 Failure Path Coverage — **Major gap**

**Tested**:
- Nil embedding → `embeddingFailed` ✓
- Corrupt checkpoint JSON ✓
- Missing checkpoint file ✓
- Session not started ✓
- Dimension mismatch ✓

**Not tested**:
- CoreML model loading failure (swallowed silently — see §2.6)
- Metal device unavailable (init throws, but never tested)
- Metal command buffer failure
- Disk full during checkpoint
- Embedding provider timeout / hanging
- ANN index corruption

### 7.3 Flaky Test Patterns — **Minor**

| Test | File | Risk |
|------|------|------|
| `autoConsolidation` | AgentContextTests:141 | 50ms polling, 4s timeout — timing-dependent |
| `schedulerNonBlocking` | ConsolidationTests:376 | Asserts < 10ms elapsed — fails on slow CI |
| `oneHalfLife` | RecencyTests:22 | `abs(weights[0] - 0.5) < 1e-3` — architecture-dependent FP variance |
| `centralityParity` | AttentionTests:92 | `maxError < 1e-4` — tight GPU/CPU tolerance |

### 7.4 Missing Edge Case Coverage — **Minor**

- Empty string content in chunks/turns
- Unicode/emoji in content (tokenizer behavior)
- Zero-dimension embeddings
- maxTokens = 1 (minimal budget)
- Thousands of chunks (no load/stress tests)
- Duplicate UUIDs in stores

### 7.5 Property Testing Opportunities

- **Invariant**: `ContextWindow.totalTokens <= budget` (for any input)
- **Invariant**: `buildWindow` output is deterministic for identical inputs
- **Invariant**: Compression never increases token count
- **Invariant**: Deduplication is idempotent
- **Roundtrip**: `checkpoint(to:)` → `load(from:)` preserves all state

---

## 8. Refactoring Opportunities

### 8.1 Extract Shared Cosine Similarity — **Easy win**

Three identical implementations (see §3.2). Extract to a single `package` utility function in `ContextCoreEngine` or `ContextCoreTypes`.

### 8.2 Extract Sentence Splitting — **Easy win**

Three identical `NLTokenizer`-based sentence splitters (see §3.3). Extract to a shared internal helper.

### 8.3 Protocol-ize Top-Level Stores — **Medium effort**

Extract `EpisodicStore`, `SemanticStore`, `ProceduralStore` interfaces behind protocols. This enables:
- Testing `AgentContext` with mock stores
- Persistent storage backends
- Migration strategies

### 8.4 Replace EmbeddingCache LRU with O(1) Structure — **Easy win**

Replace the `[String]` LRU array with a `Dictionary + doubly-linked list` pattern for O(1) cache operations.

### 8.5 Improve ContextConfiguration Validation — **Easy win**

Add a `validate() throws` method or use `init` validation:
```swift
guard maxTokens > 0 else { throw ... }
guard (0...1).contains(tokenBudgetSafetyMargin) else { throw ... }
```

### 8.6 Rename `scoreWindowForEviction` — **Naming clarity**

The method name suggests it scores *for* eviction, but it's used in `buildWindow` for *retention* reranking. The caller inverts the score (1.0 - normalized). Consider renaming to `computeAttentionScores` or similar.

### 8.7 Make `ExtractiveFallbackDelegate.extractFacts` Internal or Remove — **Dead code**

This public method appears to have no callers and duplicates private `splitSentences` logic.

---

## Appendix A: File-Level Summary

| File | Lines | Issues | Severity |
|------|-------|--------|----------|
| `AgentContext.swift` | 719 | Buffer calc redundancy, procedural embedding reuse, checkpoint race | Major |
| `WindowPacker.swift` | 297 | Clean implementation | — |
| `ContextWindow.swift` | 183 | Template injection surface | Minor |
| `ContextConfiguration.swift` | 109 | No validation | Minor |
| `EpisodicStore.swift` | 174 | Clean | — |
| `SemanticStore.swift` | 199 | O(n) deduplication scan | Blocker |
| `ProceduralStore.swift` | 68 | Clean | — |
| `ProgressiveCompressor.swift` | 194 | Clean | — |
| `EmbeddingProviders.swift` | 218 | Silent CoreML fallback | Major |
| `ChunkOrderer.swift` | 89 | Clean | — |
| `SessionStore.swift` | 51 | Clean | — |
| `ContextCheckpoint.swift` | 132 | Clean | — |
| `ContextStats.swift` | 52 | Clean | — |
| `AttentionEngine.swift` | ~187 | Threadgroup waste | Minor |
| `ScoringEngine.swift` | ~472 | @unchecked Sendable | Minor |
| `CompressionEngine.swift` | ~214 | Clean | — |
| `ConsolidationEngine.swift` | ~808 | Task.detached lifecycle, partial processing | Major |
| `CPUReference.swift` | ~263 | precondition in public API | Major |
| `EmbeddingCache.swift` | ~74 | O(n) LRU, SHA256 overhead | Minor |
| `MetalContext.swift` | ~76 | Clean | — |
| `ExtractiveFallbackDelegate.swift` | ~136 | Dead public method | Minor |
| Metal shaders (5 files) | ~150 | Buffer overflow silence, div-by-zero | Major |

## Appendix B: Dependency Analysis

| Dependency | Version | Risk |
|-----------|---------|------|
| `MetalANNS` | ≥ 0.1.2 | First-party (same author). Pinned to minor version. ANN correctness not audited. |
| `swift-docc-plugin` | ≥ 1.4.0 | Build-time only. No runtime risk. |

The dependency footprint is minimal and well-scoped. No transitive dependency concerns.

## Appendix C: Build Configuration

- **Swift tools version**: 6.2 (cutting-edge — verify CI toolchain availability)
- **Framework links**: Metal, CoreML, Accelerate — all Apple system frameworks
- **Resource processing**: `.process("Shaders")` for Metal, `.process("Resources")` for CoreML model
- **Missing**: No `.metallib` pre-compilation step. Shaders compile at runtime as fallback, adding latency on first use.
- **Missing**: No CI configuration file found in repository.

---

## Verdict

ContextCore demonstrates strong architectural taste: clean actor boundaries, progressive compression, attention-aware reranking, and GPU-accelerated scoring. The Swift 6.2 concurrency model is applied correctly in the common path.

The three release blockers — SemanticStore linear scan, CPUReference precondition abuse, and missing concurrency tests — must be resolved before production deployment. The silent CoreML fallback (§2.6) is a high-risk silent correctness bug that could go undetected for months.

After addressing blockers and major issues, this library is well-positioned for production use.
