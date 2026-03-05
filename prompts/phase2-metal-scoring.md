# Phase 2: Metal Scoring Engine

You are implementing Phase 2 of ContextCore. Phase 1 is complete — the package scaffold, data models (`Turn`, `MemoryChunk`, `MemoryType`), memory stores (`EpisodicStore`, `SemanticStore`, `ProceduralStore`), `EmbeddingProvider` protocol with CoreML backend and LRU cache, `TokenCounter`, and `ContextConfiguration` all exist and compile.

Phase 2 adds the Metal compute shaders that make ContextCore fast. You are writing 5 kernels across 4 `.metal` files plus their Swift actor wrappers and CPU reference implementations for test validation. After this phase, all scoring infrastructure is GPU-accelerated and verified against CPU baselines.

---

## What You Are Building

Four GPU subsystems, each a `.metal` file + Swift wrapper:

| Subsystem | Metal File | Kernels | Purpose |
|---|---|---|---|
| **Relevance** | `Relevance.metal` | `relevance_score`, `topk_indices` | Score memory chunks against current task |
| **Recency** | `Recency.metal` | `compute_recency_weights` | Exponential time-decay weights from timestamps |
| **Attention** | `Attention.metal` | `token_centrality`, `cross_attention_score` | Find central vs peripheral context for eviction |
| **Compression** | `Compression.metal` | `sentence_importance` | Rank sentences within a chunk for extractive compression |

Every kernel operates on `Float32` arrays. Thread dispatch is one-thread-per-chunk (or one-thread-per-sentence for compression). All kernels are verified against CPU reference implementations with max absolute error < 1e-4.

## Metal Development Rules

- All kernels live in `Sources/ContextCoreShaders/`.
- All Swift wrappers live in `Sources/ContextCoreEngine/` as actors.
- Every kernel function must include `[[kernel]]` attribute and use `uint gid [[thread_position_in_grid]]` for indexing.
- Every kernel must bounds-check `gid` against the element count before writing output.
- Configurable weight parameters are passed as `device float2*` or `constant float&` — never hardcoded in the shader.
- Shared memory usage must declare `threadgroup float shared_data[...]` with explicit size.
- Swift wrappers handle all `MTLBuffer` creation, pipeline state caching, and command buffer lifecycle.
- Pipeline states are created once at init and reused — never compiled per-call.
- Use `commandBuffer.addCompletedHandler` or `await` on the GPU work — never spin-wait.

## File Map (Phase 2 additions)

```
Sources/
  ContextCoreShaders/
    Relevance.metal          # NEW — relevance_score, topk_indices
    Recency.metal            # NEW — compute_recency_weights
    Attention.metal          # NEW — token_centrality, cross_attention_score
    Compression.metal        # NEW — sentence_importance
  ContextCoreEngine/
    ScoringEngine.swift      # NEW — relevance + recency dispatch
    AttentionEngine.swift    # NEW — centrality + eviction dispatch
    CompressionEngine.swift  # NEW (partial) — sentence ranking dispatch
    CPUReference.swift       # NEW — Accelerate-based reference impls for tests
Tests/
  ContextCoreTests/
    ScoringTests.swift       # NEW
    RecencyTests.swift       # NEW
    AttentionTests.swift     # NEW
    CompressionScoringTests.swift  # NEW
```

---

## Execution Plan

Work through these 4 sub-tasks in order. Each follows TDD:

1. Write CPU reference implementation (the ground truth).
2. Write the test (comparing GPU output to CPU reference).
3. Write the Metal kernel.
4. Write the Swift actor wrapper.
5. Run `swift build && swift test`.
6. Commit.

---

### 2.1 — Relevance Scoring

The core scoring kernel. Takes a task embedding and an array of chunk embeddings, outputs a blended relevance score per chunk.

#### Metal Kernels

**`relevance_score`** (`Relevance.metal`):

```metal
kernel void relevance_score(
    device const float* query        [[buffer(0)]],  // [dim]
    device const float* chunks       [[buffer(1)]],  // [n * dim]
    device const float* recencyWts   [[buffer(2)]],  // [n]
    device const float2* weights     [[buffer(3)]],  // single float2: (relevanceWt, recencyWt)
    device float* scores             [[buffer(4)]],  // [n] output
    constant uint& dim               [[buffer(5)]],
    constant uint& n                 [[buffer(6)]],
    uint gid                         [[thread_position_in_grid]]
)
```

Per-thread logic (one thread per chunk `gid`):
1. Bounds check: `if (gid >= n) return;`
2. Compute dot product of `query` and `chunks[gid * dim ... (gid+1) * dim - 1]`.
3. Compute L2 norms of both vectors.
4. Cosine similarity = dot / (norm_query * norm_chunk). Guard against zero norms.
5. `scores[gid] = cosine * weights->x + recencyWts[gid] * weights->y`

**`topk_indices`** (`Relevance.metal`):

```metal
kernel void topk_indices(
    device const float* scores   [[buffer(0)]],  // [n]
    device uint* indices         [[buffer(1)]],  // [k] output
    constant uint& n             [[buffer(2)]],
    constant uint& k             [[buffer(3)]],
    uint gid                     [[thread_position_in_grid]]
)
```

For n <= 4096 (typical on-device), use a single-pass approach:
- Thread 0 performs a linear scan, maintaining a min-heap of size k.
- Write the k indices to the output buffer.

For a more parallel approach: each threadgroup finds its local top-k via shared memory, then a final reduction pass merges. Choose the simpler approach first and optimize only if benchmarks warrant it.

#### CPU Reference

Implement in `CPUReference.swift` using Accelerate:

```swift
enum CPUReference {
    /// Cosine similarity + recency blend — ground truth for GPU validation
    static func relevanceScores(
        query: [Float],
        chunks: [[Float]],
        recencyWeights: [Float],
        relevanceWeight: Float = 0.7,
        recencyWeight: Float = 0.3
    ) -> [Float]
}
```

Use `vDSP_dotpr` for dot products and `vDSP_svesq` + `sqrt` for norms.

#### Swift Wrapper

```swift
public actor ScoringEngine {
    private let device: MTLDevice
    private let relevancePipeline: MTLComputePipelineState
    private let topkPipeline: MTLComputePipelineState

    public init(device: MTLDevice? = nil) throws

    /// Score chunks against a query with recency blending.
    /// Returns chunks paired with scores, sorted descending.
    public func scoreChunks(
        query: [Float],
        chunks: [MemoryChunk],
        recencyWeights: [Float],
        relevanceWeight: Float = 0.7,
        recencyWeight: Float = 0.3
    ) async throws -> [(chunk: MemoryChunk, score: Float)]

    /// Return indices of top-k highest scores.
    public func topKIndices(
        scores: [Float],
        k: Int
    ) async throws -> [Int]
}
```

Pipeline state creation happens in `init` — load the Metal library from bundle, find functions by name, create pipeline states. Cache them as stored properties.

#### Tests (`ScoringTests.swift`)

| # | Test | Setup | Assertion |
|---|---|---|---|
| 1 | CPU reference correctness | 10 chunks dim=384, known query, hand-verify one score | Score matches manual calculation within 1e-6 |
| 2 | GPU vs CPU parity | 500 random chunks dim=384, random query, random recency weights | Max absolute error < 1e-4 across all 500 scores |
| 3 | Top-k correctness | 500 scores, k=10 | Returned 10 indices point to the 10 highest scores |
| 4 | Custom weights | Same chunks, weights (0.5, 0.5) vs (0.7, 0.3) | Scores differ |
| 5 | Single chunk | n=1 | Returns valid score, no crash |
| 6 | Zero recency weights | All recencyWeights = 0.0, relevanceWeight = 0.7 | Score = cosine_similarity * 0.7 for each chunk |
| 7 | Sorted output | scoreChunks returns array | First element has highest score, last has lowest |

**Generate test data**: Use `Float.random(in: -1...1)` seeded with a fixed `RandomNumberGenerator` for reproducibility. L2-normalize all vectors after generation.

**Simulator fallback**: Gate GPU tests with `#if !targetEnvironment(simulator)`. On simulator, run only CPU reference tests.

**Commit:** `feat(phase2): 2.1 — Relevance scoring kernel with CPU reference validation`

---

### 2.2 — Recency Decay

Computes exponential decay weights from timestamps on GPU. Avoids CPU date arithmetic overhead at scale.

#### Metal Kernel

**`compute_recency_weights`** (`Recency.metal`):

```metal
kernel void compute_recency_weights(
    device const float* timestamps     [[buffer(0)]],  // [n] Unix timestamps (Float32)
    device float* weights              [[buffer(1)]],  // [n] output
    constant float& currentTime        [[buffer(2)]],
    constant float& halfLifeSeconds    [[buffer(3)]],
    constant uint& n                   [[buffer(4)]],
    uint gid                           [[thread_position_in_grid]]
)
```

Per-thread logic:
1. Bounds check: `if (gid >= n) return;`
2. `float age = currentTime - timestamps[gid];`
3. `weights[gid] = exp(-0.693147f * age / halfLifeSeconds);` (0.693147 = ln(2))
4. Clamp to [0, 1]: `weights[gid] = clamp(weights[gid], 0.0f, 1.0f);`

#### Swift Wrapper

Add to `ScoringEngine` or create a separate actor — your call based on cohesion:

```swift
/// Compute exponential decay weights for an array of timestamps.
/// halfLife is in seconds (e.g. 7 * 86400 for 7 days).
public func computeRecencyWeights(
    timestamps: [Date],
    halfLife: TimeInterval
) async throws -> [Float]
```

The wrapper converts `[Date]` to `[Float]` (Unix timestamps) on CPU before dispatching to GPU.

#### Tests (`RecencyTests.swift`)

| # | Test | Input | Assertion |
|---|---|---|---|
| 1 | Current timestamp | age = 0 seconds | weight == 1.0 (exact) |
| 2 | One half-life ago | age = halfLife | weight ≈ 0.5 (within 1e-3) |
| 3 | Two half-lives ago | age = 2 * halfLife | weight ≈ 0.25 (within 1e-3) |
| 4 | Ten half-lives ago | age = 10 * halfLife | weight < 0.01 |
| 5 | Episodic vs semantic | Same age, halfLife 7d vs 90d | 7d weight << 90d weight |
| 6 | Monotonicity | 100 timestamps spaced 1 hour apart | weights[i] > weights[i+1] for all i |
| 7 | Range check | 1000 random timestamps spanning 1 year | All weights in [0, 1] |
| 8 | GPU vs CPU parity | 1000 timestamps | Max absolute error < 1e-5 |

**CPU reference** (add to `CPUReference.swift`):

```swift
static func recencyWeights(
    timestamps: [Date],
    currentTime: Date,
    halfLife: TimeInterval
) -> [Float]
```

Use `vForce_exp` from Accelerate for the batch exponential.

**Commit:** `feat(phase2): 2.2 — Recency decay kernel with exponential half-life`

---

### 2.3 — Attention Approximation

Two kernels that estimate which context window chunks are "central" (attended to by many other chunks) vs "peripheral" (candidates for compression or eviction).

#### Mental Model

Think of the context window as a graph. Each chunk is a node. Edge weight = cosine similarity between chunks. **Centrality** = a chunk's average edge weight. High centrality = the chunk is topically connected to many other chunks = important to keep. Low centrality = the chunk is isolated = safe to compress or drop.

**Eviction score** combines centrality with task relevance. A chunk can be central but irrelevant to the current task — or relevant but peripheral. The eviction score balances both signals.

#### Metal Kernels

**`token_centrality`** (`Attention.metal`):

```metal
kernel void token_centrality(
    device const float* embeddings   [[buffer(0)]],  // [n * dim]
    device float* centrality         [[buffer(1)]],  // [n] output
    constant uint& dim               [[buffer(2)]],
    constant uint& n                 [[buffer(3)]],
    uint gid                         [[thread_position_in_grid]]
)
```

Per-thread logic (one thread per chunk `gid`):
1. Bounds check.
2. For each other chunk `j` where `j != gid`: compute cosine similarity between `embeddings[gid*dim]` and `embeddings[j*dim]`.
3. `centrality[gid]` = sum of similarities / (n - 1). This is the mean cosine similarity to all other chunks.

**Shared memory optimization**: For each threadgroup, load a tile of embeddings into `threadgroup float shared_tile[TILE_SIZE * DIM]`. Compute partial dot products within the tile, then accumulate across tiles. This avoids redundant global memory reads. Use `TILE_SIZE = 16` for dim=384 (fits 16 * 384 * 4 = 24KB in shared memory).

**`cross_attention_score`** (`Attention.metal`):

```metal
kernel void cross_attention_score(
    device const float* taskQuery      [[buffer(0)]],  // [dim]
    device const float* embeddings     [[buffer(1)]],  // [n * dim]
    device const float* centrality     [[buffer(2)]],  // [n] (precomputed)
    device const float2* weights       [[buffer(3)]],  // single float2: (relevanceWt, centralityWt)
    device float* evictionScores       [[buffer(4)]],  // [n] output
    constant uint& dim                 [[buffer(5)]],
    constant uint& n                   [[buffer(6)]],
    uint gid                           [[thread_position_in_grid]]
)
```

Per-thread logic:
1. Bounds check.
2. Compute cosine similarity between `taskQuery` and `embeddings[gid * dim]` → `relevance`.
3. `evictionScores[gid] = relevance * weights->x + centrality[gid] * weights->y`

Low eviction score = first to be compressed or dropped.

#### Swift Wrapper

```swift
public actor AttentionEngine {
    private let device: MTLDevice
    private let centralityPipeline: MTLComputePipelineState
    private let crossAttentionPipeline: MTLComputePipelineState

    public init(device: MTLDevice? = nil) throws

    /// Compute mean cosine similarity of each chunk to all others.
    public func computeCentrality(
        embeddings: [[Float]]
    ) async throws -> [Float]

    /// Score window chunks for eviction.
    /// Low score = evict first. High score = keep.
    public func scoreWindowForEviction(
        taskQuery: [Float],
        windowChunks: [ContextChunk],
        relevanceWeight: Float = 0.6,
        centralityWeight: Float = 0.4
    ) async throws -> [(chunk: ContextChunk, evictionScore: Float)]
}
```

Note: `ContextChunk` is defined in Phase 3. For now, define a minimal internal struct or use `MemoryChunk` with an added `embedding` field. Phase 3 will formalize the type.

#### Tests (`AttentionTests.swift`)

| # | Test | Setup | Assertion |
|---|---|---|---|
| 1 | Centrality — central chunk | 10 chunks: 9 similar to each other, 1 orthogonal outlier | Outlier has lowest centrality |
| 2 | Centrality — uniform | 10 identical chunks | All centrality scores ≈ 1.0 |
| 3 | Centrality — single chunk | n=1 | centrality = 0.0 (no peers) |
| 4 | Cross-attention — relevant + central | Query about "Swift concurrency". Chunks: 3 about Swift, 2 about cooking, 2 generic filler | Swift chunks score highest |
| 5 | Cross-attention — weights | Same data, weights (0.8, 0.2) vs (0.6, 0.4) | Scores differ, ranking may change |
| 6 | GPU vs CPU parity (centrality) | 50 random chunks dim=384 | Max absolute error < 1e-4 |
| 7 | GPU vs CPU parity (cross-attention) | 50 chunks + query | Max absolute error < 1e-4 |
| 8 | Eviction ordering | scoreWindowForEviction returns sorted | First element has lowest score (evict first) |

**CPU reference** (add to `CPUReference.swift`):

```swift
static func centrality(embeddings: [[Float]]) -> [Float]
static func crossAttentionScores(
    query: [Float],
    embeddings: [[Float]],
    centrality: [Float],
    relevanceWeight: Float,
    centralityWeight: Float
) -> [Float]
```

**Test data for semantic tests (4, 5)**: Use the `EmbeddingProvider` from Phase 1 to embed real sentences. This makes the test meaningful — random vectors won't exhibit semantic clustering.

```swift
let swiftChunks = [
    "Swift actors provide data race safety through isolation",
    "async/await simplifies concurrent code in Swift",
    "Sendable conformance prevents data races at compile time"
]
let cookingChunks = [
    "Preheat the oven to 350 degrees for best results",
    "Whisk the eggs until light and fluffy"
]
let fillerChunks = [
    "This is some generic placeholder text",
    "Another filler sentence with no specific topic"
]
let query = "How does Swift handle concurrency safely?"
```

**Commit:** `feat(phase2): 2.3 — Attention approximation with centrality and eviction scoring`

---

### 2.4 — Compression Candidate Scoring

Within a single chunk, rank sentences by importance so the compression engine (Phase 5) knows which to keep and which to drop.

#### Mental Model

A chunk's embedding represents its overall meaning. Each sentence within the chunk has its own embedding. Sentences whose embeddings are close to the chunk embedding are "representative" — they carry the chunk's core meaning. Sentences far from the chunk embedding are tangential — safe to drop during compression.

#### Metal Kernel

**`sentence_importance`** (`Compression.metal`):

```metal
kernel void sentence_importance(
    device const float* sentenceEmbs  [[buffer(0)]],  // [m * dim]
    device const float* chunkQuery    [[buffer(1)]],  // [dim]
    device float* importance          [[buffer(2)]],  // [m] output
    constant uint& dim                [[buffer(3)]],
    constant uint& m                  [[buffer(4)]],
    uint gid                          [[thread_position_in_grid]]
)
```

Per-thread logic (one thread per sentence `gid`):
1. Bounds check.
2. Compute cosine similarity between `sentenceEmbs[gid * dim]` and `chunkQuery`.
3. `importance[gid] = cosine_similarity`

#### Swift Wrapper

Add to `CompressionEngine.swift` (this file will grow in Phase 5):

```swift
public actor CompressionEngine {
    private let device: MTLDevice
    private let sentenceImportancePipeline: MTLComputePipelineState
    private let embeddingProvider: any EmbeddingProvider

    public init(
        device: MTLDevice? = nil,
        embeddingProvider: any EmbeddingProvider
    ) throws

    /// Rank sentences within a chunk by importance.
    /// Returns sentences paired with importance scores, sorted descending (most important first).
    public func rankSentences(
        in chunk: String,
        chunkEmbedding: [Float]
    ) async throws -> [(sentence: String, importance: Float)]
}
```

The wrapper does the preprocessing:
1. Split `chunk` into sentences using `NLTokenizer` with `.sentence` unit (import `NaturalLanguage`).
2. Embed each sentence via `embeddingProvider.embedBatch(_:)`.
3. Create Metal buffers: sentence embeddings (flattened), chunk embedding.
4. Dispatch `sentence_importance` kernel.
5. Read back importance scores.
6. Zip sentences with scores, sort descending, return.

#### Tests (`CompressionScoringTests.swift`)

| # | Test | Setup | Assertion |
|---|---|---|---|
| 1 | Off-topic sentence ranks last | Chunk: "Swift is a compiled language. It uses LLVM for optimization. It supports generics and protocols. The weather is nice today. Swift was created by Apple." | "The weather is nice today." has the lowest importance score |
| 2 | Representative sentence ranks first | Chunk about memory management. One sentence is a near-summary of the whole chunk. | That sentence has the highest importance |
| 3 | Equal sentences | 3 identical sentences | All importance scores within 1e-4 of each other |
| 4 | Sentence splitting | "First sentence. Second sentence. Third!" | Splits into exactly 3 sentences |
| 5 | GPU vs CPU parity | 10 sentences, dim=384 | Max absolute error < 1e-4 |
| 6 | Single sentence | Chunk with one sentence | Returns that sentence with importance = 1.0 (cosine of vector with itself) |

**CPU reference** (add to `CPUReference.swift`):

```swift
static func sentenceImportance(
    sentenceEmbeddings: [[Float]],
    chunkEmbedding: [Float]
) -> [Float]
```

**Commit:** `feat(phase2): 2.4 — Sentence importance scoring for compression candidates`

---

## Shared Infrastructure

### Metal Device & Library Loading

Create a shared utility (or use a pattern within each engine's init):

```swift
enum MetalContext {
    /// Returns the default Metal device. Throws on simulator.
    static func device() throws -> MTLDevice {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw ContextCoreError.metalDeviceUnavailable
        }
        return device
    }

    /// Loads the ContextCoreShaders Metal library from the bundle.
    static func library(device: MTLDevice) throws -> MTLLibrary {
        let bundle = Bundle.module  // SPM resource bundle
        guard let url = bundle.url(forResource: "default", withExtension: "metallib"),
              let library = try? device.makeLibrary(URL: url) else {
            // Fallback: compile from source
            let sources = // concatenate .metal source files
            return try device.makeLibrary(source: sources, options: nil)
        }
        return library
    }
}
```

### Buffer Helpers

Each engine will repeatedly create `MTLBuffer` from `[Float]`. Extract a helper:

```swift
extension MTLDevice {
    func makeBuffer(from array: [Float]) -> MTLBuffer? {
        array.withUnsafeBufferPointer { ptr in
            makeBuffer(bytes: ptr.baseAddress!, length: ptr.count * MemoryLayout<Float>.stride, options: .storageModeShared)
        }
    }
}
```

### Test Utilities

```swift
enum TestHelpers {
    /// Generate a reproducible random Float32 vector of given dimension, L2-normalized.
    static func randomVector(dim: Int, seed: UInt64) -> [Float]

    /// Generate n random vectors.
    static func randomVectors(n: Int, dim: Int, seed: UInt64) -> [[Float]]

    /// Max absolute difference between two Float arrays.
    static func maxAbsError(_ a: [Float], _ b: [Float]) -> Float
}
```

---

## Final Verification

After all 4 sub-tasks:

```bash
swift build 2>&1      # zero errors, zero warnings
swift test 2>&1       # all Phase 1 + Phase 2 tests green
```

Report: total test count, pass/fail, any GPU vs CPU error margins observed.

## Phase 2 Is Done When

- All 5 Metal kernels compile without warnings
- Every kernel has a CPU reference implementation in `CPUReference.swift`
- GPU vs CPU max absolute error < 1e-4 for all scoring kernels
- GPU vs CPU max absolute error < 1e-5 for recency kernel (simpler math, tighter tolerance)
- All Swift actor wrappers handle buffer lifecycle correctly (no leaks)
- Pipeline states are created once at init, not per-call
- Simulator tests run CPU-only path without crashing
- Phase 1 tests still pass (no regressions)
- 4 clean atomic commits in git history
