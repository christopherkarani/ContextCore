# Phase 2: Metal Scoring Engine — Task List

## 2.1 Relevance Scoring Kernel
- [ ] **TEST FIRST**: Write `ScoringTests.swift`
  - [ ] Test: CPU reference — 500 chunks dim=384, compute cosine similarity + recency blend via Accelerate vDSP
  - [ ] Test: GPU `relevance_score` kernel — same 500 chunks, compare scores vs CPU reference, max absolute error < 1e-4
  - [ ] Test: GPU `topk_indices` kernel — 500 scores, k=10, verify returned indices match the 10 highest scores
  - [ ] Test: configurable weights — pass (0.5, 0.5) instead of (0.7, 0.3), verify scores differ from default
  - [ ] Test: ScoringEngine.scoreChunks — returns tuples sorted by score descending
  - [ ] Test: single chunk edge case — n=1, verify score is valid
  - [ ] Test: zero recency weight — all recencyWeights = 0.0, verify score equals pure cosine similarity * relevanceWeight
- [ ] Write `Relevance.metal`
  - [ ] `relevance_score` kernel: query[dim], chunks[n*dim], recencyWeights[n], weights(float2) → scores[n]
  - [ ] `topk_indices` kernel: scores[n], k → indices[k] (parallel reduction or single-pass sort for n <= 4096)
- [ ] Write CPU reference implementation using Accelerate `vDSP_dotpr` for cosine similarity
- [ ] Write `ScoringEngine` actor in `ContextCoreEngine/ScoringEngine.swift`
  - [ ] Metal pipeline state setup (load library, create compute pipeline)
  - [ ] Buffer management (query, chunks, recencyWeights, scores, indices)
  - [ ] `func scoreChunks(query:chunks:recencyWeights:) async -> [(chunk: MemoryChunk, score: Float)]`
- [ ] Run tests — all green
- [ ] Commit: `feat(phase2): 2.1 — Relevance scoring kernel with CPU reference validation`

## 2.2 Recency Decay Kernel
- [ ] **TEST FIRST**: Add to `ScoringTests.swift` or create `RecencyTests.swift`
  - [ ] Test: weight = 1.0 for current timestamp (age = 0)
  - [ ] Test: weight ≈ 0.5 (within 1e-3) for timestamp exactly one half-life ago
  - [ ] Test: weight ≈ 0.25 (within 1e-3) for timestamp exactly two half-lives ago
  - [ ] Test: weight ≈ 0.0 (< 0.01) for timestamp 10 half-lives ago
  - [ ] Test: different half-life values — 7 days (episodic) vs 90 days (semantic) produce different weights for same age
  - [ ] Test: n=1000 timestamps, verify all weights in [0, 1] range
- [ ] Write `Recency.metal`
  - [ ] `compute_recency_weights` kernel: timestamps[n], currentTime, halfLifeSeconds → weights[n]
  - [ ] Formula: `exp(-ln(2) * (currentTime - timestamps[i]) / halfLifeSeconds)`
- [ ] Write Swift wrapper in `ScoringEngine` or dedicated `RecencyEngine`
  - [ ] `func computeRecencyWeights(timestamps: [Date], halfLife: TimeInterval) async -> [Float]`
  - [ ] Convert Date array to Float32 Unix timestamps on CPU before dispatch
- [ ] Run tests — all green
- [ ] Commit: `feat(phase2): 2.2 — Recency decay kernel with exponential half-life`

## 2.3 Attention Approximation Kernel
- [ ] **TEST FIRST**: Write `AttentionTests.swift`
  - [ ] Test: `token_centrality` — 50 chunks, verify chunk with highest mean similarity to others gets highest centrality
  - [ ] Test: `token_centrality` — one chunk that is orthogonal to all others gets lowest centrality
  - [ ] Test: `cross_attention_score` — chunk highly relevant to query AND central scores highest
  - [ ] Test: `cross_attention_score` — filler/generic chunk scores lower than task-relevant chunk
  - [ ] Test: configurable weights — pass (0.8, 0.2) instead of (0.6, 0.4), verify scores change
  - [ ] Test: n=1 chunk edge case — centrality should be 0.0 (no other chunks to compare)
  - [ ] Test: CPU vs GPU parity — 50 chunks, max absolute error < 1e-4
- [ ] Write `Attention.metal`
  - [ ] `token_centrality` kernel: tokenEmbeddings[n*dim] → centrality[n]
    - [ ] Use threadgroup shared memory for tiled dot products
    - [ ] Mean cosine similarity to all other chunks j ≠ i
  - [ ] `cross_attention_score` kernel: taskQuery[dim], windowEmbeddings[n*dim], centrality[n], weights(float2) → evictionScores[n]
    - [ ] Score = relevance * weight.x + centrality * weight.y
- [ ] Write Swift wrapper in `ContextCoreEngine/AttentionEngine.swift`
  - [ ] `func computeCentrality(chunks: [[Float]]) async -> [Float]`
  - [ ] `func scoreWindowForEviction(taskQuery: [Float], windowChunks: [ContextChunk]) async -> [(chunk: ContextChunk, evictionScore: Float)]`
- [ ] Run tests — all green
- [ ] Commit: `feat(phase2): 2.3 — Attention approximation with centrality scoring`

## 2.4 Compression Candidate Kernel
- [ ] **TEST FIRST**: Write `CompressionScoringTests.swift`
  - [ ] Test: 5 sentences, one clearly off-topic — off-topic sentence ranks last in importance
  - [ ] Test: sentence identical to chunk embedding gets highest importance score
  - [ ] Test: all sentences equally similar to chunk — importance scores are approximately equal
  - [ ] Test: NLTokenizer sentence splitting — verify correct sentence count
  - [ ] Test: CPU vs GPU parity — max absolute error < 1e-4
- [ ] Write `Compression.metal`
  - [ ] `sentence_importance` kernel: sentenceEmbeddings[m*dim], chunkQuery[dim] → importance[m]
  - [ ] Score = cosine similarity of each sentence embedding to chunk-level embedding
- [ ] Write Swift wrapper in `ContextCoreEngine/CompressionEngine.swift` (partial — full engine in Phase 5)
  - [ ] `func rankSentences(in chunk: String, query: String) async -> [(sentence: String, importance: Float)]`
  - [ ] Split chunk into sentences using `NLTokenizer` with `.sentence` unit
  - [ ] Embed each sentence via `EmbeddingProvider`
  - [ ] Dispatch Metal kernel
  - [ ] Return sorted by importance descending
- [ ] Run tests — all green
- [ ] Commit: `feat(phase2): 2.4 — Sentence importance scoring for compression candidates`

## Final Verification
- [ ] Run full `swift build` — zero errors, zero warnings
- [ ] Run full `swift test` — all Phase 1 + Phase 2 tests green
- [ ] Verify all Metal kernels compile on macOS (real device)
- [ ] Verify simulator fallback path compiles (CPU reference for tests)
- [ ] 4 clean commits in git log for Phase 2
