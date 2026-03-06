---
sidebar_position: 1
title: "GPU Scoring"
---

# GPU Scoring

ContextCore accelerates context scoring on-device using Apple Metal compute shaders. All scoring runs on the GPU, keeping latency low even at high chunk counts.

## Metal Compute Shaders

ContextCore compiles five Metal compute shaders at initialization:

| Shader | Purpose |
|---|---|
| `relevance_score` | Cosine similarity between query and chunk embeddings |
| `topk_indices` | Extracts top-k indices from a scored buffer |
| `compute_recency_weights` | Exponential decay weights based on chunk age |
| Attention scoring | Computes attention centrality for eviction decisions |
| Compression scoring | Scores chunks for compression candidacy |

## ScoringEngine

`ScoringEngine` is a Swift actor that owns the Metal pipeline state objects and dispatches GPU work.

```swift
let engine = ScoringEngine(metalContext: context)

let scores = try await engine.scoreChunks(
    query: queryEmbedding,
    chunks: candidates,
    recencyWeights: weights,
    relevanceWeight: 0.7,
    recencyWeight: 0.3
)
```

### Relevance Scoring

Each chunk embedding is compared to the query embedding via cosine similarity on the GPU. The final score combines relevance and recency:

```
score = relevanceWeight * cosineSimilarity + recencyWeight * recencyWeight
```

### Recency Weights

Recency weights follow an exponential decay curve with a configurable half-life (default: 7 days for episodic, 90 days for semantic). The `compute_recency_weights` shader computes these in parallel on the GPU.

## AttentionEngine

`AttentionEngine` computes attention centrality scores across chunks. These scores inform eviction decisions during consolidation — chunks that are frequently "attended to" by other chunks receive higher centrality and are retained longer.

## MetalContext

`MetalContext` manages the shared Metal device, command queue, and shader library loading from the framework bundle. It is created once and shared across all engine actors.

```swift
let metalContext = try MetalContext()
// metalContext.device — MTLDevice
// metalContext.commandQueue — MTLCommandQueue
```

## Thread Dispatch

Metal compute dispatches use helper functions to calculate optimal `threadsPerThreadgroup` and `threadgroups` values based on the pipeline's `maxTotalThreadsPerThreadgroup` and the input data size. This ensures full GPU utilization across different hardware.

## Performance

GPU scoring at 2000 chunks completes in approximately 5ms, with the total scoring pipeline (including recency weights and top-k extraction) at ~6.5ms p99 on Apple Silicon.
