---
sidebar_position: 1
title: "Benchmarks"
---

# Benchmarks

All benchmarks measured on an M2 Max running macOS 26.0, with embedding dimension 384.

## buildWindow Latency

Target: p99 < 20ms for 500 turns.

| Turns | Budget (tokens) | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|---|
| 10 | 2048 | 1.57 | 3.62 | 3.87 |
| 10 | 4096 | 1.37 | 3.05 | 3.42 |
| 50 | 4096 | 1.63 | 3.28 | 4.13 |
| 200 | 4096 | 2.23 | 3.46 | 3.57 |
| 500 | 4096 | 4.03 | 5.93 | 6.54 |

At 500 turns, the p99 latency of 6.54ms is well within the 20ms target. Latency scales sub-linearly due to GPU-accelerated scoring and HNSW index pruning.

## Consolidation Latency

Target: p99 < 500ms for 2000 chunks.

| Chunks | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|
| 100 | 0.93 | 2.06 | 2.06 |
| 500 | 3.72 | 5.04 | 5.04 |
| 2000 | 18.76 | 19.71 | 19.71 |

Consolidation at 2000 chunks completes in under 20ms, far below the 500ms target.

## Memory Footprint

Measured during a 500-turn session with embedding dimension 384.

| Component | Size |
|---|---|
| Episodic HNSW index | ~0.9 MB |
| Semantic HNSW index | ~0.1 MB |
| Scoring buffers (per call) | ~0.01 MB |
| Total GPU memory | ~1 MB |

## Methodology

- Each benchmark runs 100 iterations after a 10-iteration warmup.
- Latency percentiles are computed from wall-clock time using `ContinuousClock`.
- Memory footprint is measured via Metal resource allocation tracking.
- Embedding computation time is excluded from buildWindow measurements (embeddings are pre-computed).
