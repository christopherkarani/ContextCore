# ContextCore Benchmarks

> Profile: `full`.

Measured on chriskarani.local, Version 26.0 (Build 25A354), 2026-03-05T19:47:37Z.

## buildWindow Latency

| Turns | Budget | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|---|
| 10 | 2048 | 1.57 | 3.62 | 3.87 |
| 10 | 4096 | 1.37 | 3.05 | 3.42 |
| 10 | 8192 | 1.34 | 3.13 | 3.37 |
| 50 | 2048 | 1.60 | 3.46 | 4.03 |
| 50 | 4096 | 1.63 | 3.28 | 4.13 |
| 50 | 8192 | 1.60 | 3.39 | 4.43 |
| 200 | 2048 | 2.09 | 3.56 | 3.95 |
| 200 | 4096 | 2.23 | 3.46 | 3.57 |
| 200 | 8192 | 2.26 | 3.85 | 4.49 |
| 500 | 2048 | 3.70 | 5.30 | 5.62 |
| 500 | 4096 | 4.03 | 5.93 | 6.54 |
| 500 | 8192 | 3.45 | 4.83 | 4.92 |

**Target**: p99 < 20ms for 500 turns on M2.

## Consolidation Latency

| Chunks | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|
| 100 | 0.93 | 2.06 | 2.06 |
| 500 | 3.72 | 5.04 | 5.04 |
| 2000 | 18.76 | 19.71 | 19.71 |

**Target**: p99 < 500ms for 2000 chunks on M2.

## Metal vs CPU Scoring

| n | GPU p50 (ms) | CPU p50 (ms) | Speedup |
|---|---|---|---|
| 100 | 0.47 | 0.00 | 0.01x |
| 500 | 1.44 | 0.02 | 0.02x |
| 2000 | 5.06 | 0.09 | 0.02x |

## Recall Quality

| k | Precision@k |
|---|---|
| 3 | 0.333 |
| 5 | 0.200 |
| 8 | 0.125 |

## Memory Footprint

500-turn session, dim=384, degree=32:
- Episodic index: ~0.9 MB
- Semantic index: ~0.1 MB
- Scoring buffers: ~0.01 MB per call
- **Total GPU memory: ~1 MB**