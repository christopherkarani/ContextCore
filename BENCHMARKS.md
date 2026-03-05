# ContextCore Benchmarks

> Profile: `full`.

Measured on chriskarani.local, Version 26.0 (Build 25A354), 2026-03-05T22:24:47Z.

## buildWindow Latency

| Turns | Budget | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|---|
| 10 | 2048 | 4.70 | 7.66 | 8.10 |
| 10 | 4096 | 4.14 | 7.19 | 7.30 |
| 10 | 8192 | 3.83 | 6.79 | 7.07 |
| 50 | 2048 | 4.07 | 6.72 | 11.35 |
| 50 | 4096 | 3.93 | 6.48 | 7.20 |
| 50 | 8192 | 3.84 | 6.67 | 6.96 |
| 200 | 2048 | 1.82 | 6.03 | 6.52 |
| 200 | 4096 | 5.34 | 7.89 | 7.95 |
| 200 | 8192 | 6.99 | 8.04 | 8.22 |
| 500 | 2048 | 6.82 | 7.98 | 8.21 |
| 500 | 4096 | 5.16 | 6.91 | 7.38 |
| 500 | 8192 | 7.45 | 8.57 | 8.93 |

**Target**: p99 < 20ms for 500 turns on M2.

## Consolidation Latency

| Chunks | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|
| 100 | 2.57 | 4.38 | 4.38 |
| 500 | 8.19 | 8.55 | 8.55 |
| 2000 | 33.61 | 39.80 | 39.80 |

**Target**: p99 < 500ms for 2000 chunks on M2.

## Metal vs CPU Scoring

Math-only isolates score computation over pre-flattened inputs. End-to-end includes public API validation, flattening, zip, and final sort.

### Math-only

| n | GPU p50 | GPU p99 | GPU throughput | CPU p50 | CPU p99 | CPU throughput | Speedup |
|---|---|---|---|---|---|---|---|
| 100 | 365.21us | 4.18ms | 273.82K chunks/s | 3.96us | 5.12us | 25.27M chunks/s | 0.01x |
| 500 | 387.25us | 3.87ms | 1.29M chunks/s | 19.12us | 29.88us | 26.14M chunks/s | 0.05x |
| 2000 | 839.96us | 4.23ms | 2.38M chunks/s | 76.67us | 101.29us | 26.09M chunks/s | 0.09x |
| 10000 | 1.29ms | 3.70ms | 7.74M chunks/s | 370.54us | 372.04us | 26.99M chunks/s | 0.29x |
| 50000 | 2.76ms | 5.53ms | 18.12M chunks/s | 2.27ms | 2.28ms | 22.06M chunks/s | 0.82x |

### End-to-end

| n | GPU p50 | GPU p99 | GPU throughput | CPU p50 | CPU p99 | CPU throughput | Speedup |
|---|---|---|---|---|---|---|---|
| 100 | 531.46us | 3.99ms | 188.16K chunks/s | 178.92us | 224.46us | 558.92K chunks/s | 0.34x |
| 500 | 2.42ms | 4.99ms | 206.76K chunks/s | 854.67us | 1.15ms | 585.02K chunks/s | 0.35x |
| 2000 | 6.64ms | 8.38ms | 301.01K chunks/s | 3.68ms | 4.14ms | 543.02K chunks/s | 0.55x |
| 10000 | 18.66ms | 20.82ms | 535.84K chunks/s | 16.34ms | 17.03ms | 612.09K chunks/s | 0.88x |
| 50000 | 116.48ms | 122.15ms | 429.26K chunks/s | 118.36ms | 124.52ms | 422.44K chunks/s | 1.02x |

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
- Scoring buffers: reused shared Metal buffers sized to the largest active scoring workload
- **Total GPU memory: dominated by active embedding and scoring workload size**