# ContextCore Benchmarks

> Profile: `full`.

Measured on chriskarani.local, Version 26.0 (Build 25A354), 2026-03-05T22:45:03Z.

## buildWindow Latency

| Turns | Budget | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|---|
| 10 | 2048 | 1.75 | 2.67 | 3.67 |
| 10 | 4096 | 1.64 | 2.28 | 3.29 |
| 10 | 8192 | 1.67 | 2.90 | 3.52 |
| 50 | 2048 | 2.02 | 2.96 | 3.21 |
| 50 | 4096 | 1.97 | 2.78 | 3.08 |
| 50 | 8192 | 1.67 | 2.34 | 3.28 |
| 200 | 2048 | 2.47 | 3.24 | 3.48 |
| 200 | 4096 | 2.62 | 3.51 | 3.96 |
| 200 | 8192 | 2.58 | 3.47 | 3.81 |
| 500 | 2048 | 3.31 | 3.94 | 4.36 |
| 500 | 4096 | 3.35 | 4.27 | 4.89 |
| 500 | 8192 | 3.34 | 4.14 | 4.54 |

**Target**: p99 < 20ms for 500 turns on M2.

## Consolidation Latency

| Chunks | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|
| 100 | 0.65 | 0.95 | 0.95 |
| 500 | 1.59 | 1.92 | 1.92 |
| 2000 | 13.63 | 15.61 | 15.61 |

**Target**: p99 < 500ms for 2000 chunks on M2.

## Metal vs CPU Scoring

Math-only isolates score computation over pre-flattened inputs with resident GPU buffers. End-to-end includes public API validation, flattening, zip, and final sort.

### Math-only

| n | GPU p50 | GPU p99 | GPU throughput | CPU p50 | CPU p99 | CPU throughput | Speedup |
|---|---|---|---|---|---|---|---|
| 100 | 196.79us | 765.92us | 508.15K chunks/s | 3.88us | 4.00us | 25.81M chunks/s | 0.02x |
| 500 | 201.54us | 600.04us | 2.48M chunks/s | 19.21us | 21.67us | 26.03M chunks/s | 0.10x |
| 2000 | 222.33us | 799.25us | 9.00M chunks/s | 75.58us | 88.71us | 26.46M chunks/s | 0.34x |
| 10000 | 336.50us | 761.50us | 29.72M chunks/s | 385.38us | 429.00us | 25.95M chunks/s | 1.15x |
| 50000 | 789.17us | 1.12ms | 63.36M chunks/s | 1.93ms | 1.96ms | 25.91M chunks/s | 2.45x |

### End-to-end

| n | GPU p50 | GPU p99 | GPU throughput | CPU p50 | CPU p99 | CPU throughput | Speedup |
|---|---|---|---|---|---|---|---|
| 100 | 366.67us | 817.33us | 272.73K chunks/s | 174.96us | 187.29us | 571.57K chunks/s | 0.48x |
| 500 | 1.21ms | 2.06ms | 413.72K chunks/s | 924.83us | 1.01ms | 540.64K chunks/s | 0.77x |
| 2000 | 4.26ms | 4.53ms | 469.57K chunks/s | 3.74ms | 4.05ms | 534.52K chunks/s | 0.88x |
| 10000 | 16.73ms | 17.27ms | 597.71K chunks/s | 16.19ms | 16.78ms | 617.76K chunks/s | 0.97x |
| 50000 | 115.59ms | 117.28ms | 432.58K chunks/s | 117.35ms | 118.81ms | 426.09K chunks/s | 1.02x |

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