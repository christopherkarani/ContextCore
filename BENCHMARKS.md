# ContextCore Benchmarks

> Profile: `full`.

Measured on chriskarani.local, Version 26.0 (Build 25A354), 2026-03-05T22:38:29Z.

## buildWindow Latency

| Turns | Budget | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|---|
| 10 | 2048 | 1.33 | 2.19 | 2.55 |
| 10 | 4096 | 1.31 | 1.74 | 1.90 |
| 10 | 8192 | 1.25 | 1.89 | 2.27 |
| 50 | 2048 | 1.52 | 2.56 | 2.69 |
| 50 | 4096 | 1.46 | 2.28 | 2.57 |
| 50 | 8192 | 1.46 | 2.16 | 2.32 |
| 200 | 2048 | 1.72 | 2.46 | 2.60 |
| 200 | 4096 | 1.67 | 2.74 | 2.82 |
| 200 | 8192 | 1.78 | 2.65 | 3.13 |
| 500 | 2048 | 2.41 | 3.45 | 3.54 |
| 500 | 4096 | 2.43 | 3.80 | 4.08 |
| 500 | 8192 | 2.22 | 2.86 | 3.53 |

**Target**: p99 < 20ms for 500 turns on M2.

## Consolidation Latency

| Chunks | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|
| 100 | 0.89 | 2.17 | 2.17 |
| 500 | 3.82 | 4.96 | 4.96 |
| 2000 | 18.76 | 22.43 | 22.43 |

**Target**: p99 < 500ms for 2000 chunks on M2.

## Metal vs CPU Scoring

Math-only isolates score computation over pre-flattened inputs with resident GPU buffers. End-to-end includes public API validation, flattening, zip, and final sort.

### Math-only

| n | GPU p50 | GPU p99 | GPU throughput | CPU p50 | CPU p99 | CPU throughput | Speedup |
|---|---|---|---|---|---|---|---|
| 100 | 221.88us | 912.83us | 450.70K chunks/s | 3.92us | 4.00us | 25.53M chunks/s | 0.02x |
| 500 | 176.92us | 783.88us | 2.83M chunks/s | 18.54us | 20.88us | 26.97M chunks/s | 0.10x |
| 2000 | 210.33us | 381.88us | 9.51M chunks/s | 72.46us | 73.67us | 27.60M chunks/s | 0.34x |
| 10000 | 314.54us | 713.33us | 31.79M chunks/s | 370.17us | 404.96us | 27.01M chunks/s | 1.18x |
| 50000 | 832.58us | 1.19ms | 60.05M chunks/s | 1.84ms | 2.10ms | 27.12M chunks/s | 2.21x |

### End-to-end

| n | GPU p50 | GPU p99 | GPU throughput | CPU p50 | CPU p99 | CPU throughput | Speedup |
|---|---|---|---|---|---|---|---|
| 100 | 368.54us | 969.17us | 271.34K chunks/s | 177.29us | 195.83us | 564.04K chunks/s | 0.48x |
| 500 | 1.14ms | 1.52ms | 438.84K chunks/s | 900.42us | 988.88us | 555.30K chunks/s | 0.79x |
| 2000 | 4.11ms | 4.82ms | 486.39K chunks/s | 3.47ms | 3.66ms | 576.20K chunks/s | 0.84x |
| 10000 | 16.15ms | 16.67ms | 619.03K chunks/s | 15.62ms | 17.00ms | 640.27K chunks/s | 0.97x |
| 50000 | 111.25ms | 113.32ms | 449.43K chunks/s | 110.47ms | 114.88ms | 452.60K chunks/s | 0.99x |

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
