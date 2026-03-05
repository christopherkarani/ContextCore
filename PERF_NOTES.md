# ContextCore Performance Notes

## Bottlenecks Found

- `ScoringEngine.scoreChunks` flattened every chunk embedding with a fresh `flatMap`, then sorted results even when callers reweighted and resorted later.
- The scoring and recency paths allocated fresh `MTLBuffer` objects on every invocation, including tiny constant buffers.
- The scoring benchmark compared GPU end-to-end work against CPU math-only work, which overstated GPU overhead and hid where time was actually going.
- The Metal relevance kernel recomputed `queryNorm` inside every thread.
- `topKIndices` used a single-thread GPU kernel for effectively serial work.

## Changes Made

- Added package-scoped scoring hooks:
  - `scoreChunksUnsorted(...)`
  - `scoreFlattenedEmbeddings(...)`
- Switched `AgentContext` to the unsorted scoring path so retrieval scoring no longer pays a redundant full sort before retention weighting and final ranking.
- Replaced closure-heavy validation/flattening with a single-pass `reserveCapacity` + append path.
- Added reusable shared-memory `MTLBuffer` caches for scoring and recency inputs/outputs.
- Replaced per-call scalar buffers with `setBytes(...)` constants.
- Precomputed `queryNorm` on CPU once per scoring call and passed it into the Metal kernel.
- Replaced the serial GPU top-k implementation with a deterministic CPU sort path for current workload sizes.
- Split scoring benchmarks into:
  - `math-only`: pre-flattened scoring math only
  - `end-to-end`: public API validation + flatten + zip + sort
- Added throughput reporting (`chunks/s`), larger scoring workloads, alternating CPU/GPU measurement order, and adaptive `us`/`ms` formatting.
- Added scoring tests for the new package-scoped raw/unsorted paths.

## Before / After Metrics

- Historical baseline from `BENCHMARKS.md` before this pass:
  - `buildWindow` p99 at `(500 turns, 4096 tokens)`: `6.54ms`
  - `consolidate` p99 at `(2000 chunks)`: `19.71ms`
  - scoring at `n=2000`: `0.02x` GPU vs CPU
- Latest full release benchmark after this pass:
  - `buildWindow` p99 at `(500 turns, 4096 tokens)`: `7.38ms`
  - `consolidate` p99 at `(2000 chunks)`: `39.80ms`
  - scoring `math-only`:
    - `n=2000`: `0.09x`
    - `n=10000`: `0.29x`
    - `n=50000`: `0.82x`
  - scoring `end-to-end`:
    - `n=2000`: `0.55x`
    - `n=10000`: `0.88x`
    - `n=50000`: `1.02x`

## Remaining Limits

- The relevance kernel is still launch-overhead bound for small `n` at `dim=384`; CPU remains faster for the low-thousands regime.
- Public `scoreChunks(...)` still must flatten and sort to preserve API behavior. Internal callers now have a lower-overhead path, but the public API cannot skip that work.
- The full benchmark reruns did not reproduce the historical `buildWindow` and `consolidate` baseline on this machine. Since those code paths were not materially changed here, that needs a separate controlled benchmark investigation before claiming regression-free non-scoring latency.
