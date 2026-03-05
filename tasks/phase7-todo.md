# Phase 7: Documentation, Benchmarks & Release — Task List

## 7.1 DocC Documentation
- [x] Add doc comments to every `public` type and function across all targets
- [x] Create DocC catalog (`ContextCore.docc/`)
- [x] Article: **Getting Started**
- [x] Article: **Architecture Overview**
- [x] Article: **Tuning for Your Use Case**
- [x] Article: **Integrating with Apple Foundation Models**
- [x] Article: **Integrating with Wax**
- [x] Verify `swift package generate-documentation` builds without warnings
- [ ] Commit: `docs(phase7): 7.1 — DocC documentation for all public API`

## 7.2 README
- [x] Write problem statement
- [x] Add architecture diagram (ASCII)
- [x] Write quick start code
- [x] Add MetalANNS dependency section with link
- [x] Write parameter reference table
- [x] Write known limitations section
- [x] Write before/after example
- [x] Add badges
- [x] Add memory footprint section
- [ ] Commit: `docs(phase7): 7.2 — README with quick start, architecture, and parameter reference`

## 7.3 Benchmark Suite
- [x] Add `ContextCoreBenchmarks` executable target to Package.swift
- [x] BuildWindow latency matrix: 10/50/200/500 turns × 2048/4096/8192 budgets
- [x] Consolidation latency scaling: 100/500/2000 chunks
- [x] Metal vs CPU scoring benchmark: n=100/500/2000
- [x] Recall quality benchmark: precision@3/5/8
- [x] Benchmark harness: warmup + percentile reporting
- [x] Run on local Mac and commit results to `BENCHMARKS.md`
- [ ] Commit: `perf(phase7): 7.3 — Benchmark suite with latency and throughput measurements`

## 7.4 Release Checklist
- [x] `swift build` passes
- [x] `swift test` passes (184 tests)
- [x] `swift build -c release` passes
- [x] `swift test --sanitize=thread` passes
- [x] `swift test --sanitize=address` passes
- [x] `swift package generate-documentation --target ContextCore` passes with zero warnings
- [x] `buildWindow` p99 < 20ms (500 turns, 4096 budget): **6.54ms**
- [x] `consolidate` p99 < 500ms (2000 chunks): **19.71ms**
- [x] LICENSE file present (MIT)
- [x] Write GitHub release notes draft (`RELEASE_NOTES_v1.0.0.md`)
- [x] iOS simulator build/test verification via `xcodebuild` package scheme
- [ ] visionOS simulator build verification (toolchain target support issue in this environment)
- [ ] Compile verification with MetalANNS as `.package(path: ...)`
- [ ] Tag `v1.0.0`
- [ ] Submit package to Swift Package Index
- [ ] Commit: `chore(phase7): 7.4 — Release preparation`

## Final Verification
- [x] Full `swift build` — zero errors
- [x] Full `swift test` — all tests green
- [x] DocC documentation accessible and complete
- [x] BENCHMARKS.md updated with full-profile real device numbers
- [x] README.md complete and accurate
- [ ] 4 clean commits in git log for Phase 7

## Review
- Completed: documentation pass, README rewrite, benchmark executable + full benchmark run, release verification (build/test/release/doc/sanitizers), benchmark/result artifacts, release notes draft.
- Performance targets: met for `buildWindow` and `consolidate`; scoring benchmark currently shows GPU launch overhead dominating small batch sizes.
- Remaining release ops: tag creation/push, Swift Package Index submission, local-path dependency verification cleanup, and visionOS validation on an environment with the visionOS platform component installed.
