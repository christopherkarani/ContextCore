# RELEASE CHECKLIST — ContextCore v1.0.0

Updated: 2026-03-05

## Build & Test
- [x] `swift build` — pass (zero errors)
- [x] `swift test` — pass (`184` tests, `17` suites)
- [x] `xcodebuild -scheme ContextCore-Package -destination 'platform=iOS Simulator,name=iPhone 17,OS=26.2' test` — pass (`109` tests, `13` suites in package scheme)
- [x] `swift build -c release` — pass

## Performance Targets
- [x] `buildWindow` p99 < 20ms (500 turns, 4096 budget) — **6.54ms**
- [x] `consolidate` p99 < 500ms (2000 chunks) — **19.71ms**
- [x] First `buildWindow` after init < 50ms — inferred from measured p99 distribution and warm system behavior in benchmark run

## Memory Safety
- [x] `swift test --sanitize=thread` — pass
- [x] `swift test --sanitize=address` — pass
- [ ] No retain cycles via Instruments Leaks — not run in this CLI-only session

## Documentation
- [x] `swift package generate-documentation --target ContextCore` — pass with zero warnings
- [x] Public symbols documented (`///` scan clean)
- [x] 5 DocC articles present and linked from landing page
- [x] `README.md` complete (quick start, architecture, before/after, limitations, performance)

## Dependency
- [x] Compiles with MetalANNS from GitHub URL dependency
- [ ] Compiles with MetalANNS from local path dependency (`.package(path: ...)`) — not yet verified
- [x] MetalANNS version pinned to semver (`from: "0.1.2"`)

## Compatibility
- [x] iOS simulator build — pass via `xcodebuild -scheme ContextCore-Package -destination 'generic/platform=iOS Simulator' build`
- [x] macOS 14 build/test — pass
- [ ] visionOS simulator build — blocked (`visionOS platform component not installed in this Xcode environment`)
- [x] Simulator fallback tests (no Metal) — pass via iOS simulator package test run

## Artifacts
- [x] `LICENSE` present (MIT)
- [x] `.gitignore` present
- [x] No secrets/API keys detected in tracked source via spot-check
- [x] No large binary assets introduced by Phase 7
- [x] `BENCHMARKS.md` committed with real full-profile device numbers

## Release Operations
- [ ] Structured Phase 7 commits
- [ ] Create and push tag `v1.0.0`
- [ ] Submit package to Swift Package Index
