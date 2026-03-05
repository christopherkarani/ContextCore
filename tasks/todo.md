# Phase 1: Foundation & Data Model — Task List

## 1.1 Package Scaffold
- [ ] Create directory structure: `Sources/ContextCoreShaders/`, `Sources/ContextCoreEngine/`, `Sources/ContextCore/`, `Tests/ContextCoreTests/`
- [ ] Create placeholder files in empty targets so they compile (e.g. empty `.swift` exports)
- [ ] Write `Package.swift` with 4 targets and correct dependency graph
- [ ] Add MetalANNS dependency (`.package(url: "https://github.com/chriskarani/MetalANNS.git", branch: "main")`)
- [ ] Set platforms: iOS 17, macOS 14, visionOS 1
- [ ] Configure resource processing rules (Shaders, CoreML model)
- [ ] Link system frameworks: Metal, CoreML, Accelerate
- [ ] Run `swift build` — zero errors, zero warnings
- [ ] Init git repo, create `.gitignore`, commit

## 1.2 Turn Data Model
- [ ] **TEST FIRST**: Write `TurnTests.swift`
  - [ ] Test: Turn JSON encode/decode roundtrip (all fields survive)
  - [ ] Test: ToolCall JSON encode/decode roundtrip
  - [ ] Test: Turn with ToolCall serialized in metadata roundtrips
  - [ ] Test: Two Turns with different UUIDs are not equal
  - [ ] Test: Two Turns with same UUID are equal (Hashable contract)
- [ ] Implement `TurnRole` enum (user, assistant, tool, system) — `Codable`, `Sendable`, `Hashable`
- [ ] Implement `Turn` struct with all fields — `Identifiable`, `Codable`, `Sendable`, `Hashable`
- [ ] Implement `ToolCall` struct — `Codable`, `Sendable`, `Hashable`
- [ ] Add public initializers with sensible defaults (e.g. `timestamp: Date = .now`, `metadata: [:]`)
- [ ] Run tests — all green
- [ ] Commit: `feat(phase1): 1.2 — Turn data model`

## 1.3 Memory Types
- [ ] **TEST FIRST**: Write `MemoryStoreTests.swift`
  - [ ] Test: MemoryChunk JSON roundtrip
  - [ ] Test: EpisodicStore — insert 10 chunks, retrieve by query vector, results non-empty
  - [ ] Test: EpisodicStore — count increments after each insert
  - [ ] Test: SemanticStore — insert 10 chunks, retrieve by query, results non-empty
  - [ ] Test: SemanticStore — upsert near-duplicate increments accessCount instead of creating new entry
  - [ ] Test: ProceduralStore — record 5 task types, retrieve by exact key
  - [ ] Test: ProceduralStore — retrieve by prefix match returns correct results
  - [ ] Test: ProceduralStore — insert 1001 entries, verify count capped at 1000 (LRU eviction)
- [ ] Implement `MemoryType` enum — `Codable`, `Sendable`, `Hashable`
- [ ] Implement `MemoryChunk` struct with all fields — `Identifiable`, `Codable`, `Sendable`, `Hashable`
- [ ] Implement `EpisodicStore` actor wrapping MetalANNS index
  - [ ] `insert(turn:)` — creates MemoryChunk, inserts into ANNS
  - [ ] `retrieve(query:k:)` — kNN search, returns [MemoryChunk]
  - [ ] `count` property
- [ ] Implement `SemanticStore` actor
  - [ ] Same as EpisodicStore + higher default retention (1.0)
  - [ ] `upsert(fact:embedding:)` — dedup at similarity > 0.9
- [ ] Implement `ProceduralStore` actor
  - [ ] `[String: [ToolCall]]` backing store
  - [ ] `record(taskType:tools:)` and `retrieve(taskType:)` with prefix match
  - [ ] LRU eviction at 1000 entries
- [ ] Run tests — all green
- [ ] Commit: `feat(phase1): 1.3 — Memory types and stores`

## 1.4 Embedding Provider
- [ ] **TEST FIRST**: Write `EmbeddingTests.swift`
  - [ ] Test: embed same string twice via CachingEmbeddingProvider — vectors identical, second call < 1ms
  - [ ] Test: embedBatch 10 strings — all dim=384, no two identical
  - [ ] Test: cache eviction — fill to capacity (512), insert one more, verify oldest evicted
  - [ ] Test: EmbeddingProvider.dimension returns 384
- [ ] Define `EmbeddingProvider` protocol (`embed`, `embedBatch`, `dimension`)
- [ ] Implement `CoreMLEmbeddingProvider`
  - [ ] Load mlpackage from bundle
  - [ ] `embed(_:)` via CoreML async prediction
  - [ ] `embedBatch(_:)` via MLBatchProvider
  - [ ] Simulator fallback: deterministic pseudo-random vector seeded by string hash
- [ ] Implement `EmbeddingCache` actor in ContextCoreEngine
  - [ ] LRU eviction policy
  - [ ] Capacity 512 (configurable)
  - [ ] Key by SHA256 of input string
- [ ] Implement `CachingEmbeddingProvider` wrapper
- [ ] Run tests — all green
- [ ] Commit: `feat(phase1): 1.4 — Embedding provider with LRU cache`

## 1.5 Token Counter & Configuration
- [ ] **TEST FIRST**: Write `TokenCounterTests.swift`
  - [ ] Test: 5 known strings with hardcoded GPT-2 reference counts, verify within 20%
  - [ ] Test: empty string returns 0
  - [ ] Test: ContextConfiguration.default has correct values for all 14 parameters
  - [ ] Test: ContextConfiguration is Sendable (compile-time — test instantiation across isolation boundaries)
- [ ] Define `TokenCounter` protocol
- [ ] Implement `ApproximateTokenCounter` — whitespace+punctuation split, 1.3x multiplier
- [ ] Implement `ContextConfiguration` struct with all parameters and `.default` static property
- [ ] Forward-declare `CompressionDelegate` protocol (empty, implemented in Phase 5)
- [ ] Run tests — all green
- [ ] Commit: `feat(phase1): 1.5 — Token counter and configuration`

## 1.6 Error Types
- [ ] Implement `ContextCoreError` enum with all 8 cases
- [ ] Verify it conforms to `Error` and `Sendable`
- [ ] Run `swift build` — compiles clean
- [ ] Commit: `feat(phase1): 1.6 — Error types`

## Final Verification
- [ ] Run full `swift test` — all tests green
- [ ] Run `swift build` — zero warnings
- [ ] Verify all public types conform to `Codable`, `Sendable`, `Hashable`
- [ ] Verify 6 clean commits in git log
- [ ] Verify MetalANNS dependency resolves correctly

# Phase 3: Window Packer — Task List

## 3.1 ContextWindow Type
- [x] **TEST FIRST**: Add ContextWindow/ContextChunk tests in `WindowPackerTests.swift`
- [x] Implement `ContextWindow.swift` (`ContextChunk`, `CompressionLevel`, `FormatStyle`, `ContextWindow`)
- [x] Verify formatting rules for `.raw`, `.chatML`, `.alpaca`, `.custom`
- [x] Run `swift test` and confirm 3.1 tests are green
- [ ] Commit: `feat(phase3): 3.1 — ContextWindow and ContextChunk types with formatting`

## 3.2 WindowPacker
- [x] **TEST FIRST**: Add budget/compression/ordering tests in `WindowPackerTests.swift`
- [x] Implement `WindowPacker.swift` actor with guaranteed system/recent inclusion
- [x] Implement compression fallback via `CompressionEngine.rankSentences`
- [x] Run `swift test` and confirm 3.2 tests are green
- [ ] Commit: `feat(phase3): 3.2 — WindowPacker with budget accounting and compression fallback`

## 3.3 ChunkOrderer
- [x] **TEST FIRST**: Add ordering strategy tests in `ChunkOrdererTests.swift`
- [x] Implement `ChunkOrderer.swift` with `.typeGrouped`, `.relevanceAscending`, `.chronological`
- [x] Verify system prompt pinning and guaranteed recent separation
- [x] Run `swift test` and confirm 3.3 tests are green
- [ ] Commit: `feat(phase3): 3.3 — ChunkOrderer with three ordering strategies`

## Final Verification
- [ ] `swift build` passes with zero errors/warnings
- [ ] `swift test` passes for all suites (Phase 1 + 2 + 3)
- [ ] Verify 3 clean Phase 3 commits in git log

## Review
- [ ] Summary of implementation results
- [ ] Total test count and pass/fail
- [ ] Residual risks / follow-ups
