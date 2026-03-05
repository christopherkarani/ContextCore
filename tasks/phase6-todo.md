# Phase 6: Public API & AgentContext — Task List

## 6.1 AgentContext Actor
- [x] **TEST FIRST**: Write `AgentContextTests.swift`
  - [x] Test: init — creates all internal engines and stores without crash
  - [x] Test: beginSession — sets session ID, stores system prompt
  - [x] Test: endSession — triggers consolidation
  - [x] Test: append — embeds turn, counts tokens, inserts into EpisodicStore
  - [x] Test: append — auto-consolidation triggers at threshold (50 insertions)
  - [x] Test: append without active session — throws sessionNotStarted
  - [x] Test: buildWindow — 20 turns across 3 sessions, output fits token budget
  - [x] Test: buildWindow — most recent turns always included
  - [x] Test: buildWindow — semantic memory chunks present in output
  - [x] Test: buildWindow — procedural memory retrieved by task type match
  - [x] Test: buildWindow — respects maxTokens override
  - [x] Test: buildWindow — applies safety margin (effective budget = maxTokens * (1 - margin))
  - [x] Test: buildWindow without active session — throws sessionNotStarted
  - [x] Test: remember — inserts fact into SemanticStore
  - [x] Test: forget — soft-deletes chunk by ID
  - [x] Test: recall — returns k nearest chunks by query
  - [x] Test: consolidate — manual trigger works
- [x] Implement `AgentContext` actor
  - [x] Internal state: EpisodicStore, SemanticStore, ProceduralStore, ScoringEngine, AttentionEngine, CompressionEngine, ConsolidationEngine, ConsolidationScheduler, WindowPacker, ChunkOrderer, EmbeddingProvider, TokenCounter
  - [x] `init(configuration:)` — create all engines, warm up Metal pipelines on background task
  - [x] `beginSession(id:systemPrompt:)` — store session ID, system prompt
  - [x] `endSession()` — trigger consolidation, clear session state
  - [x] `append(turn:)` — embed → count → insert → notify scheduler
  - [x] `buildWindow(currentTask:maxTokens:)` — embed → recency → score → pack → order → return
  - [x] `remember(_:)`, `forget(id:)`, `recall(query:k:)`
  - [x] `consolidate()` — manual trigger
- [x] Run tests — all green
- [x] Commit: `feat(phase6): 6.1 — AgentContext actor with full pipeline`

## 6.2 ContextStats (Observability)
- [x] **TEST FIRST**: Add `ContextStatsTests.swift`
  - [x] Test: stats.episodicCount matches EpisodicStore.count after appends
  - [x] Test: stats.semanticCount matches SemanticStore.count after remember()
  - [x] Test: stats.totalSessions increments after beginSession/endSession cycle
  - [x] Test: stats.lastBuildWindowLatencyMs > 0 after buildWindow call
  - [x] Test: stats.averageRelevanceScore in [0, 1] after buildWindow
  - [x] Test: stats accessible without awaiting actor (nonisolated)
- [x] Implement `ContextStats` struct
- [x] Wire stats updates into AgentContext (append, buildWindow, consolidate)
- [x] Expose as `nonisolated` property with actor-isolated setter
- [x] Run tests — all green
- [x] Commit: `feat(phase6): 6.2 — ContextStats observability`

## 6.3 Session Persistence
- [x] **TEST FIRST**: Add `PersistenceTests.swift`
  - [x] Test: checkpoint — writes file to disk at specified URL
  - [x] Test: checkpoint — atomic write (write-to-temp + rename)
  - [x] Test: load — restores AgentContext from checkpoint
  - [x] Test: roundtrip — 50 turns, checkpoint, load, buildWindow, verify results equivalent
  - [x] Test: roundtrip — SemanticStore facts survive checkpoint/load
  - [x] Test: roundtrip — ProceduralStore patterns survive checkpoint/load
  - [x] Test: load corrupt file — throws checkpointCorrupt
  - [x] Test: load nonexistent file — throws appropriate error
  - [x] Test: checkpoint overwrites existing file cleanly
- [x] Implement `ContextCheckpoint` (Codable)
  - [x] Serialize EpisodicStore (ANNSIndex.save + chunk metadata)
  - [x] Serialize SemanticStore (ANNSIndex.save + chunk metadata)
  - [x] Serialize ProceduralStore (JSON)
  - [x] Serialize session metadata and configuration
- [x] Implement `checkpoint(to:)` with atomic write pattern
- [x] Implement `AgentContext.load(from:)` static factory
- [x] Run tests — all green
- [x] Commit: `feat(phase6): 6.3 — Session persistence with atomic checkpoints`

## 6.4 Error Handling
- [x] **TEST FIRST**: Add error path tests to `AgentContextTests.swift`
  - [x] Test: embeddingFailed — embedding provider throws → wrapped as ContextCoreError
  - [x] Test: tokenBudgetTooSmall — maxTokens = 10, system prompt > 10 tokens → error
  - [x] Test: sessionNotStarted — call append/buildWindow before beginSession
  - [x] Test: dimensionMismatch — inject provider with wrong dimension
  - [x] Test: metalDeviceUnavailable — simulator without Metal → graceful error
- [x] Verify all public throwing functions throw `ContextCoreError`
- [x] Add OSLog logging for key events (session start/end, consolidation, errors)
- [x] Run tests — all green
- [x] Commit: `feat(phase6): 6.4 — Error handling and OSLog diagnostics`

## Final Verification
- [x] Run full `swift build` — zero errors, zero warnings
- [x] Run full `swift test` — all Phase 1–6 tests green
- [x] Verify full pipeline: append → buildWindow → consolidate → checkpoint → load → buildWindow
- [x] Verify no regressions across all prior phases
- [x] 4 clean commits in git log for Phase 6

## Review
- [x] Implemented `AgentContext` public actor with append/buildWindow/checkpoint orchestration over all Phase 1–5 subsystems.
- [x] Added nonisolated `ContextStats` with lock-backed reads and actor-safe updates.
- [x] Added checkpoint persistence model and restore path with atomic file writes.
- [x] Added integration suites: `AgentContextTests`, `ContextStatsTests`, `PersistenceTests`.
- [x] Verified full package build and test regression status.
- Build result: `swift build` passed.
- Test result: `swift test` passed (`184` tests in `17` suites).
- Notable implementation decision: `checkpoint(to:)` is `async throws` to preserve actor correctness when reading async stores.
