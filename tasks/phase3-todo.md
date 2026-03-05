# Phase 3: Window Packer — Task List

## 3.1 Context Window Type
- [ ] **TEST FIRST**: Write `WindowPackerTests.swift`
  - [ ] Test: ContextChunk init — all fields set correctly
  - [ ] Test: ContextWindow.totalTokens matches sum of chunk tokenCounts
  - [ ] Test: ContextWindow.budgetUsed = totalTokens / maxBudget
  - [ ] Test: ContextWindow.formatted(.raw) — plain concatenation of chunk content, newline-separated
  - [ ] Test: ContextWindow.formatted(.chatML) — each chunk wrapped in `<|im_start|>role\ncontent<|im_end|>`
  - [ ] Test: ContextWindow.formatted(.alpaca) — `### Instruction:` / `### Response:` markers
  - [ ] Test: ContextWindow.formatted(.custom("<<{role}>>")) — custom template with `{role}` and `{content}` placeholders
  - [ ] Test: empty window — totalTokens = 0, formatted returns empty string
  - [ ] Test: ContextChunk Codable roundtrip
- [ ] Implement `ContextChunk` struct (Identifiable, Codable, Sendable, Hashable)
- [ ] Implement `FormatStyle` enum (raw, chatML, alpaca, custom(String))
- [ ] Implement `ContextWindow` struct with `formatted(style:)` method
- [ ] Run tests — all green
- [ ] Commit: `feat(phase3): 3.1 — ContextWindow and ContextChunk types with formatting`

## 3.2 Window Packer
- [ ] **TEST FIRST**: Add to `WindowPackerTests.swift`
  - [ ] Test: 20 scored chunks (8000 tokens total), budget=4096 — output fits budget
  - [ ] Test: highest-scored chunks are included before lower-scored ones
  - [ ] Test: last 3 turns always present regardless of score
  - [ ] Test: system prompt always present and consumes reserved budget
  - [ ] Test: system prompt + 3 recent turns exceed budget — still all included, no memory chunks
  - [ ] Test: chunk too large to fit → attempt compression → if still too large → dropped
  - [ ] Test: remainingTokens < minimumChunkSize (50) → packing stops early
  - [ ] Test: empty inputs — no system prompt, no recent turns, no memory → empty window
  - [ ] Test: budget = 0 → returns empty window
  - [ ] Test: single chunk fits exactly at budget → included, budgetUsed ≈ 1.0
  - [ ] Test: retrievedFromMemory count matches chunks sourced from memory stores
  - [ ] Test: compressedChunks count matches chunks that went through compression
- [ ] Implement `WindowPacker` actor
  - [ ] `func pack(systemPrompt:recentTurns:scoredMemory:budget:) async -> ContextWindow`
  - [ ] Budget accounting with `remainingTokens` running counter
  - [ ] Packing order: system prompt → recent turns → scored memory (descending score)
  - [ ] Compression fallback via `CompressionEngine.rankSentences` for oversized chunks
  - [ ] Short-circuit when `remainingTokens < minimumChunkSize`
- [ ] Run tests — all green
- [ ] Commit: `feat(phase3): 3.2 — WindowPacker with budget accounting and compression fallback`

## 3.3 Chunk Orderer
- [ ] **TEST FIRST**: Add `ChunkOrdererTests.swift`
  - [ ] Test: `.typeGrouped` — system first, semantic second, episodic (chronological) third, recent turns fourth, current user message last
  - [ ] Test: `.relevanceAscending` — lowest score first, highest score last
  - [ ] Test: `.chronological` — ordered by timestamp
  - [ ] Test: mixed types — verify system prompt never moves from position 0
  - [ ] Test: single chunk — returns unchanged
  - [ ] Test: all same type — falls back to secondary sort (chronological within type)
- [ ] Implement `OrderingStrategy` enum (relevanceAscending, chronological, typeGrouped)
- [ ] Implement `ChunkOrderer` struct (stateless, no actor needed)
  - [ ] `func order(_ chunks: [ContextChunk], strategy: OrderingStrategy) -> [ContextChunk]`
- [ ] Run tests — all green
- [ ] Commit: `feat(phase3): 3.3 — ChunkOrderer with three ordering strategies`

## Final Verification
- [ ] Run full `swift build` — zero errors, zero warnings
- [ ] Run full `swift test` — all Phase 1 + Phase 2 + Phase 3 tests green
- [ ] Verify WindowPacker integrates with Phase 2 CompressionEngine for oversized chunks
- [ ] 3 clean commits in git log for Phase 3
