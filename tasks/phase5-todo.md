# Phase 5: Compression Engine — Task List

## 5.1 Compression Delegate Protocol & Extractive Fallback
- [ ] **TEST FIRST**: Write `CompressionEngineTests.swift`
  - [ ] Test: ExtractiveFallbackDelegate — 500-token paragraph to 150 tokens, output <= 150 tokens
  - [ ] Test: ExtractiveFallbackDelegate — output contains the most important sentence (highest similarity to chunk embedding)
  - [ ] Test: ExtractiveFallbackDelegate — sentence order preserved in output (extractive, not reordered)
  - [ ] Test: ExtractiveFallbackDelegate — target >= input tokens, returns original text unchanged
  - [ ] Test: ExtractiveFallbackDelegate — single sentence chunk, target < sentence tokens, returns that sentence (minimum 1 sentence)
  - [ ] Test: ExtractiveFallbackDelegate.extractFacts — 3-sentence paragraph, returns each sentence as a fact
  - [ ] Test: MockLLMDelegate — compress called with correct text and targetTokens
- [ ] Flesh out `CompressionDelegate` protocol (forward-declared in Phase 1)
  - [ ] `func compress(_ text: String, targetTokens: Int) async throws -> String`
  - [ ] `func extractFacts(from text: String) async throws -> [String]`
- [ ] Implement `ExtractiveFallbackDelegate`
  - [ ] Uses `CompressionEngine.rankSentences` (Phase 2.4) for importance scores
  - [ ] Greedy top-k sentence selection until token budget met
  - [ ] Preserves original sentence order in output
- [ ] Update `ContextConfiguration` — set `ExtractiveFallbackDelegate` as default compressionDelegate
- [ ] Run tests — all green
- [ ] Commit: `feat(phase5): 5.1 — CompressionDelegate protocol with extractive fallback`

## 5.2 Full Compression Engine
- [ ] **TEST FIRST**: Add to `CompressionEngineTests.swift`
  - [ ] Test: compress(chunk:targetTokens:) — 400-token chunk, target 100, extractive fallback — output <= 100 tokens
  - [ ] Test: compress(chunk:targetTokens:) — most important sentence present in output
  - [ ] Test: compress(chunk:targetTokens:) — chunk already under target — returned as-is, no Metal dispatch
  - [ ] Test: compress(chunk:targetTokens:) — metadata["compressionRatio"] set correctly
  - [ ] Test: compress(chunk:targetTokens:) — with MockLLMDelegate, verify delegate.compress called instead of extractive
  - [ ] Test: compressTurn(turn:targetTokens:) — 400-token turn compressed to 100 tokens
  - [ ] Test: compressTurn(turn:targetTokens:) — turn ID and role preserved, content replaced
  - [ ] Test: compressTurn(turn:targetTokens:) — tokenCount updated to match new content
- [ ] Extend `CompressionEngine` actor (partial from Phase 2.4)
  - [ ] Add `compressionDelegate` property (injected or default ExtractiveFallbackDelegate)
  - [ ] `func compress(chunk: MemoryChunk, targetTokens: Int) async throws -> MemoryChunk`
  - [ ] `func compressTurn(turn: Turn, targetTokens: Int) async throws -> Turn`
  - [ ] Set metadata["compressionRatio"] = "\(originalTokens / newTokens)"
- [ ] Run tests — all green
- [ ] Commit: `feat(phase5): 5.2 — Full compression engine with delegate routing`

## 5.3 Progressive Compression
- [ ] **TEST FIRST**: Write `ProgressiveCompressionTests.swift`
  - [ ] Test: 10 chunks (2000 tokens total, budget deficit 600) — deficit covered, lowest-scored compressed first
  - [ ] Test: Level 1 (50% reduction) tried before Level 2 on same chunk
  - [ ] Test: Level 2 (75% reduction) tried before Level 3 (drop) on same chunk
  - [ ] Test: highest-scored chunks remain untouched when deficit covered by compressing low-scored ones
  - [ ] Test: all chunks need Level 1 to cover deficit — all get .light compression
  - [ ] Test: deficit larger than total recoverable — some chunks dropped (Level 3)
  - [ ] Test: zero deficit — no compression applied, all chunks returned unchanged
  - [ ] Test: compressionLevel set correctly on each output ContextChunk
  - [ ] Test: token savings accounting — sum of saved tokens >= deficit
- [ ] Implement `ProgressiveCompressor`
  - [ ] Input: `[(chunk: MemoryChunk, evictionScore: Float)]` sorted ascending by score, `tokenDeficit: Int`
  - [ ] Output: `[(chunk: ContextChunk, tokensSaved: Int)]`
  - [ ] Level escalation per chunk: .light → .heavy → .dropped
  - [ ] Stop as soon as cumulative savings >= deficit
- [ ] Run tests — all green
- [ ] Commit: `feat(phase5): 5.3 — Progressive compression with level escalation`

## Final Verification
- [ ] Run full `swift build` — zero errors, zero warnings
- [ ] Run full `swift test` — all Phase 1–5 tests green
- [ ] Verify WindowPacker (Phase 3) can use CompressionEngine for oversized chunks
- [ ] Verify ExtractiveFallbackDelegate is wired as default in ContextConfiguration
- [ ] 3 clean commits in git log for Phase 5
