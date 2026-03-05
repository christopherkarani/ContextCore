# Phase 4: Consolidation Engine — Task List

## 4.1 Pairwise Similarity Kernel
- [ ] **TEST FIRST**: Write `ConsolidationTests.swift`
  - [ ] Test: CPU reference — 20 chunks dim=384, compute all-pairs cosine similarity, verify symmetric and diagonal = 1.0
  - [ ] Test: GPU `pairwise_similarity` — 100 chunks, compare upper triangle vs CPU reference, max error < 1e-4
  - [ ] Test: GPU `find_merge_candidates` — 100 chunks (10 near-duplicate pairs), threshold=0.92, all 10 pairs detected
  - [ ] Test: no false positives — 100 unique chunks (no duplicates), threshold=0.92, zero pairs returned
  - [ ] Test: threshold sensitivity — same data at threshold=0.80 returns more pairs than 0.92
  - [ ] Test: n=1 edge case — single chunk, no pairs returned
  - [ ] Test: n=2 edge case — two identical chunks, one pair returned
  - [ ] Test: tiled fallback — 2100 chunks (triggers tiled path n > 2048), verify correct results
- [ ] Write `Consolidation.metal`
  - [ ] `pairwise_similarity` kernel — upper triangle, triangular index mapping
  - [ ] `find_merge_candidates` kernel — threshold scan with atomic counter
- [ ] Write CPU reference in `CPUReference.swift`
  - [ ] `pairwiseSimilarity(embeddings:) -> [[Float]]` — full n×n matrix
  - [ ] `findMergeCandidates(similarities:threshold:) -> [(Int, Int)]`
- [ ] Write `ConsolidationEngine` actor in `ContextCoreEngine/ConsolidationEngine.swift`
  - [ ] Metal pipeline state setup
  - [ ] Buffer management for n×n similarity matrix (upper triangle storage)
  - [ ] Tiled dispatch for n > 2048 (512×512 tiles)
  - [ ] `func findDuplicates(in store: EpisodicStore, threshold: Float) async throws -> [(UUID, UUID)]`
- [ ] Run tests — all green
- [ ] Commit: `feat(phase4): 4.1 — Pairwise similarity kernel with tiled dispatch`

## 4.2 Semantic Extraction & Scheduling
- [ ] **TEST FIRST**: Add to `ConsolidationTests.swift`
  - [ ] Test: consolidate — 60 chunks (50 unique + 10 near-duplicate pairs), SemanticStore gains >= 8 facts
  - [ ] Test: consolidate — EpisodicStore count decreases after consolidation
  - [ ] Test: consolidate — promoted fact is the shorter of each duplicate pair
  - [ ] Test: consolidate — original episodic chunks' retentionScore decremented by 0.2
  - [ ] Test: consolidate — chunks with retentionScore < 0.1 are evicted from EpisodicStore
  - [ ] Test: consolidate — SemanticStore.upsert deduplicates if same fact promoted twice
  - [ ] Test: ConsolidationScheduler — auto-triggers when episodicStore.count > 200
  - [ ] Test: ConsolidationScheduler — auto-triggers when insertionsSinceLastConsolidation > 50
  - [ ] Test: ConsolidationScheduler — does not block the calling task (runs at .background priority)
  - [ ] Test: consolidate on empty store — no crash, no changes
- [ ] Implement `func consolidate(session: UUID) async throws` on ConsolidationEngine
  - [ ] findDuplicates pipeline
  - [ ] Shorter-chunk-as-fact selection
  - [ ] SemanticStore.upsert for promoted facts
  - [ ] retentionScore decrement on originals
  - [ ] Eviction of chunks with retentionScore < 0.1
- [ ] Implement `ConsolidationScheduler` (internal)
  - [ ] Track `insertionsSinceLastConsolidation` counter
  - [ ] Check thresholds: count > 200 OR insertions > 50
  - [ ] Dispatch consolidation on `Task.detached(priority: .background)`
  - [ ] Debounce: don't trigger if consolidation already in progress
- [ ] Run tests — all green
- [ ] Commit: `feat(phase4): 4.2 — Semantic extraction with auto-scheduling`

## 4.3 Contradiction Detection
- [ ] **TEST FIRST**: Write `ContradictionTests.swift`
  - [ ] Test: two embeddings that are exact negations — antipodal fraction > 0.5
  - [ ] Test: two nearly identical embeddings — antipodal fraction < 0.3
  - [ ] Test: GPU vs CPU parity on antipodal_test — max error < 1e-6 (integer comparison, should be exact)
  - [ ] Test: contradiction candidates — 2 contradictory facts among 20 — both returned
  - [ ] Test: no contradictions — 20 non-contradictory facts — empty result
  - [ ] Test: similarity filter — pair with similarity < 0.75 not considered even if highly antipodal
  - [ ] Test: n=0 semantic facts — returns empty, no crash
- [ ] Write `antipodal_test` kernel (add to `Consolidation.metal` or new file)
  - [ ] For each pair: count dimensions where sign differs, divide by dim
- [ ] Write CPU reference
  - [ ] `antipodalFraction(a:b:) -> Float`
- [ ] Extend `ConsolidationEngine`
  - [ ] `func contradictionCandidates() async throws -> [(MemoryChunk, MemoryChunk)]`
  - [ ] Filter: similarity > 0.75 AND antipodalFraction > 0.30
- [ ] Run tests — all green
- [ ] Commit: `feat(phase4): 4.3 — Contradiction detection with antipodal heuristic`

## Final Verification
- [ ] Run full `swift build` — zero errors, zero warnings
- [ ] Run full `swift test` — all Phase 1–4 tests green
- [ ] Verify consolidation runs without blocking main actor
- [ ] Verify tiled pairwise similarity works for n > 2048
- [ ] 3 clean commits in git log for Phase 4
