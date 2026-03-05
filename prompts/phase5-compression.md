# Phase 5: Compression Engine

You are implementing Phase 5 of ContextCore. Phases 1–4 are complete. You have data models, memory stores, Metal scoring engines (relevance, recency, attention, sentence importance), `WindowPacker`, `ChunkOrderer`, and the full `ConsolidationEngine` with contradiction detection. All compile and pass tests.

Phase 5 completes the compression subsystem. Phase 2.4 built the Metal kernel that ranks sentences by importance. Phase 3 built the `WindowPacker` that calls compression when a chunk doesn't fit. This phase wires them together through a delegate pattern: Metal ranks sentences, a delegate decides *how* to compress (extractive or LLM-based), and a progressive compressor decides *which* chunks to compress and *how aggressively*.

After this phase, the full pipeline from scoring → packing → compression → output is functional.

---

## What You Are Building

Three components that form a compression pipeline:

| Component | File | Responsibility |
|---|---|---|
| **CompressionDelegate** | `Protocols/CompressionDelegate.swift` | Defines how compression happens (extractive vs LLM) |
| **CompressionEngine** | `ContextCoreEngine/CompressionEngine.swift` | Orchestrates: Metal ranking → delegate → output |
| **ProgressiveCompressor** | `ContextCoreEngine/ProgressiveCompressor.swift` | Decides which chunks to compress and how aggressively |

```
WindowPacker detects: chunk too large for remaining budget
    │
    ▼
CompressionEngine.compress(chunk, targetTokens)
    │
    ├─ chunk.tokenCount <= targetTokens? → return as-is
    │
    ├─ rankSentences (Metal, Phase 2.4) → importance scores
    │
    └─ delegate.compress(text, targetTokens)
        │
        ├─ ExtractiveFallbackDelegate (default)
        │   └─ greedily select top sentences until budget met
        │
        └─ Consumer's LLM delegate (injected)
            └─ abstractive summarization
```

For budget deficits that span multiple chunks, the `ProgressiveCompressor` escalates:

```
Deficit = 600 tokens across 10 chunks

Chunk (lowest score) → Level 1 (50% reduction) → saved 100 tokens → deficit = 500
Chunk (next lowest)  → Level 1 (50% reduction) → saved 80 tokens  → deficit = 420
Chunk (next lowest)  → Level 1 (50% reduction) → saved 120 tokens → deficit = 300
...continue until deficit <= 0

If Level 1 not enough on a chunk → escalate to Level 2 (75% reduction)
If Level 2 not enough → Level 3 (drop entirely)
Never touch higher-scored chunks if lower-scored ones cover the deficit
```

---

## Phase 1–4 Types You Depend On

Already exist — import, do not redefine:

```swift
// Phase 1
public struct Turn { id, role, content, timestamp, tokenCount, embedding, metadata }
public struct MemoryChunk { id, content, embedding, type, createdAt, lastAccessedAt, accessCount, retentionScore, sourceSessionID, metadata }
public protocol TokenCounter { func count(_ text: String) -> Int }
public struct ContextConfiguration { /* ... compressionDelegate field exists but may be nil */ }

// Phase 2.4
public actor CompressionEngine {
    // Already has:
    func rankSentences(in chunk: String, chunkEmbedding: [Float]) async throws -> [(sentence: String, importance: Float)]
    // You will EXTEND this actor with compress() and compressTurn() methods
}

// Phase 3
public struct ContextChunk { id, content, role, tokenCount, score, source, compressionLevel, timestamp, isGuaranteedRecent }
public enum CompressionLevel: Int { case none = 0, light = 1, heavy = 2, dropped = 3 }
```

**Important**: `CompressionDelegate` was forward-declared in Phase 1 (empty protocol body). You are now fleshing it out with real methods. The `CompressionEngine` actor from Phase 2.4 already has `rankSentences`. You are extending it with `compress()` and `compressTurn()` — do not create a second actor.

---

## File Map (Phase 5 additions and modifications)

```
Sources/
  ContextCore/
    Protocols/
      CompressionDelegate.swift     # MODIFIED — flesh out from forward declaration
    Compression/
      ExtractiveFallbackDelegate.swift  # NEW
  ContextCoreEngine/
    CompressionEngine.swift         # MODIFIED — add compress(), compressTurn()
    ProgressiveCompressor.swift     # NEW
Tests/
  ContextCoreTests/
    CompressionEngineTests.swift    # NEW
    ProgressiveCompressionTests.swift  # NEW
```

---

## Execution Plan

Three sub-tasks. TDD for each.

---

### 5.1 — CompressionDelegate Protocol & Extractive Fallback

The delegate pattern that separates *how* to compress from *what* to compress. The default implementation uses no LLM — it extracts the most important sentences.

#### Protocol

Flesh out the forward-declared `CompressionDelegate` in `Protocols/CompressionDelegate.swift`:

```swift
public protocol CompressionDelegate: Sendable {
    /// Compress text to fit within targetTokens.
    /// The implementation decides how: extractive (sentence selection) or abstractive (LLM summarization).
    func compress(_ text: String, targetTokens: Int) async throws -> String

    /// Extract standalone facts from text.
    /// Each returned string should be a self-contained statement.
    func extractFacts(from text: String) async throws -> [String]
}
```

#### ExtractiveFallbackDelegate

A compression delegate that uses zero LLM calls. It relies on the sentence importance scores from Phase 2.4's Metal kernel.

```swift
public actor ExtractiveFallbackDelegate: CompressionDelegate {
    private let compressionEngine: CompressionEngine
    private let tokenCounter: any TokenCounter

    public init(
        compressionEngine: CompressionEngine,
        tokenCounter: any TokenCounter
    )

    public func compress(_ text: String, targetTokens: Int) async throws -> String

    public func extractFacts(from text: String) async throws -> [String]
}
```

**`compress` implementation**:

```
func compress(_ text: String, targetTokens: Int) async throws -> String:

    let currentTokens = tokenCounter.count(text)

    // Already fits — return unchanged
    if currentTokens <= targetTokens:
        return text

    // Embed the full text to use as the query for sentence ranking
    let textEmbedding = try await compressionEngine.embeddingProvider.embed(text)

    // Rank sentences by importance (Metal kernel)
    let ranked = try await compressionEngine.rankSentences(in: text, chunkEmbedding: textEmbedding)

    // Greedy selection: take most important sentences until budget full
    var selected: [(sentence: String, importance: Float, originalIndex: Int)] = []
    var tokensUsed = 0

    // Tag each sentence with its original position for order preservation
    let withIndices = ranked.enumerated().map { (offset, pair) in
        // ranked is sorted by importance descending — but we need original order
        // Find the original index of this sentence in the text
        (sentence: pair.sentence, importance: pair.importance, originalIndex: findOriginalIndex(pair.sentence, in: text))
    }

    // Sort by importance descending (already is), then select greedily
    for item in withIndices.sorted(by: { $0.importance > $1.importance }):
        let sentenceTokens = tokenCounter.count(item.sentence)
        if tokensUsed + sentenceTokens > targetTokens:
            continue  // skip this sentence, try smaller ones
        selected.append(item)
        tokensUsed += sentenceTokens

    // Reassemble in original order
    selected.sort(by: { $0.originalIndex < $1.originalIndex })
    return selected.map(\.sentence).joined(separator: " ")
```

Key behaviors:
- **Preserves sentence order** — sentences appear in their original sequence, not sorted by importance. This maintains readability.
- **Greedy with skip** — if a sentence doesn't fit, skip it and try the next (smaller) one. This maximizes budget utilization.
- **Minimum 1 sentence** — if even the most important sentence exceeds the target, return it anyway. Never return empty string.

**`extractFacts` implementation**:

Split text into sentences using `NLTokenizer(.sentence)`. Return each sentence as a standalone fact. This is the simplest possible extraction — no semantic analysis.

```swift
public func extractFacts(from text: String) async throws -> [String] {
    let tokenizer = NLTokenizer(unit: .sentence)
    tokenizer.string = text
    var facts: [String] = []
    tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
        let sentence = String(text[range]).trimmingCharacters(in: .whitespacesAndNewlines)
        if !sentence.isEmpty {
            facts.append(sentence)
        }
        return true
    }
    return facts
}
```

#### Update ContextConfiguration

In `ContextConfiguration`, the `compressionDelegate` field was declared as `(any CompressionDelegate)?`. Change the `.default` static property to instantiate an `ExtractiveFallbackDelegate`:

```swift
public static var `default`: ContextConfiguration {
    // Note: this requires a CompressionEngine, which requires a Metal device.
    // On simulator or when Metal is unavailable, compressionDelegate stays nil
    // and compression falls back to simple truncation.
    // The actual wiring happens in AgentContext.init (Phase 6).
    ...
}
```

**Practical consideration**: `ExtractiveFallbackDelegate` requires a `CompressionEngine` instance, which requires Metal. The `.default` configuration can't create these eagerly. Instead:
- Keep `compressionDelegate` as `nil` in `.default`.
- Document that `AgentContext.init` (Phase 6) creates the real delegate when it initializes the Metal engines.
- For Phase 5, tests create the delegate explicitly by passing a `CompressionEngine` instance.

#### Tests (`CompressionEngineTests.swift` — first section)

| # | Test | Setup | Assertion |
|---|---|---|---|
| 1 | Extractive compress — budget met | 500-token paragraph (5–6 sentences), target 150 tokens | Output token count <= 150 |
| 2 | Extractive compress — best sentence kept | Same paragraph, identify most important sentence via rankSentences | That sentence appears in output |
| 3 | Extractive compress — order preserved | 5 sentences, compress to 3 | Surviving sentences appear in same order as original |
| 4 | Extractive compress — already fits | 100-token text, target 200 | Output == input (unchanged) |
| 5 | Extractive compress — single sentence exceeds target | 1-sentence text of 300 tokens, target 100 | Returns that sentence anyway (minimum 1) |
| 6 | extractFacts — sentence split | "First fact. Second fact. Third fact!" | Returns ["First fact.", "Second fact.", "Third fact!"] |
| 7 | extractFacts — empty string | "" | Returns [] |
| 8 | Mock LLM delegate | MockCompressionDelegate injected, compress called | Delegate's compress method invoked with correct args |

**Test paragraph** — use real English text with clearly unequal importance:

```swift
let testParagraph = """
Swift is a powerful programming language developed by Apple. \
It was first released in 2014 as a replacement for Objective-C. \
Swift uses LLVM for compilation and achieves performance comparable to C++. \
The weather in Cupertino is generally mild year-round. \
Swift's type system prevents common programming errors at compile time. \
Memory management in Swift is handled automatically through ARC.
"""
// "The weather in Cupertino..." should rank lowest in importance
// when the chunk embedding represents a programming language topic.
```

**MockCompressionDelegate** for test 8:

```swift
final class MockCompressionDelegate: CompressionDelegate, @unchecked Sendable {
    var compressCalled = false
    var lastText: String?
    var lastTargetTokens: Int?

    func compress(_ text: String, targetTokens: Int) async throws -> String {
        compressCalled = true
        lastText = text
        lastTargetTokens = targetTokens
        return String(text.prefix(targetTokens * 4))  // crude mock
    }

    func extractFacts(from text: String) async throws -> [String] {
        return [text]
    }
}
```

**Commit:** `feat(phase5): 5.1 — CompressionDelegate protocol with extractive fallback`

---

### 5.2 — Full Compression Engine

Extend the existing `CompressionEngine` actor (from Phase 2.4) with methods that compress entire chunks and turns.

#### Extensions to CompressionEngine

Add these to the existing `CompressionEngine` actor in `ContextCoreEngine/CompressionEngine.swift`:

```swift
// Add to existing CompressionEngine actor
extension CompressionEngine {
    // --- NEW: these properties need to be added ---

    /// The delegate that performs actual compression (extractive or LLM-based).
    /// Set during init or via a configure method.
    private var compressionDelegate: any CompressionDelegate

    /// Token counter for measuring compression results.
    private var tokenCounter: any TokenCounter
}
```

Since you can't add stored properties in extensions on actors, restructure the actor init to accept these:

```swift
public actor CompressionEngine {
    private let device: MTLDevice
    private let sentenceImportancePipeline: MTLComputePipelineState
    let embeddingProvider: any EmbeddingProvider   // internal access for ExtractiveFallbackDelegate
    private let tokenCounter: any TokenCounter
    private var compressionDelegate: (any CompressionDelegate)?

    public init(
        device: MTLDevice? = nil,
        embeddingProvider: any EmbeddingProvider,
        tokenCounter: any TokenCounter,
        compressionDelegate: (any CompressionDelegate)? = nil
    ) throws

    // Phase 2.4 method (already exists)
    public func rankSentences(in chunk: String, chunkEmbedding: [Float]) async throws -> [(sentence: String, importance: Float)]

    // Phase 5 methods (NEW)
    public func compress(chunk: MemoryChunk, targetTokens: Int) async throws -> MemoryChunk
    public func compressTurn(turn: Turn, targetTokens: Int) async throws -> Turn

    /// Update the compression delegate (e.g. when consumer injects LLM-based one).
    public func setCompressionDelegate(_ delegate: any CompressionDelegate)
}
```

#### `compress(chunk:targetTokens:)`

```
func compress(chunk: MemoryChunk, targetTokens: Int) async throws -> MemoryChunk:

    let currentTokens = tokenCounter.count(chunk.content)

    // Already fits — return unchanged
    if currentTokens <= targetTokens:
        return chunk

    // Determine which delegate to use
    let delegate = compressionDelegate ?? makeDefaultExtractiveDelegate()

    // Compress via delegate
    let compressedContent = try await delegate.compress(chunk.content, targetTokens: targetTokens)

    // Build new MemoryChunk with compressed content
    var compressed = chunk
    compressed.content = compressedContent

    // Re-embed the compressed content
    compressed.embedding = try await embeddingProvider.embed(compressedContent)

    // Track compression ratio in metadata
    let newTokens = tokenCounter.count(compressedContent)
    let ratio = Float(currentTokens) / Float(max(newTokens, 1))
    compressed.metadata["compressionRatio"] = String(format: "%.2f", ratio)
    compressed.metadata["originalTokenCount"] = "\(currentTokens)"

    return compressed
```

**Key decisions**:
- The compressed chunk gets a **new embedding** — the old embedding represented the original content and is no longer accurate.
- **Compression ratio** is stored as metadata for observability. A ratio of 4.0 means the content was reduced to 25% of original size.
- The chunk **retains its original ID** — it's the same memory, just compressed. Consumers can track what happened via `compressionRatio` metadata.

#### `compressTurn(turn:targetTokens:)`

Same pipeline applied to a `Turn`:

```swift
public func compressTurn(turn: Turn, targetTokens: Int) async throws -> Turn {
    let currentTokens = tokenCounter.count(turn.content)

    if currentTokens <= targetTokens {
        return turn
    }

    let delegate = compressionDelegate ?? makeDefaultExtractiveDelegate()
    let compressedContent = try await delegate.compress(turn.content, targetTokens: targetTokens)
    let newEmbedding = try await embeddingProvider.embed(compressedContent)
    let newTokenCount = tokenCounter.count(compressedContent)

    var compressed = turn
    compressed.tokenCount = newTokenCount
    compressed.embedding = newEmbedding
    // Turn.content is let — you may need to make it var, or create a new Turn
    // If Turn.content is immutable, construct a new Turn:
    return Turn(
        id: turn.id,
        role: turn.role,
        content: compressedContent,
        timestamp: turn.timestamp,
        tokenCount: newTokenCount,
        embedding: newEmbedding,
        metadata: turn.metadata.merging(
            ["compressionRatio": String(format: "%.2f", Float(currentTokens) / Float(max(newTokenCount, 1)))],
            uniquingKeysWith: { _, new in new }
        )
    )
}
```

**Note on Turn mutability**: Check Phase 1's `Turn` definition. If `content` is `let`, you must construct a new `Turn` instance. Do not change `content` to `var` unless it was already `var` — that would break Phase 1's design intent. The implementation above handles both cases.

#### `makeDefaultExtractiveDelegate()`

A private helper that lazily creates an `ExtractiveFallbackDelegate` when no delegate was injected:

```swift
private func makeDefaultExtractiveDelegate() -> ExtractiveFallbackDelegate {
    ExtractiveFallbackDelegate(compressionEngine: self, tokenCounter: tokenCounter)
}
```

**Circular reference note**: `ExtractiveFallbackDelegate` holds a reference to `CompressionEngine`, and `CompressionEngine` can create an `ExtractiveFallbackDelegate`. This is not a retain cycle because:
- The delegate is created on-demand, not stored.
- If a delegate *is* stored (via `compressionDelegate` property), it's the consumer's injected delegate, not the default.

If you want to avoid any ambiguity, make `makeDefaultExtractiveDelegate()` create a fresh instance each call rather than caching it.

#### Tests (`CompressionEngineTests.swift` — second section)

| # | Test | Setup | Assertion |
|---|---|---|---|
| 9 | compress(chunk) — basic | 400-token chunk, target 100, extractive fallback | Output chunk content <= 100 tokens |
| 10 | compress(chunk) — best sentence present | Same chunk | Most important sentence (from rankSentences) in output content |
| 11 | compress(chunk) — already under target | 50-token chunk, target 200 | Returned chunk == original (same content, no metadata changes) |
| 12 | compress(chunk) — metadata set | 400-token → 100-token | `compressed.metadata["compressionRatio"]` exists and is "4.00" ± 0.5 |
| 13 | compress(chunk) — embedding updated | Compare original and compressed embeddings | Embeddings differ (content changed, so embedding must change) |
| 14 | compress(chunk) — with LLM delegate | Inject MockCompressionDelegate | Mock's `compress` called, returned content used |
| 15 | compressTurn — basic | 400-token turn, target 100 | Output turn content <= 100 tokens |
| 16 | compressTurn — identity preserved | Same turn | `compressed.id == original.id`, `compressed.role == original.role` |
| 17 | compressTurn — tokenCount updated | Same turn | `compressed.tokenCount` matches `tokenCounter.count(compressed.content)` |
| 18 | compressTurn — already fits | 50-token turn, target 200 | Returned turn == original |

**Commit:** `feat(phase5): 5.2 — Full compression engine with delegate routing`

---

### 5.3 — Progressive Compression

When the `WindowPacker` has a token deficit across multiple chunks, the `ProgressiveCompressor` decides which chunks to compress and how aggressively — always starting with the least important chunks.

#### Mental Model

Think of it as a budget negotiation. You're 600 tokens over budget. Rather than dropping the least important chunk entirely (losing all its information), you first try gentle compression (50% reduction). If that's not enough savings, you compress harder (75%). Only if both levels fail do you drop the chunk. And you move to the next chunk only after exhausting all levels on the current one.

This maximizes information retention — every chunk gets the lightest compression that covers the deficit.

#### CompressionLevel ↔ Target Mapping

```
CompressionLevel.none   → target = original tokens (no compression)
CompressionLevel.light  → target = original tokens * 0.50 (50% reduction)
CompressionLevel.heavy  → target = original tokens * 0.25 (75% reduction)
CompressionLevel.dropped → target = 0 (evicted, saves all tokens)
```

#### Implementation

```swift
public actor ProgressiveCompressor {
    private let compressionEngine: CompressionEngine
    private let tokenCounter: any TokenCounter

    public init(
        compressionEngine: CompressionEngine,
        tokenCounter: any TokenCounter
    )

    /// Progressively compress chunks to cover a token deficit.
    ///
    /// - Parameters:
    ///   - candidates: Chunks sorted by eviction score ASCENDING (lowest score = compress first).
    ///   - tokenDeficit: How many tokens need to be saved.
    ///
    /// - Returns: Compressed chunks with their final compression level and tokens saved.
    ///            Chunks not touched are returned unchanged with compressionLevel = .none.
    public func compress(
        candidates: [(chunk: MemoryChunk, evictionScore: Float)],
        tokenDeficit: Int
    ) async throws -> [ProgressiveCompressionResult]
}

public struct ProgressiveCompressionResult: Sendable {
    public let originalChunk: MemoryChunk
    public let compressedContent: String?  // nil if dropped
    public let compressionLevel: CompressionLevel
    public let originalTokens: Int
    public let compressedTokens: Int       // 0 if dropped
    public let tokensSaved: Int            // originalTokens - compressedTokens

    /// Convert to ContextChunk for inclusion in the window.
    public func toContextChunk(score: Float, source: MemoryType) -> ContextChunk?
    // Returns nil if compressionLevel == .dropped
}
```

#### Algorithm

```
func compress(candidates, tokenDeficit) -> [ProgressiveCompressionResult]:

    var remainingDeficit = tokenDeficit
    var results: [ProgressiveCompressionResult] = []

    // If no deficit, return all unchanged
    if remainingDeficit <= 0:
        return candidates.map { makeUnchangedResult($0) }

    for (chunk, score) in candidates:  // already sorted ascending by score

        if remainingDeficit <= 0:
            // Deficit covered — remaining chunks pass through unchanged
            results.append(makeUnchangedResult(chunk, score))
            continue

        let originalTokens = tokenCounter.count(chunk.content)

        // Try Level 1: 50% reduction
        let level1Target = originalTokens / 2
        let level1Content = try await compressionEngine.compress(
            chunk: chunk,
            targetTokens: level1Target
        )
        let level1Tokens = tokenCounter.count(level1Content.content)
        let level1Saved = originalTokens - level1Tokens

        if level1Saved >= remainingDeficit:
            // Level 1 is enough
            remainingDeficit -= level1Saved
            results.append(ProgressiveCompressionResult(
                originalChunk: chunk,
                compressedContent: level1Content.content,
                compressionLevel: .light,
                originalTokens: originalTokens,
                compressedTokens: level1Tokens,
                tokensSaved: level1Saved
            ))
            continue

        // Try Level 2: 75% reduction
        let level2Target = originalTokens / 4
        let level2Content = try await compressionEngine.compress(
            chunk: chunk,
            targetTokens: level2Target
        )
        let level2Tokens = tokenCounter.count(level2Content.content)
        let level2Saved = originalTokens - level2Tokens

        if level2Saved >= remainingDeficit:
            // Level 2 is enough
            remainingDeficit -= level2Saved
            results.append(ProgressiveCompressionResult(
                originalChunk: chunk,
                compressedContent: level2Content.content,
                compressionLevel: .heavy,
                originalTokens: originalTokens,
                compressedTokens: level2Tokens,
                tokensSaved: level2Saved
            ))
            continue

        // Level 3: drop entirely
        remainingDeficit -= originalTokens
        results.append(ProgressiveCompressionResult(
            originalChunk: chunk,
            compressedContent: nil,
            compressionLevel: .dropped,
            originalTokens: originalTokens,
            compressedTokens: 0,
            tokensSaved: originalTokens
        ))

    return results
```

**Key invariant**: After the loop, `remainingDeficit <= 0` if there was enough total content to cover the deficit. If all chunks are dropped and deficit still remains, the caller (`WindowPacker`) knows it can't fit everything.

#### Integration with WindowPacker

After Phase 5, update `WindowPacker.pack()` (Phase 3.2) to use `ProgressiveCompressor` instead of its simpler inline compression. This is a targeted change:

```swift
// In WindowPacker.pack(), replace the compression fallback block with:
if totalOverBudget > 0 {
    let compressionResults = try await progressiveCompressor.compress(
        candidates: overBudgetChunks.sorted(by: { $0.evictionScore < $1.evictionScore }),
        tokenDeficit: totalOverBudget
    )
    // Replace chunks with their compressed versions
    for result in compressionResults {
        if let contextChunk = result.toContextChunk(score: ..., source: ...) {
            packedChunks.append(contextChunk)
        }
        // dropped chunks are simply not appended
    }
}
```

This integration is optional for Phase 5 — it can also be done in Phase 6 when `AgentContext` wires everything together. If you do it now, add an integration test.

#### Tests (`ProgressiveCompressionTests.swift`)

| # | Test | Setup | Assertion |
|---|---|---|---|
| 1 | Deficit covered at Level 1 | 10 chunks (200 tokens each = 2000 total), deficit 600 | Cumulative tokensSaved >= 600. Some chunks have .light, rest have .none |
| 2 | Lowest scored first | Chunks with scores [0.1, 0.3, 0.5, 0.7, 0.9], deficit 200 | Chunk with score 0.1 compressed first, score 0.9 untouched |
| 3 | Level escalation | 1 chunk of 200 tokens, deficit 180 | Level 1 (50%) saves ~100 tokens — not enough. Level 2 (75%) saves ~150 — not enough. Level 3 (drop) saves 200 — deficit covered. Final level = .dropped |
| 4 | Level 1 before Level 2 | 1 chunk, deficit 80 | Level 1 saves ~100 tokens, sufficient. compressionLevel == .light (never escalated to .heavy) |
| 5 | Multiple chunks compressed | 5 chunks of 200 tokens each, deficit 400 | First 2–3 chunks compressed, remaining untouched |
| 6 | All chunks dropped | 3 chunks of 100 tokens each, deficit 500 | All 3 dropped (300 saved), remainingDeficit still > 0 |
| 7 | Zero deficit | 5 chunks, deficit 0 | All chunks returned with .none, no compression applied |
| 8 | compressionLevel correct | Mixed results | Each result's compressionLevel matches the level that was applied |
| 9 | tokensSaved accounting | Full run | `results.map(\.tokensSaved).reduce(0, +) >= deficit` |
| 10 | toContextChunk — dropped | Result with .dropped | Returns nil |
| 11 | toContextChunk — light | Result with .light | Returns ContextChunk with compressed content and .light level |
| 12 | Order unchanged | 5 chunks in, 5 results out | Results in same order as input candidates |

**Generating test chunks with controllable token counts**:

```swift
/// Create a MemoryChunk with content that has approximately `targetTokens` tokens.
func makeChunkWithTokens(_ targetTokens: Int, score: Float = 0.5) -> MemoryChunk {
    // Generate repeating sentences to hit the target
    let sentence = "This is a test sentence for compression evaluation purposes. "
    let tokensPerSentence = tokenCounter.count(sentence)
    let repetitions = max(1, targetTokens / tokensPerSentence)
    let content = String(repeating: sentence, count: repetitions)

    return MemoryChunk(
        id: UUID(),
        content: content,
        embedding: TestHelpers.randomVector(dim: 384, seed: UInt64(targetTokens)),
        type: .episodic,
        createdAt: .now,
        lastAccessedAt: .now,
        accessCount: 1,
        retentionScore: 0.5,
        sourceSessionID: UUID(),
        metadata: [:]
    )
}
```

**Commit:** `feat(phase5): 5.3 — Progressive compression with level escalation`

---

## Design Decisions to Be Aware Of

### Why Extractive Over Abstractive by Default?

Abstractive compression (LLM summarization) produces better results but requires an LLM call — which adds latency and creates a dependency ContextCore explicitly avoids. The extractive fallback runs in < 5ms (Metal sentence ranking + greedy selection) and produces acceptable results for most agent contexts. Consumers who want abstractive quality inject their own `CompressionDelegate`.

### Why Re-Embed After Compression?

The compressed content has different semantics than the original. If a 6-sentence chunk is reduced to 2 sentences, the original embedding (representing all 6 sentences) is misleading for future similarity searches. Re-embedding ensures the compressed chunk's vector accurately represents its actual content.

### Why Not Compress In-Place in the Memory Store?

Compression is a view-layer operation. The episodic store keeps the original, uncompressed chunk — future retrieval might score it differently for a different task. Compression only happens when building a specific context window. The same chunk might be fully included in one window and compressed in another.

### Progressive vs. Uniform Compression

An alternative design would compress all chunks equally (e.g., everyone gets 30% reduction). Progressive is better because:
- Low-scored chunks lose less information when compressed (they were barely relevant anyway).
- High-scored chunks stay pristine — they're the most important context.
- The escalation ladder (.light → .heavy → .dropped) gives finer-grained budget control than binary include/exclude.

---

## Final Verification

After all 3 sub-tasks:

```bash
swift build 2>&1      # zero errors, zero warnings
swift test 2>&1       # all Phase 1–5 tests green
```

Report: total test count, pass/fail, compression ratios achieved in tests.

## Phase 5 Is Done When

- `CompressionDelegate` protocol has both methods (`compress`, `extractFacts`) fully specified
- `ExtractiveFallbackDelegate` compresses text to within target tokens using sentence importance ranking
- Extractive output preserves original sentence order
- `CompressionEngine.compress(chunk:)` and `compressTurn(turn:)` work end-to-end
- Compressed chunks have updated embeddings and `compressionRatio` metadata
- Chunks already under target are returned unchanged (no unnecessary Metal dispatch)
- `ProgressiveCompressor` escalates through levels: .light → .heavy → .dropped
- Lowest-scored chunks are compressed first; highest-scored stay untouched
- Cumulative `tokensSaved >= tokenDeficit` when sufficient content exists
- Mock LLM delegate can be injected and is called correctly
- Phase 1–4 tests still pass (no regressions)
- 3 clean atomic commits in git history
