# Phase 3: Window Packer

You are implementing Phase 3 of ContextCore. Phases 1–2 are complete. You have data models (`Turn`, `MemoryChunk`), memory stores, `EmbeddingProvider`, `TokenCounter`, and four Metal scoring engines (`ScoringEngine`, `AttentionEngine`, `CompressionEngine` partial, plus recency in `ScoringEngine`). All compile and pass tests.

Phase 3 is pure CPU logic — no new Metal kernels. You are building the system that takes scored memory chunks and assembles the optimal context window that fits within a token budget. This is the bridge between "we scored everything" (Phase 2) and "here's what the model sees" (Phase 6).

---

## What You Are Building

Three components:

| Component | File | Responsibility |
|---|---|---|
| **ContextWindow** | `ContextWindow.swift` | The output type — what `buildWindow` returns to the consumer |
| **WindowPacker** | `WindowPacker.swift` | Budget-aware assembly — decides what fits |
| **ChunkOrderer** | `ChunkOrderer.swift` | Reorders packed chunks for optimal model attention |

The flow: scored chunks → **WindowPacker** (selects what fits) → **ChunkOrderer** (arranges for attention) → **ContextWindow** (formatted output).

## Why Order Matters

LLMs attend more strongly to content near the end of the context window (recency bias) and near the beginning (primacy bias). The middle gets the least attention — the "lost in the middle" problem. ContextCore exploits this:

- **System prompt** → position 0 (primacy)
- **Current user message** → last position (recency)
- **High-relevance memory** → near the end, just before current message
- **Lower-relevance context** → middle positions

The `ChunkOrderer` implements this via configurable strategies.

---

## Phase 1–2 Types You Depend On

These already exist. Do not redefine them — import and use them:

```swift
// From Phase 1
public struct Turn { id, role, content, timestamp, tokenCount, embedding, metadata }
public enum TurnRole { user, assistant, tool, system }
public struct MemoryChunk { id, content, embedding, type, createdAt, lastAccessedAt, accessCount, retentionScore, sourceSessionID, metadata }
public enum MemoryType { episodic, semantic, procedural }
public protocol TokenCounter { func count(_ text: String) -> Int }

// From Phase 2
public actor CompressionEngine {
    func rankSentences(in chunk: String, chunkEmbedding: [Float]) async throws -> [(sentence: String, importance: Float)]
}
```

---

## File Map (Phase 3 additions)

```
Sources/
  ContextCore/
    ContextWindow.swift      # NEW — ContextChunk, ContextWindow, FormatStyle
    WindowPacker.swift       # NEW — budget-aware assembly actor
    ChunkOrderer.swift       # NEW — ordering strategies
Tests/
  ContextCoreTests/
    WindowPackerTests.swift  # NEW
    ChunkOrdererTests.swift  # NEW
```

---

## Execution Plan

Three sub-tasks. TDD for each: test → implement → verify → commit.

---

### 3.1 — Context Window Type

The output types that consumers receive from `buildWindow`.

#### Types

**`ContextChunk`** (`ContextWindow.swift`):

```swift
public struct ContextChunk: Identifiable, Codable, Sendable, Hashable {
    public let id: UUID
    public let content: String
    public let role: TurnRole
    public let tokenCount: Int
    public let score: Float
    public let source: MemoryType
    public var compressionLevel: CompressionLevel
    public let timestamp: Date

    public init(
        id: UUID = UUID(),
        content: String,
        role: TurnRole,
        tokenCount: Int,
        score: Float,
        source: MemoryType,
        compressionLevel: CompressionLevel = .none,
        timestamp: Date = .now
    )
}

public enum CompressionLevel: Int, Codable, Sendable, Hashable, Comparable {
    case none = 0       // original content, uncompressed
    case light = 1      // extractive 50% reduction
    case heavy = 2      // extractive 75% reduction
    case dropped = 3    // evicted entirely
}
```

`CompressionLevel` is `Comparable` via raw value — `.none < .light < .heavy < .dropped`. This lets Phase 5's progressive compressor iterate through levels.

**`FormatStyle`**:

```swift
public enum FormatStyle: Sendable {
    case raw
    case chatML
    case alpaca
    case custom(template: String)  // placeholders: {role}, {content}
}
```

**`ContextWindow`**:

```swift
public struct ContextWindow: Sendable {
    public let chunks: [ContextChunk]
    public let totalTokens: Int
    public let budgetUsed: Float        // totalTokens / budget (0.0–1.0)
    public let budget: Int              // the budget this window was packed against
    public let retrievedFromMemory: Int // chunks that came from episodic/semantic stores
    public let compressedChunks: Int    // chunks with compressionLevel > .none

    /// Assemble chunks into a formatted string for model injection.
    public func formatted(style: FormatStyle) -> String
}
```

#### Formatting Rules

**`.raw`**: Concatenate `chunk.content` separated by `\n\n`. No role markers.

**`.chatML`**:
```
<|im_start|>system
{content}<|im_end|>
<|im_start|>user
{content}<|im_end|>
<|im_start|>assistant
{content}<|im_end|>
```

**`.alpaca`**:
- `system` role → `### Instruction:\n{content}\n\n`
- `user` role → `### Input:\n{content}\n\n`
- `assistant` role → `### Response:\n{content}\n\n`
- `tool` role → `### Tool Output:\n{content}\n\n`

**`.custom(template:)`**: Replace `{role}` with `chunk.role.rawValue` and `{content}` with `chunk.content` for each chunk. Separate chunks with `\n`.

#### Tests (`WindowPackerTests.swift` — first section)

| # | Test | Setup | Assertion |
|---|---|---|---|
| 1 | totalTokens accuracy | 3 chunks with tokenCount 100, 200, 300 | totalTokens == 600 |
| 2 | budgetUsed calculation | 3 chunks totalling 600, budget 1000 | budgetUsed == 0.6 |
| 3 | `.raw` format | 2 chunks: "Hello", "World" | Output == `"Hello\n\nWorld"` |
| 4 | `.chatML` format | 1 system chunk "You are helpful", 1 user chunk "Hi" | Output contains `<\|im_start\|>system\nYou are helpful<\|im_end\|>` and `<\|im_start\|>user\nHi<\|im_end\|>` |
| 5 | `.alpaca` format | 1 system chunk, 1 user chunk, 1 assistant chunk | Output contains `### Instruction:`, `### Input:`, `### Response:` in order |
| 6 | `.custom` format | Template: `"[{role}] {content}"`, 2 chunks | Output == `"[user] Hello\n[assistant] World"` |
| 7 | Empty window | No chunks | totalTokens == 0, formatted(.raw) == "" |
| 8 | ContextChunk Codable | Create chunk, encode JSON, decode | All fields survive roundtrip |
| 9 | CompressionLevel ordering | `.none < .light < .heavy < .dropped` | All comparisons true |
| 10 | retrievedFromMemory count | 2 episodic + 1 semantic + 1 system | retrievedFromMemory == 3 |

**Commit:** `feat(phase3): 3.1 — ContextWindow and ContextChunk types with formatting`

---

### 3.2 — Window Packer

The core budget allocation algorithm. Takes scored candidates, fits them within a token budget, and returns a `ContextWindow`.

#### Algorithm

```
func pack(systemPrompt, recentTurns, scoredMemory, budget) -> ContextWindow:

    remainingTokens = budget
    packedChunks = []

    // 1. System prompt — always included, non-negotiable
    if systemPrompt != nil:
        chunk = makeChunk(systemPrompt, role: .system, source: .semantic)
        remainingTokens -= chunk.tokenCount
        packedChunks.append(chunk)

    // 2. Recent turns — guaranteed inclusion (last N, default 3)
    for turn in recentTurns.suffix(recentTurnsGuaranteed):
        chunk = makeChunk(turn)
        remainingTokens -= chunk.tokenCount
        packedChunks.append(chunk)

    // Note: if system prompt + recent turns already exceed budget,
    // that's fine — we never drop them. Budget can go negative here.
    // The memory fill step simply won't add anything.

    // 3. Memory chunks — fill remaining budget by score (descending)
    let sorted = scoredMemory.sorted(by: { $0.score > $1.score })
    for (memChunk, score) in sorted:
        if remainingTokens < minimumChunkSize:
            break  // short-circuit

        chunk = makeChunk(memChunk, score: score)

        if chunk.tokenCount <= remainingTokens:
            // Fits whole — include as-is
            packedChunks.append(chunk)
            remainingTokens -= chunk.tokenCount

        else:
            // Too large — attempt extractive compression
            compressed = attemptCompression(memChunk, target: remainingTokens)
            if compressed != nil && compressed.tokenCount <= remainingTokens:
                packedChunks.append(compressed)
                remainingTokens -= compressed.tokenCount
            // else: drop this chunk entirely

    return ContextWindow(chunks: packedChunks, budget: budget, ...)
```

**`minimumChunkSize`**: configurable, default 50 tokens. When `remainingTokens` drops below this, stop trying to pack — any chunk small enough to fit is likely too small to be useful.

#### Compression Fallback

When a chunk is too large for the remaining budget:

1. Call `CompressionEngine.rankSentences(in: chunk.content, chunkEmbedding: chunk.embedding)`.
2. Greedily select sentences from the top (most important first) until their total token count reaches `remainingTokens`.
3. Create a new `ContextChunk` with the reduced content and `compressionLevel = .light`.
4. If even the single most important sentence exceeds `remainingTokens`, drop the chunk entirely.

The `WindowPacker` takes a `CompressionEngine` and `TokenCounter` at init:

```swift
public actor WindowPacker {
    private let compressionEngine: CompressionEngine
    private let tokenCounter: any TokenCounter
    private let minimumChunkSize: Int

    public init(
        compressionEngine: CompressionEngine,
        tokenCounter: any TokenCounter,
        minimumChunkSize: Int = 50
    )

    public func pack(
        systemPrompt: String?,
        recentTurns: [Turn],
        scoredMemory: [(chunk: MemoryChunk, score: Float)],
        budget: Int
    ) async throws -> ContextWindow
}
```

#### Helper: makeChunk

A private method that converts `Turn` or `MemoryChunk` into `ContextChunk`:

```swift
private func makeChunk(from turn: Turn) -> ContextChunk {
    ContextChunk(
        id: turn.id,
        content: turn.content,
        role: turn.role,
        tokenCount: tokenCounter.count(turn.content),
        score: 1.0,  // recent turns get max score
        source: .episodic,
        timestamp: turn.timestamp
    )
}

private func makeChunk(from memory: MemoryChunk, score: Float) -> ContextChunk {
    ContextChunk(
        id: memory.id,
        content: memory.content,
        role: .system,  // memory chunks injected as system context
        tokenCount: tokenCounter.count(memory.content),
        score: score,
        source: memory.type,
        timestamp: memory.createdAt
    )
}
```

#### Tests (`WindowPackerTests.swift` — second section)

Use `ApproximateTokenCounter` from Phase 1 for all token counting in tests.

| # | Test | Setup | Assertion |
|---|---|---|---|
| 1 | Budget respected | 20 chunks totalling ~8000 tokens, budget=4096 | `window.totalTokens <= 4096` |
| 2 | Highest-scored first | 20 chunks with distinct scores | Included chunks have higher scores than excluded chunks |
| 3 | Recent turns guaranteed | 5 recent turns (last 3 guaranteed), budget enough for all | Last 3 turns present in output regardless of score |
| 4 | System prompt always included | System prompt = 200 tokens, budget=4096 | System chunk present, `remainingBudget` reduced by 200 |
| 5 | System + recent exceed budget | System (500 tokens) + 3 turns (2000 tokens each), budget=4096 | All 4 included. No memory chunks. totalTokens > budget (guaranteed items override budget) |
| 6 | Compression fallback | 1 chunk of 800 tokens, remaining budget = 400 tokens | Chunk included with reduced content, compressionLevel == .light, tokenCount <= 400 |
| 7 | Chunk too large even after compression | 1 chunk, single sentence > remaining budget | Chunk dropped, not in output |
| 8 | Short-circuit at minimumChunkSize | Remaining budget = 30 tokens (< 50 minimum), 5 more scored chunks available | None of the 5 chunks included |
| 9 | Empty inputs | No prompt, no turns, no memory | Empty window, totalTokens == 0 |
| 10 | Budget = 0 | budget=0 | Empty window (no guaranteed items to override) — or: only guaranteed items if present |
| 11 | Single chunk fits exactly | 1 chunk of exactly 4096 tokens, budget=4096 | Included, budgetUsed ≈ 1.0 |
| 12 | retrievedFromMemory | 3 episodic + 2 semantic chunks packed | retrievedFromMemory == 5 |
| 13 | compressedChunks | 2 chunks compressed during packing | compressedChunks == 2 |
| 14 | Score ordering preserved | Pack 10 chunks, check scores of included chunks | `packedChunks.map(\.score)` is non-increasing (when filtered to memory-sourced chunks) |

**Test data factory**: Create a helper that generates scored MemoryChunks with controllable token counts and scores:

```swift
func makeScoredChunk(
    content: String = "Test content for memory chunk evaluation.",
    tokenCount: Int = 100,
    score: Float = 0.5,
    type: MemoryType = .episodic
) -> (chunk: MemoryChunk, score: Float)
```

For the compression fallback tests (6, 7), you need a `CompressionEngine` instance. Use the real one if on device, or create a `MockCompressionEngine` that returns extractive results by simply truncating sentences.

**Commit:** `feat(phase3): 3.2 — WindowPacker with budget accounting and compression fallback`

---

### 3.3 — Chunk Orderer

After packing selects *which* chunks to include, the orderer decides *where* they appear. This is a pure function — no actor needed.

#### Ordering Strategies

**`.typeGrouped`** (default) — exploits both primacy and recency bias:

```
Position 0:     System prompt (primacy — always first)
Positions 1–N:  Semantic facts (stable knowledge, anchors the context)
Positions N+1:  Episodic memory chunks (chronological within group)
Positions ...:  Procedural memory (action patterns)
Positions ...:  Recent turns (chronological)
Last position:  Current user message (recency — most attended to)
```

Within each group, sort chronologically by `timestamp`.

**`.relevanceAscending`** — exploits recency bias specifically:

All chunks sorted by `score` ascending. Lowest score first (middle, least attended). Highest score last (end, most attended). System prompt pinned at position 0 regardless.

**`.chronological`**:

All chunks sorted by `timestamp` ascending. System prompt pinned at position 0.

#### Implementation

```swift
public enum OrderingStrategy: Sendable {
    case typeGrouped
    case relevanceAscending
    case chronological
}

public struct ChunkOrderer: Sendable {
    /// Reorder chunks for optimal model attention.
    /// System prompt (role == .system with source == .semantic) is always pinned at position 0.
    public func order(
        _ chunks: [ContextChunk],
        strategy: OrderingStrategy = .typeGrouped
    ) -> [ContextChunk]
}
```

The orderer is a struct, not an actor — it's a pure function with no mutable state.

#### Type-Grouped Ordering Detail

To implement `.typeGrouped`, partition chunks into buckets:

```swift
let system = chunks.filter { $0.role == .system }
let semantic = chunks.filter { $0.source == .semantic && $0.role != .system }
let episodic = chunks.filter { $0.source == .episodic }
let procedural = chunks.filter { $0.source == .procedural }
let recentTurns = chunks.filter { /* flagged as recent turn during packing */ }
```

Problem: `ContextChunk` doesn't currently distinguish "retrieved from memory" from "recent turn" if both are `.episodic`. Add a field:

```swift
public struct ContextChunk {
    // ... existing fields ...
    public let isGuaranteedRecent: Bool  // true for recent turns included by recency guarantee
}
```

This lets the orderer separate recent turns from retrieved episodic memory. Default to `false`. The `WindowPacker` sets it to `true` for the `recentTurnsGuaranteed` chunks.

#### Tests (`ChunkOrdererTests.swift`)

| # | Test | Setup | Assertion |
|---|---|---|---|
| 1 | `.typeGrouped` basic | 1 system, 2 semantic, 3 episodic, 1 procedural, 2 recent turns | Order: system → semantic → episodic → procedural → recent turns |
| 2 | `.typeGrouped` chronological within group | 3 episodic chunks with timestamps t1 < t2 < t3 | Within episodic group: t1, t2, t3 |
| 3 | `.relevanceAscending` | 5 chunks with scores [0.9, 0.3, 0.7, 0.1, 0.5] | Output order by score: 0.1, 0.3, 0.5, 0.7, 0.9 |
| 4 | `.relevanceAscending` system pinned | System chunk (score 0.0) + 3 others | System stays at index 0 even though its score isn't the lowest |
| 5 | `.chronological` | 5 chunks with random timestamps | Sorted ascending by timestamp |
| 6 | `.chronological` system pinned | System chunk with newest timestamp | System still at index 0 |
| 7 | Single chunk | 1 chunk, any strategy | Returns `[chunk]` unchanged |
| 8 | Empty input | No chunks | Returns `[]` |
| 9 | All same type | 5 episodic chunks | `.typeGrouped` falls back to chronological sort within the single group |
| 10 | `isGuaranteedRecent` separation | 2 episodic memory chunks + 2 recent turns (isGuaranteedRecent=true) | `.typeGrouped` places episodic memory before recent turns |

**Commit:** `feat(phase3): 3.3 — ChunkOrderer with three ordering strategies`

---

## Integration Point

After Phase 3, the full pipeline from scoring to output is wirable:

```swift
// Phase 2 outputs
let scoredMemory: [(MemoryChunk, Float)] = await scoringEngine.scoreChunks(...)
let recencyWeights: [Float] = await scoringEngine.computeRecencyWeights(...)

// Phase 3 pipeline
let window = try await windowPacker.pack(
    systemPrompt: "You are a helpful assistant.",
    recentTurns: lastThreeTurns,
    scoredMemory: scoredMemory,
    budget: 4096
)
let ordered = chunkOrderer.order(window.chunks, strategy: .typeGrouped)
let formatted = ContextWindow(/* re-wrap with ordered chunks */).formatted(style: .chatML)

// → formatted string goes to the model
```

Phase 6 (`AgentContext`) will wire this end-to-end. For now, verify each component independently.

---

## Final Verification

After all 3 sub-tasks:

```bash
swift build 2>&1      # zero errors, zero warnings
swift test 2>&1       # all Phase 1 + Phase 2 + Phase 3 tests green
```

Report: total test count, pass/fail.

## Phase 3 Is Done When

- `ContextChunk` and `ContextWindow` are `Codable`, `Sendable`, `Hashable`
- `formatted(style:)` produces correct output for all 4 styles
- `WindowPacker` respects token budget — output never exceeds budget (except for guaranteed items)
- `WindowPacker` includes highest-scored chunks first, compresses when possible, drops when necessary
- `WindowPacker` always includes system prompt and last N turns
- `ChunkOrderer` correctly implements all 3 strategies with system prompt pinning
- `CompressionLevel` is `Comparable` and flows through the packing pipeline
- `isGuaranteedRecent` flag separates recent turns from retrieved episodic memory
- Phase 1 and Phase 2 tests still pass (no regressions)
- 3 clean atomic commits in git history
