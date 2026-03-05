# Phase 1: Foundation & Data Model

You are implementing Phase 1 of ContextCore — a Swift package that manages context windows for on-device AI agents using Metal GPU shaders. Phase 1 builds the compilable package scaffold, all data types, the embedding pipeline, and token counting. No Metal kernels yet.

---

## What ContextCore Is

ContextCore sits between an agent's reasoning loop and the LLM's context window. Before every model call, it answers three questions on GPU: what to retrieve from memory, what to compress or drop from the current window, and how to arrange what survives. It depends on MetalANNS for vector indexing. Everything else is new work.

ContextCore is NOT a RAG framework. It manages a living, evolving context window across an entire agent session — it knows about turns, task state, memory types, token budgets, and temporal relevance.

## Four Memory Types

These have different retention and retrieval semantics. Understand them before writing any code:

| Type | What it stores | Backed by | Retrieved by |
|---|---|---|---|
| **Working** | Current active context window | Token-budgeted sliding buffer | N/A (it IS the window) |
| **Episodic** | Individual turns and tool results | MetalANNS index | Semantic similarity to current task |
| **Semantic** | Consolidated facts, preferences, patterns | MetalANNS index (higher retention) | Semantic similarity (more aggressive) |
| **Procedural** | Successful tool call sequences | Dictionary (no embeddings) | Task type string prefix match |

---

## Architecture Constraints

- **4 package targets**: `ContextCoreShaders` (empty this phase), `ContextCoreEngine` (EmbeddingCache only this phase), `ContextCore` (all data models + protocols), `ContextCoreTests`
- **Platforms**: iOS 17.0+, macOS 14.0+, visionOS 1.0+
- **Dependency**: MetalANNS — `https://github.com/chriskarani/MetalANNS.git`, branch `main`
- **Frameworks**: Metal, CoreML, Accelerate
- **Swift 6.2** strict concurrency. All stores are actors. All public types conform to `Codable`, `Sendable`, `Hashable`.
- **Default embeddings**: MiniLM-L6-v2 quantized to CoreML, dim=384, bundled in `Resources/Embeddings/`
- **TDD mandatory**: Write every test before its implementation. Red-green-refactor.

## File Map

```
Sources/
  ContextCoreShaders/
    (empty placeholder — .metal files added in Phase 2)
  ContextCoreEngine/
    EmbeddingCache.swift
  ContextCore/
    Turn.swift
    ContextConfiguration.swift
    Errors.swift
    Memory/
      MemoryChunk.swift
      EpisodicStore.swift
      SemanticStore.swift
      ProceduralStore.swift
    Protocols/
      EmbeddingProvider.swift
      TokenCounter.swift
Tests/
  ContextCoreTests/
    TurnTests.swift
    MemoryStoreTests.swift
    EmbeddingTests.swift
    TokenCounterTests.swift
```

---

## Execution Plan

Work through these 6 sub-tasks in order. Each follows the same loop:

1. Write the test file (red).
2. Write the implementation (green).
3. Run `swift build && swift test`.
4. Commit.

---

### 1.1 — Package Scaffold

Create the directory structure and `Package.swift`. No tests — just compile.

**Package.swift requirements:**
- 4 targets with correct inter-target dependencies: `ContextCore` depends on `ContextCoreEngine`; `ContextCoreEngine` depends on `ContextCoreShaders`; `ContextCoreTests` depends on `ContextCore`
- MetalANNS package dependency: `ContextCore` and `ContextCoreEngine` depend on `MetalANNS`
- Platform minimums: `.iOS(.v17)`, `.macOS(.v14)`, `.visionOS(.v1)`
- Resource processing: `.process("Shaders")` on ContextCoreShaders, `.process("Resources")` on ContextCore
- System framework linking: Metal, CoreML, Accelerate

**Create placeholder files** in each target directory so they compile (empty exports are fine).

**Exit criteria:** `swift build` succeeds with zero errors and zero warnings.

**Commit:** `feat(phase1): 1.1 — Package scaffold with MetalANNS dependency`

---

### 1.2 — Turn Data Model

**Types to implement** (in `Turn.swift`):

```swift
public enum TurnRole: String, Codable, Sendable, Hashable {
    case user, assistant, tool, system
}

public struct Turn: Identifiable, Codable, Sendable, Hashable {
    public let id: UUID
    public let role: TurnRole
    public let content: String
    public let timestamp: Date
    public var tokenCount: Int
    public var embedding: [Float]?
    public var metadata: [String: String]

    public init(
        id: UUID = UUID(),
        role: TurnRole,
        content: String,
        timestamp: Date = .now,
        tokenCount: Int = 0,
        embedding: [Float]? = nil,
        metadata: [String: String] = [:]
    )
}

public struct ToolCall: Codable, Sendable, Hashable {
    public let name: String
    public let input: String
    public let output: String
    public let durationMs: Double
}
```

**Tests** (`TurnTests.swift`):

| # | Test | Assertion |
|---|---|---|
| 1 | Create Turn, encode to JSON, decode | All fields match original exactly |
| 2 | Create ToolCall, encode to JSON, decode | All fields match original exactly |
| 3 | Encode ToolCall to JSON string, store in Turn.metadata["toolCall"], roundtrip the Turn | ToolCall decodes correctly from metadata value |
| 4 | Two Turns with different UUIDs | `turn1 != turn2` |
| 5 | Two Turns with same UUID | `turn1 == turn2` (Hashable contract) |

**Commit:** `feat(phase1): 1.2 — Turn data model with Codable roundtrip tests`

---

### 1.3 — Memory Types

**MemoryChunk** (`Memory/MemoryChunk.swift`):

```swift
public enum MemoryType: String, Codable, Sendable, Hashable {
    case episodic, semantic, procedural
}

public struct MemoryChunk: Identifiable, Codable, Sendable, Hashable {
    public let id: UUID
    public var content: String
    public var embedding: [Float]
    public let type: MemoryType
    public let createdAt: Date
    public var lastAccessedAt: Date
    public var accessCount: Int
    public var retentionScore: Float
    public let sourceSessionID: UUID
    public var metadata: [String: String]
}
```

**EpisodicStore** (`Memory/EpisodicStore.swift`) — actor:

| Method | Behavior |
|---|---|
| `insert(turn: Turn) async throws` | Create MemoryChunk from turn (requires turn.embedding to be non-nil), insert into MetalANNS index, increment count |
| `retrieve(query: [Float], k: Int) async throws -> [MemoryChunk]` | kNN search via MetalANNS, return k nearest chunks |
| `var count: Int` | Number of stored chunks |

Default retention score for episodic: **0.5**

**SemanticStore** (`Memory/SemanticStore.swift`) — actor:

Same as EpisodicStore plus:

| Method | Behavior |
|---|---|
| `upsert(fact: String, embedding: [Float]) async throws` | Search for existing chunk with cosine similarity > 0.9. If found: increment accessCount, update lastAccessedAt. If not found: insert new chunk. |

Default retention score for semantic: **1.0**

**ProceduralStore** (`Memory/ProceduralStore.swift`) — actor:

| Method | Behavior |
|---|---|
| `record(taskType: String, tools: [ToolCall]) async` | Store tools under taskType key. If at 1000 entries, evict least-recently-accessed entry first. |
| `retrieve(taskType: String) async -> [ToolCall]` | Return tools for all keys where `key.hasPrefix(taskType)`. Return empty array if no match. |
| `var count: Int` | Number of stored task type entries |

**Tests** (`MemoryStoreTests.swift`):

| # | Test | Assertion |
|---|---|---|
| 1 | MemoryChunk JSON roundtrip | All fields survive encode/decode |
| 2 | EpisodicStore: insert 10 chunks, retrieve with query vector, k=5 | Results non-empty, count <= 5 |
| 3 | EpisodicStore: insert 3 chunks | count == 3 |
| 4 | SemanticStore: insert 10 chunks, retrieve | Results non-empty |
| 5 | SemanticStore: upsert same fact twice (similarity > 0.9) | count stays at 1, accessCount increments to 2 |
| 6 | ProceduralStore: record 5 types, retrieve by exact key | Returns correct [ToolCall] |
| 7 | ProceduralStore: record "code.swift.format" and "code.swift.lint", retrieve "code.swift" | Returns tools from both entries |
| 8 | ProceduralStore: insert 1001 entries | count == 1000, oldest entry evicted |

**Commit:** `feat(phase1): 1.3 — Memory chunk model and three store actors`

---

### 1.4 — Embedding Provider

**Protocol** (`Protocols/EmbeddingProvider.swift`):

```swift
public protocol EmbeddingProvider: Sendable {
    func embed(_ text: String) async throws -> [Float]
    func embedBatch(_ texts: [String]) async throws -> [[Float]]
    var dimension: Int { get }
}
```

**CoreMLEmbeddingProvider** (internal):
- Loads `minilm-l6-v2.mlpackage` from bundle at init.
- `embed(_:)` uses CoreML async prediction.
- `embedBatch(_:)` uses `MLBatchProvider`.
- **Simulator fallback**: `#if targetEnvironment(simulator)` — return a deterministic vector seeded by the string's `hashValue`. This makes tests pass without the ANE. Vectors must still be dim=384 and L2-normalized.

**EmbeddingCache** (`ContextCoreEngine/EmbeddingCache.swift`) — actor:
- Generic LRU cache: `capacity` configurable (default 512).
- Key: SHA256 hash of input string (use `CryptoKit`).
- `func get(_ key: String) -> [Float]?`
- `func set(_ key: String, value: [Float])`
- On capacity overflow: evict least-recently-used entry.

**CachingEmbeddingProvider** (internal):
- Wraps any `EmbeddingProvider` + `EmbeddingCache`.
- On `embed(_:)`: check cache first. On miss: call inner provider, cache result, return.
- On `embedBatch(_:)`: check cache for each string individually. Batch-embed only the misses. Cache all results. Return in original order.

**Tests** (`EmbeddingTests.swift`):

| # | Test | Assertion |
|---|---|---|
| 1 | Embed "hello world" twice via CachingEmbeddingProvider | Vectors are identical (bitwise). Second call completes in < 1ms (measure with `ContinuousClock`). |
| 2 | embedBatch 10 unique strings | All vectors have exactly 384 dimensions. No two vectors are identical. |
| 3 | dimension property | Returns 384 |
| 4 | Fill cache to 512, insert 513th | Cache count stays at 512. First inserted key returns nil on get. |

**Commit:** `feat(phase1): 1.4 — Embedding provider protocol with CoreML backend and LRU cache`

---

### 1.5 — Token Counter & Configuration

**Protocol** (`Protocols/TokenCounter.swift`):

```swift
public protocol TokenCounter: Sendable {
    func count(_ text: String) -> Int
}
```

**ApproximateTokenCounter**:
- Split text on whitespace and punctuation boundaries.
- Multiply word count by 1.3, round up.
- Return 0 for empty string.

**ContextConfiguration** (`ContextConfiguration.swift`):

```swift
public struct ContextConfiguration: Sendable {
    public var maxTokens: Int                         // default: 4096
    public var tokenBudgetSafetyMargin: Float         // default: 0.10
    public var episodicMemoryK: Int                   // default: 8
    public var semanticMemoryK: Int                   // default: 4
    public var recentTurnsGuaranteed: Int              // default: 3
    public var episodicHalfLifeDays: Double            // default: 7
    public var semanticHalfLifeDays: Double            // default: 90
    public var consolidationThreshold: Int             // default: 200
    public var similarityMergeThreshold: Float         // default: 0.92
    public var relevanceWeight: Float                  // default: 0.7
    public var centralityWeight: Float                 // default: 0.4
    public var efSearch: Int                           // default: 64
    public var embeddingProvider: any EmbeddingProvider
    public var tokenCounter: any TokenCounter

    public static var `default`: ContextConfiguration { get }
}
```

`ContextConfiguration.default` uses `CoreMLEmbeddingProvider` wrapped in `CachingEmbeddingProvider` and `ApproximateTokenCounter`.

Forward-declare `CompressionDelegate` protocol in its own file (empty body for now — Phase 5 fills it in):

```swift
public protocol CompressionDelegate: Sendable {
    func compress(_ text: String, targetTokens: Int) async throws -> String
    func extractFacts(from text: String) async throws -> [String]
}
```

**Tests** (`TokenCounterTests.swift`):

| # | Input string | GPT-2 reference tokens | Pass condition |
|---|---|---|---|
| 1 | `"Hello, world!"` | 4 | Within 20% (3–5) |
| 2 | `"The quick brown fox jumps over the lazy dog."` | 10 | Within 20% (8–12) |
| 3 | `"Swift is a powerful and intuitive programming language."` | 9 | Within 20% (7–11) |
| 4 | `"GPU-accelerated context management for on-device AI agents"` | 10 | Within 20% (8–12) |
| 5 | `""` (empty) | 0 | Exactly 0 |

Additional tests:

| # | Test | Assertion |
|---|---|---|
| 6 | `ContextConfiguration.default.maxTokens` | == 4096 |
| 7 | `ContextConfiguration.default.tokenBudgetSafetyMargin` | == 0.10 |
| 8 | `ContextConfiguration.default.relevanceWeight` | == 0.7 |
| 9 | `ContextConfiguration.default.consolidationThreshold` | == 200 |
| 10 | Instantiate ContextConfiguration in a `Task {}`, pass across isolation | Compiles (Sendable proof) |

**Commit:** `feat(phase1): 1.5 — Token counter, configuration, and CompressionDelegate protocol`

---

### 1.6 — Error Types

**Implement** (`Errors.swift`):

```swift
public enum ContextCoreError: Error, Sendable, Equatable {
    case embeddingFailed(String)
    case storeFull
    case tokenBudgetTooSmall
    case sessionNotStarted
    case compressionFailed(String)
    case checkpointCorrupt
    case metalDeviceUnavailable
    case dimensionMismatch(expected: Int, got: Int)
}
```

Add `Equatable` conformance so tests can assert on specific error cases.

No dedicated test file — these errors are exercised by error-path tests in earlier test files. If you haven't already, add at least one error-path test somewhere (e.g. inserting a Turn with nil embedding into EpisodicStore throws `embeddingFailed`).

**Commit:** `feat(phase1): 1.6 — Error types`

---

## Final Verification

After all 6 sub-tasks, run:

```bash
swift build 2>&1      # zero errors, zero warnings
swift test 2>&1       # all tests green
```

Report: total test count, pass/fail, build time, any warnings.

## Phase 1 Is Done When

- `swift build` — zero errors, zero warnings
- `swift test` — all tests pass
- Every public type is `Codable`, `Sendable`, `Hashable`
- EmbeddingProvider works end-to-end: string in, 384-dim vector out, cache hits on repeat
- Token counter is within 20% of GPT-2 reference on all 5 test strings
- All three memory stores accept inserts and return results on retrieval
- MetalANNS resolves and links without manual intervention
- 6 clean atomic commits in git history
