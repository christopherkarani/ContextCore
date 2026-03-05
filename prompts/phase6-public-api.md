# Phase 6: Public API & AgentContext

You are implementing Phase 6 of ContextCore — the final integration phase. Phases 1–5 are complete. Every subsystem exists and passes tests independently: data models, memory stores, Metal scoring engines, window packing, chunk ordering, consolidation, and compression. Phase 6 wires everything behind a single public actor that consumers interact with.

After this phase, a consumer can integrate ContextCore with 3 method calls: `append`, `buildWindow`, and `checkpoint`. Everything else — embedding, scoring, packing, compression, consolidation — happens internally.

---

## The Consumer's View

This is what using ContextCore looks like after Phase 6:

```swift
// Setup (once)
let context = try AgentContext()
try await context.beginSession(id: UUID(), systemPrompt: "You are a helpful assistant.")

// Agent loop (every turn)
try await context.append(turn: Turn(role: .user, content: userMessage))

let window = try await context.buildWindow(
    currentTask: "Help the user debug their Swift code",
    maxTokens: 4096
)

let prompt = window.formatted(style: .chatML)
// → pass prompt to model

let response = model.generate(prompt)
try await context.append(turn: Turn(role: .assistant, content: response))

// Persistence (periodically)
try context.checkpoint(to: checkpointURL)

// Session end
try await context.endSession()  // triggers consolidation
```

The consumer never sees Metal, ANNS, scoring, or compression. That is the design goal.

---

## What You Are Building

| Component | File | Responsibility |
|---|---|---|
| **AgentContext** | `ContextCore/AgentContext.swift` | Public actor — the one entry point |
| **ContextStats** | `ContextCore/ContextStats.swift` | Observability — latency, counts, ratios |
| **ContextCheckpoint** | `ContextCore/Persistence/ContextCheckpoint.swift` | Serialization of all state |
| **SessionStore** | `ContextCore/Persistence/SessionStore.swift` | Session metadata tracking |

---

## Internal Wiring Diagram

`AgentContext` owns and orchestrates every subsystem from Phases 1–5:

```
AgentContext (public actor)
│
├── configuration: ContextConfiguration
│
├── Session State
│   ├── currentSessionID: UUID?
│   ├── systemPrompt: String?
│   ├── recentTurns: [Turn]        ← sliding window of last N turns
│   └── totalSessions: Int
│
├── Memory Stores (Phase 1)
│   ├── episodicStore: EpisodicStore
│   ├── semanticStore: SemanticStore
│   └── proceduralStore: ProceduralStore
│
├── Engines (Phases 2–5)
│   ├── scoringEngine: ScoringEngine          ← relevance + recency
│   ├── attentionEngine: AttentionEngine      ← centrality + eviction
│   ├── compressionEngine: CompressionEngine  ← sentence ranking + compress
│   ├── consolidationEngine: ConsolidationEngine
│   └── progressiveCompressor: ProgressiveCompressor
│
├── Assembly (Phase 3)
│   ├── windowPacker: WindowPacker
│   └── chunkOrderer: ChunkOrderer
│
├── Infrastructure (Phase 1)
│   ├── embeddingProvider: CachingEmbeddingProvider
│   └── tokenCounter: any TokenCounter
│
├── Scheduling (Phase 4)
│   └── consolidationScheduler: ConsolidationScheduler
│
└── Observability
    └── stats: ContextStats                   ← nonisolated read access
```

---

## File Map (Phase 6 additions and modifications)

```
Sources/
  ContextCore/
    AgentContext.swift                  # MODIFIED — full implementation (was stub)
    ContextStats.swift                 # NEW
    Persistence/
      ContextCheckpoint.swift          # NEW
      SessionStore.swift               # NEW
Tests/
  ContextCoreTests/
    AgentContextTests.swift            # NEW — integration tests
    ContextStatsTests.swift            # NEW
    PersistenceTests.swift             # NEW
```

---

## Execution Plan

Four sub-tasks. TDD for each.

---

### 6.1 — AgentContext Actor

The main public actor. Every method is thin orchestration — the real work is done by engines from prior phases.

#### Public API

```swift
public actor AgentContext {

    // ──── Lifecycle ────

    /// Create a new AgentContext with the given configuration.
    /// Initializes all internal engines and warms up Metal pipelines on a background task.
    public init(configuration: ContextConfiguration = .default) throws

    /// Restore an AgentContext from a checkpoint file.
    public static func load(from url: URL) async throws -> AgentContext

    // ──── Session Management ────

    /// Start a new session. Must be called before append or buildWindow.
    /// - Parameters:
    ///   - id: Unique session identifier.
    ///   - systemPrompt: Optional system prompt included in every buildWindow call.
    public func beginSession(id: UUID = UUID(), systemPrompt: String? = nil) async throws

    /// End the current session. Triggers consolidation of episodic memory.
    public func endSession() async throws

    // ──── Core Loop ────

    /// Append a turn to the session.
    /// Embeds content, counts tokens, inserts into episodic memory.
    /// May trigger background consolidation if thresholds are exceeded.
    public func append(turn: Turn) async throws

    /// Build the optimal context window for the current task.
    /// Scores all memory, packs within budget, orders for model attention.
    /// - Parameters:
    ///   - currentTask: Description of the current task (used for relevance scoring).
    ///   - maxTokens: Override the configured maxTokens. Pass nil to use configuration default.
    /// - Returns: A ContextWindow ready for model injection.
    public func buildWindow(
        currentTask: String,
        maxTokens: Int? = nil
    ) async throws -> ContextWindow

    // ──── Memory Operations ────

    /// Explicitly store a fact in semantic memory.
    /// Use for information the agent should always remember.
    public func remember(_ fact: String) async throws

    /// Soft-delete a memory chunk by ID.
    /// The chunk is marked for eviction but not immediately removed.
    public func forget(id: UUID) async throws

    /// Retrieve memory chunks similar to a query.
    public func recall(query: String, k: Int = 5) async throws -> [MemoryChunk]

    // ──── Maintenance ────

    /// Manually trigger consolidation of episodic memory.
    /// Normally handled automatically by the scheduler.
    public func consolidate() async throws

    /// Save all state to a checkpoint file for later restoration.
    public func checkpoint(to url: URL) throws

    // ──── Observability ────

    /// Current statistics. Readable without awaiting the actor.
    public nonisolated var stats: ContextStats { get }
}
```

#### `init(configuration:)`

```
public init(configuration: ContextConfiguration = .default) throws:

    self.configuration = configuration

    // 1. Metal device
    let device = try MetalContext.device()

    // 2. Embedding provider (wrapped in cache)
    let baseProvider = configuration.embeddingProvider
    self.embeddingProvider = CachingEmbeddingProvider(
        provider: baseProvider,
        cacheCapacity: 512
    )

    // 3. Token counter
    self.tokenCounter = configuration.tokenCounter

    // 4. Memory stores
    let dim = embeddingProvider.dimension
    self.episodicStore = try EpisodicStore(dimension: dim)
    self.semanticStore = try SemanticStore(dimension: dim)
    self.proceduralStore = ProceduralStore()

    // 5. Metal engines
    self.scoringEngine = try ScoringEngine(device: device)
    self.attentionEngine = try AttentionEngine(device: device)
    self.compressionEngine = try CompressionEngine(
        device: device,
        embeddingProvider: embeddingProvider,
        tokenCounter: tokenCounter,
        compressionDelegate: configuration.compressionDelegate
    )
    self.consolidationEngine = try ConsolidationEngine(
        device: device,
        embeddingProvider: embeddingProvider
    )

    // 6. Assembly
    self.progressiveCompressor = ProgressiveCompressor(
        compressionEngine: compressionEngine,
        tokenCounter: tokenCounter
    )
    self.windowPacker = WindowPacker(
        compressionEngine: compressionEngine,
        tokenCounter: tokenCounter
    )
    self.chunkOrderer = ChunkOrderer()

    // 7. Scheduler
    self.consolidationScheduler = ConsolidationScheduler(
        engine: consolidationEngine,
        countThreshold: configuration.consolidationThreshold,
        insertionThreshold: 50
    )

    // 8. Stats
    self._stats = ContextStats()

    // 9. Warm up Metal pipeline states on background task
    // Pipeline states are already compiled in engine inits,
    // but warm up the embedding model too
    Task.detached(priority: .background) { [embeddingProvider] in
        _ = try? await embeddingProvider.embed("warmup")
    }
```

#### `append(turn:)`

```
public func append(turn: Turn) async throws:

    guard currentSessionID != nil else {
        throw ContextCoreError.sessionNotStarted
    }

    // 1. Embed
    var enrichedTurn = turn
    if enrichedTurn.embedding == nil {
        enrichedTurn.embedding = try await embeddingProvider.embed(turn.content)
    }

    // 2. Count tokens
    if enrichedTurn.tokenCount == 0 {
        enrichedTurn.tokenCount = tokenCounter.count(turn.content)
    }

    // 3. Insert into episodic store
    try await episodicStore.insert(turn: enrichedTurn)

    // 4. Track recent turns (sliding window)
    recentTurns.append(enrichedTurn)
    if recentTurns.count > configuration.recentTurnsGuaranteed * 2 {
        // Keep some buffer beyond guaranteed count for context
        recentTurns.removeFirst(recentTurns.count - configuration.recentTurnsGuaranteed * 2)
    }

    // 5. Notify consolidation scheduler (non-blocking)
    await consolidationScheduler.notifyInsertion(
        episodicCount: await episodicStore.count,
        session: currentSessionID!,
        episodicStore: episodicStore,
        semanticStore: semanticStore
    )

    // 6. Update stats
    updateStats { stats in
        stats.episodicCount = await episodicStore.count
    }
```

#### `buildWindow(currentTask:maxTokens:)`

This is the most complex method — the entire scoring-packing-ordering pipeline in one async sequence.

```
public func buildWindow(currentTask: String, maxTokens: Int? = nil) async throws -> ContextWindow:

    guard let sessionID = currentSessionID else {
        throw ContextCoreError.sessionNotStarted
    }

    let clock = ContinuousClock()
    let startTime = clock.now

    // Effective budget with safety margin
    let rawBudget = maxTokens ?? configuration.maxTokens
    let effectiveBudget = Int(Float(rawBudget) * (1.0 - configuration.tokenBudgetSafetyMargin))

    // 1. Embed the current task
    let taskEmbedding = try await embeddingProvider.embed(currentTask)

    // 2. Get all episodic and semantic chunks
    let allEpisodic = await episodicStore.allChunks()
    let allSemantic = await semanticStore.allChunks()

    // 3. Parallel scoring — episodic and semantic in parallel
    async let episodicScoring = scoreEpisodicMemory(
        taskEmbedding: taskEmbedding,
        chunks: allEpisodic
    )
    async let semanticScoring = scoreSemanticMemory(
        taskEmbedding: taskEmbedding,
        chunks: allSemantic
    )

    let scoredEpisodic = try await episodicScoring
    let scoredSemantic = try await semanticScoring

    // 4. Retrieve procedural memory by task type
    let proceduralMemory = await proceduralStore.retrieve(taskType: currentTask)

    // 5. Merge and select top-k candidates
    let topEpisodic = Array(scoredEpisodic.prefix(configuration.episodicMemoryK))
    let topSemantic = Array(scoredSemantic.prefix(configuration.semanticMemoryK))

    var allScored: [(chunk: MemoryChunk, score: Float)] = []
    allScored.append(contentsOf: topEpisodic)
    allScored.append(contentsOf: topSemantic)

    // 6. Pack into window
    let recentN = Array(recentTurns.suffix(configuration.recentTurnsGuaranteed))
    let window = try await windowPacker.pack(
        systemPrompt: systemPrompt,
        recentTurns: recentN,
        scoredMemory: allScored,
        budget: effectiveBudget
    )

    // 7. Order chunks
    let orderedChunks = chunkOrderer.order(window.chunks, strategy: .typeGrouped)

    // 8. Build final window
    let finalWindow = ContextWindow(
        chunks: orderedChunks,
        totalTokens: orderedChunks.reduce(0) { $0 + $1.tokenCount },
        budgetUsed: Float(orderedChunks.reduce(0) { $0 + $1.tokenCount }) / Float(effectiveBudget),
        budget: effectiveBudget,
        retrievedFromMemory: orderedChunks.filter { !$0.isGuaranteedRecent && $0.role != .system }.count,
        compressedChunks: orderedChunks.filter { $0.compressionLevel != .none }.count
    )

    // 9. Update stats
    let elapsed = clock.now - startTime
    updateStats { stats in
        stats.lastBuildWindowLatencyMs = Double(elapsed.components.attoseconds) / 1_000_000_000_000_000.0
        stats.averageRelevanceScore = finalWindow.chunks.isEmpty ? 0 :
            finalWindow.chunks.map(\.score).reduce(0, +) / Float(finalWindow.chunks.count)
    }

    return finalWindow
```

#### Private scoring helpers

```swift
/// Score episodic memory: compute recency weights, then relevance scores.
private func scoreEpisodicMemory(
    taskEmbedding: [Float],
    chunks: [MemoryChunk]
) async throws -> [(chunk: MemoryChunk, score: Float)] {

    guard !chunks.isEmpty else { return [] }

    // Recency weights
    let timestamps = chunks.map(\.createdAt)
    let halfLife = configuration.episodicHalfLifeDays * 86400  // days to seconds
    let recencyWeights = try await scoringEngine.computeRecencyWeights(
        timestamps: timestamps,
        halfLife: halfLife
    )

    // Relevance scores (cosine similarity blended with recency)
    let scored = try await scoringEngine.scoreChunks(
        query: taskEmbedding,
        chunks: chunks,
        recencyWeights: recencyWeights,
        relevanceWeight: configuration.relevanceWeight,
        recencyWeight: 1.0 - configuration.relevanceWeight  // complement
    )

    return scored
}

/// Score semantic memory: longer half-life, same relevance scoring.
private func scoreSemanticMemory(
    taskEmbedding: [Float],
    chunks: [MemoryChunk]
) async throws -> [(chunk: MemoryChunk, score: Float)] {

    guard !chunks.isEmpty else { return [] }

    let timestamps = chunks.map(\.createdAt)
    let halfLife = configuration.semanticHalfLifeDays * 86400
    let recencyWeights = try await scoringEngine.computeRecencyWeights(
        timestamps: timestamps,
        halfLife: halfLife
    )

    let scored = try await scoringEngine.scoreChunks(
        query: taskEmbedding,
        chunks: chunks,
        recencyWeights: recencyWeights,
        relevanceWeight: configuration.relevanceWeight,
        recencyWeight: 1.0 - configuration.relevanceWeight
    )

    return scored
}
```

#### Session management

```swift
public func beginSession(id: UUID = UUID(), systemPrompt: String? = nil) async throws {
    // End previous session if active
    if currentSessionID != nil {
        try await endSession()
    }

    currentSessionID = id
    self.systemPrompt = systemPrompt
    recentTurns = []
    totalSessions += 1

    updateStats { stats in
        stats.totalSessions = totalSessions
    }
}

public func endSession() async throws {
    guard let sessionID = currentSessionID else {
        throw ContextCoreError.sessionNotStarted
    }

    // Trigger consolidation for this session
    try await consolidationEngine.consolidate(
        session: sessionID,
        episodicStore: episodicStore,
        semanticStore: semanticStore,
        threshold: configuration.similarityMergeThreshold
    )

    currentSessionID = nil
    systemPrompt = nil
    recentTurns = []

    updateStats { stats in
        stats.episodicCount = await episodicStore.count
        stats.semanticCount = await semanticStore.count
    }
}
```

#### Memory operations

```swift
public func remember(_ fact: String) async throws {
    let embedding = try await embeddingProvider.embed(fact)
    try await semanticStore.upsert(fact: fact, embedding: embedding)
    updateStats { stats in
        stats.semanticCount = await semanticStore.count
    }
}

public func forget(id: UUID) async throws {
    // Soft delete: set retention to 0, will be evicted on next consolidation
    try await episodicStore.updateRetentionScore(id: id, delta: -1.0)
    // Also try semantic store (consumer may not know which store holds it)
    try? await semanticStore.updateRetentionScore(id: id, delta: -1.0)
}

public func recall(query: String, k: Int = 5) async throws -> [MemoryChunk] {
    let embedding = try await embeddingProvider.embed(query)

    async let episodicResults = episodicStore.retrieve(query: embedding, k: k)
    async let semanticResults = semanticStore.retrieve(query: embedding, k: k)

    let combined = try await episodicResults + semanticResults

    // Sort by relevance and return top-k
    // Simple cosine similarity for recall (no recency blending)
    let scored = combined.map { chunk -> (MemoryChunk, Float) in
        let sim = CPUReference.cosineSimilarity(embedding, chunk.embedding)
        return (chunk, sim)
    }

    return scored
        .sorted { $0.1 > $1.1 }
        .prefix(k)
        .map(\.0)
}

public func consolidate() async throws {
    guard let sessionID = currentSessionID else {
        throw ContextCoreError.sessionNotStarted
    }

    let result = try await consolidationEngine.consolidate(
        session: sessionID,
        episodicStore: episodicStore,
        semanticStore: semanticStore,
        threshold: configuration.similarityMergeThreshold
    )

    updateStats { stats in
        stats.lastConsolidationLatencyMs = result.durationMs
        stats.episodicCount = await episodicStore.count
        stats.semanticCount = await semanticStore.count
    }
}
```

#### Tests (`AgentContextTests.swift`)

These are integration tests — they exercise the full pipeline end-to-end.

| # | Test | Setup | Assertion |
|---|---|---|---|
| 1 | Init succeeds | `let ctx = try AgentContext()` | No throw, stats show zero counts |
| 2 | Session lifecycle | beginSession → endSession | No throw. stats.totalSessions == 1 |
| 3 | Append basic | beginSession, append 5 turns | stats.episodicCount == 5 |
| 4 | Append embeds | Append turn with nil embedding | After append, turn in episodic store has non-nil embedding |
| 5 | Append without session | Append before beginSession | Throws `sessionNotStarted` |
| 6 | buildWindow — budget | Append 20 turns, buildWindow(maxTokens: 2048) | `window.totalTokens <= 2048` |
| 7 | buildWindow — recent turns | Append 10 turns, buildWindow | Last 3 turns present in window (check by ID) |
| 8 | buildWindow — system prompt | beginSession(systemPrompt: "Be helpful"), buildWindow | Window contains system chunk |
| 9 | buildWindow — semantic memory | Append 20 turns across 3 sessions (endSession triggers consolidation), begin 4th session, buildWindow | Window contains chunks with source == .semantic |
| 10 | buildWindow — safety margin | Config maxTokens=1000, margin=0.10, buildWindow | window.budget == 900 (effective) |
| 11 | buildWindow without session | buildWindow before beginSession | Throws `sessionNotStarted` |
| 12 | buildWindow — parallel scoring | 50 turns, buildWindow | Completes without deadlock (async let doesn't block) |
| 13 | remember | `ctx.remember("User prefers dark mode")` | stats.semanticCount incremented. recall("dark mode") returns the fact |
| 14 | forget | Append turn, forget(id: turn.id) | recall for that content returns empty or lower-ranked |
| 15 | recall | Append 10 diverse turns, recall("Swift concurrency") | Returns turns about concurrency, not about other topics |
| 16 | consolidate manual | Append 60 turns (50 unique + 10 duplicate pairs), consolidate | stats.semanticCount > 0, stats.episodicCount < 60 |
| 17 | Multi-session | 3 sessions × 10 turns each, buildWindow on 3rd | Window contains memory from sessions 1 and 2 |
| 18 | Auto-consolidation | Append 51 turns rapidly | Consolidation fires automatically (verify via stats or SemanticStore.count > 0) |
| 19 | endSession triggers consolidation | Append 60 turns (with duplicates), endSession | SemanticStore.count > 0 after endSession |
| 20 | Warmup | Init AgentContext, measure first buildWindow latency | Latency reasonable (pipeline states pre-compiled) |

**Test data — diverse turns for realistic testing:**

```swift
let diverseTurns: [(role: TurnRole, content: String)] = [
    (.user, "Help me set up a Swift Package Manager project"),
    (.assistant, "I'll help you create a new Swift package. First, run swift package init --type library."),
    (.user, "How do I add a dependency on another package?"),
    (.assistant, "Add a .package entry to your Package.swift dependencies array with the URL and version."),
    (.user, "What's the difference between actors and classes in Swift?"),
    (.assistant, "Actors provide data race safety through isolation. Unlike classes, only one task can access an actor's mutable state at a time."),
    (.user, "Can you explain async/await?"),
    (.assistant, "async/await lets you write asynchronous code that reads like synchronous code. Mark functions with async and call them with await."),
    (.user, "What is Metal used for?"),
    (.assistant, "Metal is Apple's GPU programming framework. It's used for graphics rendering and compute shaders that run parallel computations on the GPU."),
    // ... add more as needed
]
```

For test 9 (semantic memory present), use turns with repeated information to trigger consolidation-based fact promotion:

```swift
// Session 1
"The user's name is Chris."
// Session 2
"Chris mentioned his name earlier."
// After consolidation, "Chris" should be in semantic memory
```

**Commit:** `feat(phase6): 6.1 — AgentContext actor with full pipeline`

---

### 6.2 — ContextStats

Observability without performance cost. Stats are read-only from outside the actor — no `await` needed.

#### Type

```swift
public struct ContextStats: Sendable {
    public var episodicCount: Int = 0
    public var semanticCount: Int = 0
    public var proceduralCount: Int = 0
    public var totalSessions: Int = 0
    public var lastBuildWindowLatencyMs: Double = 0
    public var lastConsolidationLatencyMs: Double = 0
    public var averageRelevanceScore: Float = 0
    public var compressionRatio: Float = 0
}
```

#### Nonisolated Access Pattern

Swift actors don't allow `nonisolated var` for mutable stored properties directly. Use an atomic-like pattern:

```swift
public actor AgentContext {
    // Internal mutable state (actor-isolated)
    private var _statsStorage = ContextStats()

    // Nonisolated read access via a lock-free mechanism
    // Option A: Use OSAllocatedUnfairLock (iOS 16+)
    private let _statsLock = OSAllocatedUnfairLock(initialState: ContextStats())

    public nonisolated var stats: ContextStats {
        _statsLock.withLock { $0 }
    }

    // Actor-isolated writer
    private func updateStats(_ mutation: (inout ContextStats) async -> Void) {
        var current = _statsLock.withLock { $0 }
        await mutation(&current)
        _statsLock.withLock { $0 = current }
    }
}
```

`OSAllocatedUnfairLock` is the correct primitive here — it's `Sendable`, non-async, and fast. The `withLock` closure is synchronous, making the `nonisolated var stats` accessor safe without awaiting the actor.

**Alternative**: If `OSAllocatedUnfairLock` creates complications, use `Mutex` from the Swift Synchronization module (available in Swift 6.0+):

```swift
import Synchronization

private let _statsMutex = Mutex(ContextStats())

public nonisolated var stats: ContextStats {
    _statsMutex.withLock { $0 }
}
```

#### When Stats Update

| Event | Fields Updated |
|---|---|
| `append()` | episodicCount |
| `buildWindow()` | lastBuildWindowLatencyMs, averageRelevanceScore, compressionRatio |
| `consolidate()` | lastConsolidationLatencyMs, episodicCount, semanticCount |
| `remember()` | semanticCount |
| `beginSession()` | totalSessions |
| `endSession()` | episodicCount, semanticCount (post-consolidation) |

#### Tests (`ContextStatsTests.swift`)

| # | Test | Setup | Assertion |
|---|---|---|---|
| 1 | Initial stats | `let ctx = try AgentContext()` | All counts == 0, all latencies == 0 |
| 2 | episodicCount after append | Append 5 turns | `ctx.stats.episodicCount == 5` |
| 3 | semanticCount after remember | `ctx.remember("fact")` | `ctx.stats.semanticCount == 1` |
| 4 | totalSessions | Begin and end 3 sessions | `ctx.stats.totalSessions == 3` |
| 5 | buildWindow latency | Call buildWindow | `ctx.stats.lastBuildWindowLatencyMs > 0` |
| 6 | averageRelevanceScore | Call buildWindow | `ctx.stats.averageRelevanceScore` in [0, 1] |
| 7 | Nonisolated access | Read `ctx.stats` from a different actor/task without await | Compiles and returns valid stats (this is a compile-time check) |
| 8 | Stats survive rapid updates | Append 100 turns rapidly, read stats mid-stream | No crash, count is reasonable (may lag slightly due to lock timing) |

**Compile-time nonisolated proof (test 7):**

```swift
func testStatsNonisolated() async throws {
    let ctx = try AgentContext()
    try await ctx.beginSession(systemPrompt: nil)
    try await ctx.append(turn: Turn(role: .user, content: "Hello"))

    // This must compile WITHOUT await — proving nonisolated access
    let stats: ContextStats = ctx.stats  // no await
    XCTAssertEqual(stats.episodicCount, 1)
}
```

**Commit:** `feat(phase6): 6.2 — ContextStats observability`

---

### 6.3 — Session Persistence

Serialize all state to disk so the agent can resume across app launches.

#### ContextCheckpoint

```swift
public struct ContextCheckpoint: Codable, Sendable {
    // Memory store data
    let episodicChunks: [MemoryChunk]
    let episodicIndexData: Data      // ANNSIndex serialized binary
    let semanticChunks: [MemoryChunk]
    let semanticIndexData: Data      // ANNSIndex serialized binary
    let proceduralPatterns: [String: [ToolCall]]

    // Session metadata
    let lastSessionID: UUID?
    let totalSessions: Int

    // Configuration (enough to reconstruct)
    let maxTokens: Int
    let tokenBudgetSafetyMargin: Float
    let episodicMemoryK: Int
    let semanticMemoryK: Int
    let recentTurnsGuaranteed: Int
    let episodicHalfLifeDays: Double
    let semanticHalfLifeDays: Double
    let consolidationThreshold: Int
    let similarityMergeThreshold: Float
    let relevanceWeight: Float
    let centralityWeight: Float
    let efSearch: Int

    // Version for forward compatibility
    let version: Int = 1
}
```

#### `checkpoint(to:)`

```swift
public func checkpoint(to url: URL) throws {
    // 1. Gather all state
    let checkpoint = ContextCheckpoint(
        episodicChunks: await episodicStore.allChunks(),
        episodicIndexData: try await episodicStore.serializeIndex(),
        semanticChunks: await semanticStore.allChunks(),
        semanticIndexData: try await semanticStore.serializeIndex(),
        proceduralPatterns: await proceduralStore.allPatterns(),
        lastSessionID: currentSessionID,
        totalSessions: totalSessions,
        // ... configuration fields ...
    )

    // 2. Encode
    let encoder = JSONEncoder()
    encoder.dateEncodingStrategy = .iso8601
    let data = try encoder.encode(checkpoint)

    // 3. Atomic write: write to temp file, then rename
    let tempURL = url.deletingLastPathComponent()
        .appendingPathComponent(".\(url.lastPathComponent).tmp")
    try data.write(to: tempURL, options: .atomic)

    // Move into place (atomic on APFS/HFS+)
    let fileManager = FileManager.default
    if fileManager.fileExists(atPath: url.path) {
        try fileManager.removeItem(at: url)
    }
    try fileManager.moveItem(at: tempURL, to: url)
}
```

**Note on ANNSIndex serialization**: Check MetalANNS's API for `ANNSIndex.save(to:)` or a `serialize() -> Data` method. If MetalANNS doesn't expose serialization, you'll need to reconstruct the index from chunks on load (re-insert all chunks). This is slower but always works. Document which approach you use.

#### `load(from:)`

```swift
public static func load(from url: URL) async throws -> AgentContext {
    let data = try Data(contentsOf: url)

    let decoder = JSONDecoder()
    decoder.dateDecodingStrategy = .iso8601

    let checkpoint: ContextCheckpoint
    do {
        checkpoint = try decoder.decode(ContextCheckpoint.self, from: data)
    } catch {
        throw ContextCoreError.checkpointCorrupt
    }

    // Reconstruct configuration
    var config = ContextConfiguration.default
    config.maxTokens = checkpoint.maxTokens
    config.tokenBudgetSafetyMargin = checkpoint.tokenBudgetSafetyMargin
    config.episodicMemoryK = checkpoint.episodicMemoryK
    // ... restore all configuration fields ...

    // Create context
    let context = try AgentContext(configuration: config)

    // Restore memory stores
    for chunk in checkpoint.episodicChunks {
        try await context.episodicStore.insertChunk(chunk)
    }
    // If ANNSIndex supports deserialization:
    // try await context.episodicStore.loadIndex(from: checkpoint.episodicIndexData)

    for chunk in checkpoint.semanticChunks {
        try await context.semanticStore.insertChunk(chunk)
    }

    for (taskType, tools) in checkpoint.proceduralPatterns {
        await context.proceduralStore.record(taskType: taskType, tools: tools)
    }

    context.totalSessions = checkpoint.totalSessions

    return context
}
```

**Store extensions needed**: `EpisodicStore` and `SemanticStore` need an `insertChunk(_: MemoryChunk)` method that inserts a pre-built chunk (with existing embedding) rather than creating one from a `Turn`. `ProceduralStore` needs `allPatterns() -> [String: [ToolCall]]`.

#### Tests (`PersistenceTests.swift`)

| # | Test | Setup | Assertion |
|---|---|---|---|
| 1 | Checkpoint creates file | checkpoint(to: tempURL) | File exists at tempURL |
| 2 | Checkpoint is valid JSON | Read checkpoint file | JSONDecoder can parse it |
| 3 | Load restores context | checkpoint → load | No throw |
| 4 | Roundtrip — episodic | Append 50 turns, checkpoint, load, allChunks | Same chunks present (compare by ID) |
| 5 | Roundtrip — semantic | remember 5 facts, checkpoint, load, recall | Same facts returned |
| 6 | Roundtrip — procedural | Record 3 task patterns, checkpoint, load, retrieve | Same patterns returned |
| 7 | Roundtrip — buildWindow | Append 20 turns, buildWindow → window1. checkpoint, load, buildWindow → window2 | window1.chunks.map(\.id) == window2.chunks.map(\.id) |
| 8 | Roundtrip — configuration | Config with non-default values, checkpoint, load | Restored config matches original |
| 9 | Corrupt file | Write garbage data, load | Throws `checkpointCorrupt` |
| 10 | Nonexistent file | load(from: bogusURL) | Throws (file not found error) |
| 11 | Overwrite existing | checkpoint twice to same URL | Second write succeeds, file is valid |
| 12 | Atomic write safety | Write to read-only parent dir | Throws without leaving partial file |
| 13 | totalSessions preserved | 3 sessions, checkpoint, load | stats.totalSessions == 3 |

**Temporary directory for test files:**

```swift
let tempDir = FileManager.default.temporaryDirectory
    .appendingPathComponent("ContextCoreTests-\(UUID().uuidString)")
try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
defer { try? FileManager.default.removeItem(at: tempDir) }
let checkpointURL = tempDir.appendingPathComponent("test.checkpoint")
```

**Commit:** `feat(phase6): 6.3 — Session persistence with atomic checkpoints`

---

### 6.4 — Error Handling & Diagnostics

Ensure every public method throws `ContextCoreError` and add structured logging.

#### Error Paths to Verify

| Error Case | Trigger | Expected Error |
|---|---|---|
| Embedding fails | Inject a provider that throws | `.embeddingFailed(message)` |
| Token budget too small | maxTokens = 10, system prompt = 500 tokens | `.tokenBudgetTooSmall` |
| Session not started | Call append/buildWindow before beginSession | `.sessionNotStarted` |
| Dimension mismatch | Inject provider with dim=128 into stores expecting dim=384 | `.dimensionMismatch(expected: 384, got: 128)` |
| Checkpoint corrupt | Load from file with invalid data | `.checkpointCorrupt` |
| Metal unavailable | Simulator without Metal device | `.metalDeviceUnavailable` |

#### OSLog Diagnostics

Add structured logging for key events. Use `os.Logger` with a subsystem and categories:

```swift
import os

extension Logger {
    static let contextCore = Logger(subsystem: "com.contextcore", category: "AgentContext")
    static let consolidation = Logger(subsystem: "com.contextcore", category: "Consolidation")
    static let scoring = Logger(subsystem: "com.contextcore", category: "Scoring")
}
```

Log at these points:

| Event | Level | Message |
|---|---|---|
| Session begin | `.info` | `"Session started: \(sessionID)"` |
| Session end | `.info` | `"Session ended: \(sessionID). Episodic: \(count)"` |
| buildWindow complete | `.debug` | `"Window built: \(totalTokens) tokens, \(chunkCount) chunks, \(latencyMs)ms"` |
| Consolidation triggered | `.info` | `"Consolidation started: \(episodicCount) chunks"` |
| Consolidation complete | `.info` | `"Consolidation complete: \(factsPromoted) promoted, \(evicted) evicted, \(durationMs)ms"` |
| Contradiction detected | `.notice` | `"Contradiction candidate: '\(factA)' vs '\(factB)'"` |
| Checkpoint saved | `.info` | `"Checkpoint saved to \(url)"` |
| Error | `.error` | `"Error in \(method): \(error)"` |

#### Error Wrapping

Not all internal errors are `ContextCoreError`. Wrap them:

```swift
private func wrapError(_ operation: String, _ block: () async throws -> some Any) async throws {
    do {
        return try await block()
    } catch let error as ContextCoreError {
        Logger.contextCore.error("Error in \(operation): \(error)")
        throw error
    } catch {
        Logger.contextCore.error("Unexpected error in \(operation): \(error)")
        throw ContextCoreError.embeddingFailed(error.localizedDescription)
    }
}
```

Or handle mapping at each call site — avoid over-abstracting if the wrapping pattern doesn't fit cleanly everywhere.

#### Tests (add to `AgentContextTests.swift`)

| # | Test | Setup | Assertion |
|---|---|---|---|
| 21 | embeddingFailed | Inject `FailingEmbeddingProvider`, call append | Throws `ContextCoreError.embeddingFailed` |
| 22 | sessionNotStarted — append | Call append without beginSession | Throws `.sessionNotStarted` |
| 23 | sessionNotStarted — buildWindow | Call buildWindow without beginSession | Throws `.sessionNotStarted` |
| 24 | sessionNotStarted — consolidate | Call consolidate without beginSession | Throws `.sessionNotStarted` |
| 25 | dimensionMismatch | Inject provider with dimension=128, store expects 384 | Throws `.dimensionMismatch(expected: 384, got: 128)` |
| 26 | checkpointCorrupt | Load from file with `"not valid json"` | Throws `.checkpointCorrupt` |
| 27 | Multiple errors don't crash | Trigger 5 different errors in sequence | All throw correctly, context remains usable after recovery |

**Test helpers:**

```swift
/// An EmbeddingProvider that always throws.
final class FailingEmbeddingProvider: EmbeddingProvider, @unchecked Sendable {
    let dimension = 384
    func embed(_ text: String) async throws -> [Float] {
        throw NSError(domain: "test", code: 1, userInfo: [NSLocalizedDescriptionKey: "Embedding model failed"])
    }
    func embedBatch(_ texts: [String]) async throws -> [[Float]] {
        throw NSError(domain: "test", code: 1, userInfo: nil)
    }
}

/// An EmbeddingProvider with wrong dimension.
final class WrongDimensionProvider: EmbeddingProvider, @unchecked Sendable {
    let dimension = 128
    func embed(_ text: String) async throws -> [Float] {
        return [Float](repeating: 0.1, count: 128)
    }
    func embedBatch(_ texts: [String]) async throws -> [[Float]] {
        return texts.map { _ in [Float](repeating: 0.1, count: 128) }
    }
}
```

**Commit:** `feat(phase6): 6.4 — Error handling and OSLog diagnostics`

---

## Design Decisions to Be Aware Of

### Why `async let` for Parallel Scoring?

In `buildWindow`, episodic and semantic scoring are independent — they read from different stores and dispatch to different Metal buffers. `async let` runs them concurrently on the cooperative thread pool. This cuts `buildWindow` latency by up to 40% on multi-core devices compared to sequential scoring.

### Why `nonisolated` Stats?

The agent loop is latency-sensitive. If reading stats required `await`, the consumer would need to suspend their task to check observability metrics. With `nonisolated` access via `OSAllocatedUnfairLock`, stats are readable from any context — UI thread, background task, logging middleware — with zero contention on the actor's executor.

### Why Safety Margin on Token Budget?

The `ApproximateTokenCounter` can over- or under-count by up to 20%. A 10% safety margin ensures the returned window almost certainly fits the model's actual context limit. The effective budget is `maxTokens * 0.90`. Consumers with access to the model's exact tokenizer can set `tokenBudgetSafetyMargin = 0`.

### Why Atomic Checkpoint Writes?

If the app crashes mid-write, a partial checkpoint file would corrupt the restore. The write-to-temp + rename pattern is atomic on APFS — either the full checkpoint exists or the previous one is untouched. Zero-cost insurance against data loss.

### Why `forget()` Is Soft Delete?

Hard-deleting from an ANNS index mid-session is expensive (may require index rebuild). Setting `retentionScore = 0` marks the chunk for eviction during the next consolidation pass. The chunk still exists briefly but scores so low it won't appear in any `buildWindow` result.

---

## Final Verification

After all 4 sub-tasks:

```bash
swift build 2>&1      # zero errors, zero warnings
swift test 2>&1       # all Phase 1–6 tests green
```

**Full pipeline smoke test** — run manually to verify end-to-end:

```swift
let ctx = try AgentContext()
try await ctx.beginSession(systemPrompt: "You are a helpful assistant.")

for i in 0..<20 {
    try await ctx.append(turn: Turn(role: .user, content: "Question \(i)"))
    try await ctx.append(turn: Turn(role: .assistant, content: "Answer \(i)"))
}

let window = try await ctx.buildWindow(currentTask: "Answer the user's questions")
assert(window.totalTokens <= 4096 * 0.9)
assert(window.chunks.count > 0)
print("Window: \(window.totalTokens) tokens, \(window.chunks.count) chunks")

try ctx.checkpoint(to: URL(fileURLWithPath: "/tmp/context.checkpoint"))
let restored = try await AgentContext.load(from: URL(fileURLWithPath: "/tmp/context.checkpoint"))
let window2 = try await restored.buildWindow(currentTask: "Answer the user's questions")
assert(window.chunks.map(\.id) == window2.chunks.map(\.id))

try await ctx.endSession()
print("Stats: \(ctx.stats)")
```

Report: total test count, pass/fail, buildWindow latency, checkpoint file size.

## Phase 6 Is Done When

- `AgentContext` compiles with all engines wired internally
- Consumer-facing API is exactly: init, beginSession, endSession, append, buildWindow, remember, forget, recall, consolidate, checkpoint, stats
- `buildWindow` scores episodic and semantic memory in parallel via `async let`
- Safety margin applied correctly (effective budget = maxTokens * (1 - margin))
- `ContextStats` readable without `await` via `nonisolated var stats`
- Stats update after every relevant operation
- Checkpoint creates valid file, load restores equivalent state
- Checkpoint write is atomic (write-to-temp + rename)
- All error paths throw `ContextCoreError` with correct cases
- OSLog diagnostics present for key lifecycle events
- Full pipeline smoke test passes: append → buildWindow → checkpoint → load → buildWindow
- Phase 1–5 tests still pass (no regressions)
- 4 clean atomic commits in git history
