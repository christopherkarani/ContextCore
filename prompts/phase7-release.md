# Phase 7: Documentation, Benchmarks & Release

You are implementing Phase 7 of ContextCore — the final phase. Phases 1–6 are complete and all 184 tests pass across 17 suites. The framework is functionally complete: `AgentContext` wires all subsystems (scoring, packing, compression, consolidation, persistence) behind a clean public API. Phase 7 makes it production-ready: documentation, performance characterization, and a clean release.

No new functionality. No new Metal kernels. This phase is about proving the system works, documenting it so others can use it, and measuring how fast it runs.

---

## What You Are Delivering

| Deliverable | File(s) | Purpose |
|---|---|---|
| **DocC documentation** | `Sources/ContextCore/ContextCore.docc/` | API reference + 5 tutorial articles |
| **README** | `README.md` | First-contact document: problem, quick start, architecture |
| **Benchmark suite** | `Sources/ContextCoreBenchmarks/` | Executable target measuring latency and throughput |
| **Benchmark results** | `BENCHMARKS.md` | Real device numbers committed to the repo |
| **Release artifacts** | `LICENSE`, git tag, release notes | v1.0.0 ship |

---

## File Map (Phase 7 additions)

```
Sources/
  ContextCore/
    ContextCore.docc/
      ContextCore.md                              # Landing page
      GettingStarted.md                           # 5-minute integration guide
      ArchitectureOverview.md                     # Four memory types, pipeline
      TuningGuide.md                              # Parameter reference with guidance
      IntegratingWithAppleFoundationModels.md     # Custom EmbeddingProvider
      IntegratingWithWax.md                       # Wax memory engine bridge
  ContextCoreBenchmarks/
    main.swift                                    # Benchmark harness entry point
    BuildWindowBenchmark.swift                    # buildWindow latency matrix
    ConsolidationBenchmark.swift                  # consolidate latency scaling
    ScoringBenchmark.swift                        # Metal vs CPU throughput
    RecallQualityBenchmark.swift                  # precision@k measurement
README.md
BENCHMARKS.md
LICENSE
```

---

## Execution Plan

Four sub-tasks. No TDD this phase — these are documentation, measurement, and release tasks.

---

### 7.1 — DocC Documentation

Every public type and function gets a doc comment. Five tutorial articles provide deeper context.

#### Doc Comment Standards

Follow Apple's documentation style:

```swift
/// Build the optimal context window for the current task.
///
/// Scores all memory against the task embedding, packs within the token budget,
/// and orders chunks for optimal model attention. Episodic and semantic scoring
/// run in parallel via `async let`.
///
/// - Parameters:
///   - currentTask: A natural language description of the current task.
///     Used as the query for relevance scoring against stored memory.
///   - maxTokens: Override the configured ``ContextConfiguration/maxTokens``.
///     Pass `nil` to use the configuration default. The effective budget is
///     `maxTokens * (1 - tokenBudgetSafetyMargin)`.
///
/// - Returns: A ``ContextWindow`` containing the assembled context,
///   ready for model injection via ``ContextWindow/formatted(style:)``.
///
/// - Throws: ``ContextCoreError/sessionNotStarted`` if no session is active.
///
/// ## Example
///
/// ```swift
/// let window = try await context.buildWindow(
///     currentTask: "Help debug the user's Swift concurrency issue",
///     maxTokens: 4096
/// )
/// let prompt = window.formatted(style: .chatML)
/// ```
///
/// - Complexity: O(n log n) where n is the total number of stored memory chunks,
///   dominated by ANNS retrieval and Metal scoring dispatch.
public func buildWindow(currentTask: String, maxTokens: Int? = nil) async throws -> ContextWindow
```

Rules:
- **Summary line**: one sentence, what it does, present tense.
- **Discussion**: when helpful, explain behavior, tradeoffs, or non-obvious details.
- **Parameters**: every parameter documented with its purpose and default.
- **Returns**: what the return value represents and how to use it.
- **Throws**: every error case that can occur.
- **Example**: a code snippet showing typical usage.
- **Complexity**: for performance-sensitive methods.
- **Symbol links**: use double backtick syntax (``ContextWindow``) to link to other types.

#### Types to Document

Every `public` symbol across all targets. The complete list:

**ContextCore target:**
- `AgentContext` — actor, 12 public methods, `stats` property
- `ContextConfiguration` — struct, all 14 parameters, `.default` static property
- `ContextWindow` — struct, `formatted(style:)` method
- `ContextChunk` — struct, all properties
- `FormatStyle` — enum, 4 cases
- `CompressionLevel` — enum, 4 cases
- `OrderingStrategy` — enum, 3 cases
- `Turn` — struct, all properties
- `TurnRole` — enum, 4 cases
- `ToolCall` — struct
- `MemoryChunk` — struct, all properties
- `MemoryType` — enum, 3 cases
- `ContextStats` — struct, all 8 fields
- `ContextCoreError` — enum, all 8 cases
- `EmbeddingProvider` — protocol, 3 requirements
- `TokenCounter` — protocol, 1 requirement
- `CompressionDelegate` — protocol, 2 requirements
- `ChunkOrderer` — struct, `order(_:strategy:)` method

**ContextCoreEngine target** (internal, but document for maintainers):
- `ScoringEngine`, `AttentionEngine`, `CompressionEngine`, `ConsolidationEngine`
- `WindowPacker`, `ProgressiveCompressor`
- `CPUReference`

#### DocC Catalog Articles

Create `Sources/ContextCore/ContextCore.docc/` directory.

**`ContextCore.md`** (landing page):

```markdown
# ``ContextCore``

GPU-accelerated context management for on-device AI agents.

## Overview

ContextCore sits between an agent's reasoning loop and the model's context window.
It uses Metal compute shaders to score, rank, compress, and curate context in
real time — deciding what the model should see before every call.

## Topics

### Essentials
- <doc:GettingStarted>
- ``AgentContext``
- ``ContextWindow``

### Memory Types
- <doc:ArchitectureOverview>
- ``MemoryChunk``
- ``MemoryType``

### Configuration
- <doc:TuningGuide>
- ``ContextConfiguration``

### Integration
- <doc:IntegratingWithAppleFoundationModels>
- <doc:IntegratingWithWax>

### Data Model
- ``Turn``
- ``TurnRole``
- ``ToolCall``

### Protocols
- ``EmbeddingProvider``
- ``TokenCounter``
- ``CompressionDelegate``

### Errors
- ``ContextCoreError``
```

**`GettingStarted.md`** — 5-minute integration guide:

Cover these steps with code:
1. Add ContextCore to Package.swift.
2. `import ContextCore`
3. Create `AgentContext`.
4. Begin a session.
5. Append turns in the agent loop.
6. Call `buildWindow` before each model call.
7. Format and pass to the model.
8. End the session.
9. Checkpoint for persistence.

Keep it to 50 lines of Swift total. No explanation longer than 2 sentences per step.

**`ArchitectureOverview.md`** — The four memory types explained:

- Working memory (the context window itself)
- Episodic memory (individual turns, MetalANNS-backed)
- Semantic memory (consolidated facts, higher retention)
- Procedural memory (tool call patterns, dictionary-backed)
- How consolidation promotes episodic → semantic
- How the scoring pipeline decides what enters the window
- Include the full stack diagram from the audit doc

**`TuningGuide.md`** — Parameter reference with guidance:

For each `ContextConfiguration` parameter, document:
- What it controls
- Default value
- When to increase/decrease
- Impact on latency and quality

Include the full parameter reference table:

| Parameter | Default | Effect | Tune when... |
|---|---|---|---|
| `maxTokens` | 4096 | Hard token budget | Your model has larger/smaller context |
| `tokenBudgetSafetyMargin` | 0.10 | Headroom fraction | You have exact tokenizer (set to 0) |
| `episodicMemoryK` | 8 | Chunks from episodic per call | Sessions are short (decrease) or long (increase) |
| `semanticMemoryK` | 4 | Chunks from semantic per call | Few facts (decrease) or knowledge-heavy (increase) |
| `recentTurnsGuaranteed` | 3 | Recent turns always included | Fast back-and-forth (increase) or long turns (decrease) |
| `episodicHalfLifeDays` | 7 | Episodic recency decay | Short-lived tasks (decrease) or long projects (increase) |
| `semanticHalfLifeDays` | 90 | Semantic recency decay | Facts change often (decrease) or are stable (increase) |
| `consolidationThreshold` | 200 | Auto-consolidation trigger | Memory-constrained (decrease) or large sessions (increase) |
| `similarityMergeThreshold` | 0.92 | Duplicate detection bar | False merges (increase) or missed duplicates (decrease) |
| `relevanceWeight` | 0.7 | Similarity vs recency blend | Task-focused (increase) or time-sensitive (decrease) |
| `centralityWeight` | 0.4 | Centrality vs relevance for eviction | Prefer topically connected (increase) or task-relevant (decrease) |

**`IntegratingWithAppleFoundationModels.md`**:

Show how to implement `EmbeddingProvider` using Apple Foundation Models:

```swift
import FoundationModels

struct AppleFoundationEmbeddingProvider: EmbeddingProvider {
    let dimension = 384  // or whatever AFM provides

    func embed(_ text: String) async throws -> [Float] {
        // Use Apple Foundation Models embedding API
        // when publicly available
    }

    func embedBatch(_ texts: [String]) async throws -> [[Float]] {
        try await texts.asyncMap { try await embed($0) }
    }
}

var config = ContextConfiguration.default
config.embeddingProvider = AppleFoundationEmbeddingProvider()
let context = try AgentContext(configuration: config)
```

Also show how to inject Apple's tokenizer for exact token counting.

**`IntegratingWithWax.md`**:

Show how ContextCore can serve as the context engine for Wax sessions:

- Use Wax's long-term memory as a source for ContextCore's semantic store
- Use ContextCore's `buildWindow` to curate what Wax sends to the model
- Bridge Wax's `remember`/`recall` to ContextCore's `remember`/`recall`

#### Verification

```bash
swift package generate-documentation --target ContextCore 2>&1
# Must complete with zero warnings
```

**Commit:** `docs(phase7): 7.1 — DocC documentation for all public API`

---

### 7.2 — README

The first thing anyone sees. Must answer "what is this, should I use it, and how do I start" in under 2 minutes of reading.

#### Structure

```markdown
# ContextCore

GPU-accelerated context management for on-device AI agents.

[badges: Swift 6.2 | iOS 17+ | macOS 14+ | visionOS 1.0+ | MIT License]

## The Problem

[1 paragraph: naive agents overflow context, lose early information, produce
incoherent responses. ContextCore answers three questions on GPU before every
model call: what to retrieve, what to compress, and how to arrange.]

## Quick Start

[10 lines of Swift: init → beginSession → append → buildWindow → formatted]

## Architecture

[ASCII diagram of the full stack: Your App → ContextCore → MetalANNS → Apple Frameworks]

### Four Memory Types

[Table: Working, Episodic, Semantic, Procedural — one sentence each]

## Before & After

### Naive Agent (append-only)
[Show what the model sees after 20 turns: all 20 turns crammed in,
overflowing, front-truncated, losing the user's original goal]

### With ContextCore
[Show what the model sees: system prompt, 2 semantic facts, 3 relevant
episodic memories, last 3 turns, current message — all within budget,
ordered for attention]

## Installation

[SPM: .package(url: "...", from: "1.0.0")]

### Dependencies

- [MetalANNS](https://github.com/chriskarani/MetalANNS) — GPU-accelerated vector index

## Configuration

[Parameter reference table — all 11 parameters with default and effect]

## Memory Footprint

[Formula and example: 500-turn session ≈ 1 MB GPU memory]

## Known Limitations

- Token counting is approximate (10% safety margin by default)
- Embedding model not available on simulator (mock provider used in tests)
- Contradiction detection is heuristic, not semantic understanding
- No built-in LLM compression — inject your own CompressionDelegate for abstractive quality

## Performance

See [BENCHMARKS.md](BENCHMARKS.md) for detailed measurements.

| Operation | 500 turns, M2 | Target |
|---|---|---|
| `buildWindow` p99 | Xms | < 20ms |
| `consolidate` p99 | Xms | < 500ms |
| GPU scoring speedup | Xx | > 10x vs CPU |

## License

MIT
```

#### Before/After Example

This is the most important section for convincing someone to use ContextCore. Make it concrete.

**Naive agent — turn 20 of 20, maxTokens=2048:**

```
[TRUNCATED — first 12 turns lost to front truncation]
Turn 13: User: "What about error handling?"
Turn 14: Assistant: "You should use Result type or throws..."
Turn 15: User: "Can you show me an example?"
Turn 16: Assistant: "Here's a simple example: func fetch()..."
Turn 17: User: "That doesn't compile. I'm getting an error."
Turn 18: Assistant: "Let me fix that. The issue is..."
Turn 19: User: "Actually, go back to the original question about actors."
Turn 20: User: "How do actors prevent data races?"

// Model has NO IDEA what the "original question about actors" was.
// It was in Turn 3, which was truncated.
```

**With ContextCore — same 20 turns, same budget:**

```
[System] You are a Swift programming assistant.

[Semantic — promoted fact] The user is building a Swift Package Manager library
targeting iOS 17+ with strict concurrency checking enabled.

[Semantic — promoted fact] The user prefers code examples over explanations.

[Episodic — retrieved, relevant to "actors"] Turn 3: User asked "How do Swift
actors provide thread safety?" — this is the original question being referenced.

[Episodic — retrieved, relevant to "actors"] Turn 5: Assistant explained actor
isolation and Sendable conformance.

[Recent — guaranteed] Turn 18: Assistant: "Let me fix that..."
[Recent — guaranteed] Turn 19: User: "Go back to the original question about actors."
[Recent — guaranteed] Turn 20: User: "How do actors prevent data races?"

// Total: 1,847 tokens (within 2048 budget with 10% margin)
// Model has full context: the original actor question, user preferences,
// and the most recent conversation flow.
```

**Commit:** `docs(phase7): 7.2 — README with quick start, architecture, and before/after`

---

### 7.3 — Benchmark Suite

An executable target that measures the performance characteristics consumers care about.

#### Package.swift Addition

```swift
.executableTarget(
    name: "ContextCoreBenchmarks",
    dependencies: ["ContextCore", "ContextCoreEngine"],
    path: "Sources/ContextCoreBenchmarks"
)
```

#### Benchmark Harness

Write a simple harness — no external benchmark library needed:

```swift
// Sources/ContextCoreBenchmarks/BenchmarkHarness.swift

struct BenchmarkResult {
    let name: String
    let iterations: Int
    let p50Ms: Double
    let p95Ms: Double
    let p99Ms: Double
    let minMs: Double
    let maxMs: Double
}

func benchmark(
    name: String,
    warmup: Int = 5,
    iterations: Int = 50,
    block: () async throws -> Void
) async throws -> BenchmarkResult {
    // Warmup
    for _ in 0..<warmup {
        try await block()
    }

    // Measure
    var durations: [Double] = []
    let clock = ContinuousClock()

    for _ in 0..<iterations {
        let elapsed = try await clock.measure {
            try await block()
        }
        durations.append(Double(elapsed.components.attoseconds) / 1_000_000_000_000_000.0)
    }

    durations.sort()

    return BenchmarkResult(
        name: name,
        iterations: iterations,
        p50Ms: durations[iterations / 2],
        p95Ms: durations[Int(Double(iterations) * 0.95)],
        p99Ms: durations[Int(Double(iterations) * 0.99)],
        minMs: durations.first!,
        maxMs: durations.last!
    )
}
```

#### Benchmark Scenarios

**`BuildWindowBenchmark.swift`** — latency matrix:

```swift
func runBuildWindowBenchmarks() async throws -> [BenchmarkResult] {
    var results: [BenchmarkResult] = []

    for turnCount in [10, 50, 200, 500] {
        for budget in [2048, 4096, 8192] {
            let ctx = try AgentContext()
            try await ctx.beginSession(systemPrompt: "You are a helpful assistant.")

            // Populate with realistic turns
            for i in 0..<turnCount {
                try await ctx.append(turn: Turn(
                    role: i % 2 == 0 ? .user : .assistant,
                    content: generateRealisticContent(index: i)
                ))
            }

            let result = try await benchmark(
                name: "buildWindow(\(turnCount) turns, \(budget) budget)",
                iterations: 50
            ) {
                _ = try await ctx.buildWindow(
                    currentTask: "Help the user with their current task",
                    maxTokens: budget
                )
            }

            results.append(result)
        }
    }

    return results
}
```

Generate realistic content — not lorem ipsum. Use a mix of code discussion, tool calls, and conversational turns that an actual agent session would produce.

**`ConsolidationBenchmark.swift`** — scaling:

```swift
func runConsolidationBenchmarks() async throws -> [BenchmarkResult] {
    var results: [BenchmarkResult] = []

    for chunkCount in [100, 500, 2000] {
        let ctx = try AgentContext()
        try await ctx.beginSession(systemPrompt: nil)

        // Insert chunks (include ~10% near-duplicates for realistic consolidation)
        for i in 0..<chunkCount {
            let content = i % 10 == 0
                ? generateRealisticContent(index: i / 10)  // duplicate
                : generateRealisticContent(index: i)        // unique
            try await ctx.append(turn: Turn(role: .user, content: content))
        }

        let result = try await benchmark(
            name: "consolidate(\(chunkCount) chunks)",
            iterations: 10  // fewer iterations — consolidation is heavier
        ) {
            try await ctx.consolidate()
        }

        results.append(result)
    }

    return results
}
```

**`ScoringBenchmark.swift`** — Metal vs CPU:

```swift
func runScoringBenchmarks() async throws -> [BenchmarkResult] {
    var results: [BenchmarkResult] = []

    for n in [100, 500, 2000] {
        let chunks = TestHelpers.randomVectors(n: n, dim: 384, seed: 42)
        let query = TestHelpers.randomVector(dim: 384, seed: 99)
        let recencyWeights = (0..<n).map { Float($0) / Float(n) }

        // GPU path
        let scoringEngine = try ScoringEngine()
        let gpuResult = try await benchmark(
            name: "GPU relevance scoring (n=\(n))",
            iterations: 100
        ) {
            _ = try await scoringEngine.scoreChunks(
                query: query,
                chunks: /* wrap in MemoryChunks */,
                recencyWeights: recencyWeights
            )
        }
        results.append(gpuResult)

        // CPU path
        let cpuResult = try await benchmark(
            name: "CPU relevance scoring (n=\(n))",
            iterations: 100
        ) {
            _ = CPUReference.relevanceScores(
                query: query,
                chunks: chunks,
                recencyWeights: recencyWeights
            )
        }
        results.append(cpuResult)
    }

    return results
}
```

**`RecallQualityBenchmark.swift`** — precision@k:

This measures whether `buildWindow` retrieves the *right* chunks, not just fast enough.

```swift
func runRecallQualityBenchmark() async throws {
    // 50-turn session with known relevant chunks for a test query
    let ctx = try AgentContext()
    try await ctx.beginSession(systemPrompt: nil)

    // Ground truth: for query "Swift concurrency", these turns are relevant
    let relevantIndices: Set<Int> = [3, 7, 12, 18, 22, 31, 45]

    var turnIDs: [Int: UUID] = [:]
    for i in 0..<50 {
        let content = relevantIndices.contains(i)
            ? generateConcurrencyContent(index: i)   // relevant to query
            : generateIrrelevantContent(index: i)     // cooking, weather, etc.
        let turn = Turn(role: .user, content: content)
        try await ctx.append(turn: turn)
        turnIDs[i] = turn.id
    }

    let window = try await ctx.buildWindow(
        currentTask: "Explain Swift concurrency patterns",
        maxTokens: 4096
    )

    // Measure precision@k
    let relevantIDs = Set(relevantIndices.compactMap { turnIDs[$0] })

    for k in [3, 5, 8] {
        let retrievedIDs = Set(window.chunks
            .filter { !$0.isGuaranteedRecent && $0.role != .system }
            .prefix(k)
            .map(\.id))

        let truePositives = retrievedIDs.intersection(relevantIDs).count
        let precision = Float(truePositives) / Float(k)

        print("precision@\(k) = \(precision) (\(truePositives)/\(k) relevant)")
    }
}
```

#### Main Entry Point

```swift
// Sources/ContextCoreBenchmarks/main.swift

import Foundation
import ContextCore
import ContextCoreEngine

@main
struct BenchmarkRunner {
    static func main() async throws {
        print("ContextCore Benchmark Suite")
        print("==========================\n")
        print("Device: \(ProcessInfo.processInfo.hostName)")
        print("Date: \(ISO8601DateFormatter().string(from: Date()))\n")

        var allResults: [BenchmarkResult] = []

        print("--- buildWindow Latency ---")
        let buildResults = try await runBuildWindowBenchmarks()
        allResults.append(contentsOf: buildResults)
        for r in buildResults { printResult(r) }

        print("\n--- Consolidation Latency ---")
        let consolResults = try await runConsolidationBenchmarks()
        allResults.append(contentsOf: consolResults)
        for r in consolResults { printResult(r) }

        print("\n--- Metal vs CPU Scoring ---")
        let scoringResults = try await runScoringBenchmarks()
        allResults.append(contentsOf: scoringResults)
        for r in scoringResults { printResult(r) }

        print("\n--- Recall Quality ---")
        try await runRecallQualityBenchmark()

        // Generate BENCHMARKS.md
        try generateBenchmarksMarkdown(results: allResults)
    }

    static func printResult(_ r: BenchmarkResult) {
        print("  \(r.name): p50=\(String(format: "%.2f", r.p50Ms))ms  p95=\(String(format: "%.2f", r.p95Ms))ms  p99=\(String(format: "%.2f", r.p99Ms))ms")
    }
}
```

#### BENCHMARKS.md Format

```markdown
# ContextCore Benchmarks

Measured on [device], [date].

## buildWindow Latency

| Turns | Budget | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|---|
| 10 | 2048 | | | |
| 10 | 4096 | | | |
| 10 | 8192 | | | |
| 50 | 2048 | | | |
...
| 500 | 8192 | | | |

**Target**: p99 < 20ms for 500 turns on M2.

## Consolidation Latency

| Chunks | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|
| 100 | | | |
| 500 | | | |
| 2000 | | | |

**Target**: p99 < 500ms for 2000 chunks on M2.

## Metal vs CPU Scoring

| n | GPU p50 (ms) | CPU p50 (ms) | Speedup |
|---|---|---|---|
| 100 | | | x |
| 500 | | | x |
| 2000 | | | x |

## Recall Quality

| k | Precision@k |
|---|---|
| 3 | |
| 5 | |
| 8 | |

## Memory Footprint

500-turn session, dim=384, degree=32:
- Episodic index: ~0.9 MB
- Semantic index: ~0.1 MB
- Scoring buffers: ~0.01 MB per call
- **Total GPU memory: ~1 MB**
```

**Commit:** `perf(phase7): 7.3 — Benchmark suite with latency and throughput measurements`

---

### 7.4 — Release Checklist

A systematic verification pass before tagging v1.0.0.

#### Checklist

Run each item and record pass/fail:

```
RELEASE CHECKLIST — ContextCore v1.0.0
=======================================

Build & Test
  [ ] swift build — zero errors, zero warnings (macOS)
  [ ] swift test — all tests pass (macOS, real Metal device)
  [ ] swift test — all tests pass (iOS simulator, CPU fallback)
  [ ] swift build -c release — optimized build compiles clean

Performance Targets
  [ ] buildWindow p99 < 20ms — 500 turns, 4096 budget, M-series Mac
  [ ] consolidate p99 < 500ms — 2000 chunks, M-series Mac
  [ ] First buildWindow after init < 50ms (pipeline warmup effective)

Memory Safety
  [ ] swift test --sanitize=thread — zero data races
  [ ] swift test --sanitize=address — zero memory errors
  [ ] No retain cycles (check with Instruments Leaks or manual audit of actor references)

Documentation
  [ ] swift package generate-documentation — zero warnings
  [ ] All public symbols have doc comments
  [ ] 5 DocC articles present and linked from landing page
  [ ] README.md complete with quick start, architecture, before/after

Dependency
  [ ] Compiles with MetalANNS from GitHub URL (.package(url:...))
  [ ] Compiles with MetalANNS from local path (.package(path:...))
  [ ] MetalANNS version pinned (branch or exact version, not floating)

Compatibility
  [ ] Builds for iOS 17.0 target
  [ ] Builds for macOS 14.0 target
  [ ] Builds for visionOS 1.0 target
  [ ] Simulator fallback compiles and all tests pass without Metal

Artifacts
  [ ] LICENSE file present (MIT)
  [ ] .gitignore up to date
  [ ] No secrets, credentials, or API keys in repo
  [ ] No large binary files committed (CoreML model is in Resources, verify size reasonable)
  [ ] BENCHMARKS.md committed with real device numbers
```

#### Release Steps

After all checklist items pass:

1. **Final commit**: any remaining fixes from the checklist.

2. **Tag**:
```bash
git tag -a v1.0.0 -m "ContextCore 1.0.0 — GPU-accelerated context management for on-device AI agents"
git push origin v1.0.0
```

3. **GitHub release notes** — write using this template:

```markdown
# ContextCore 1.0.0

GPU-accelerated context management for on-device AI agents.

## What's Included

- **AgentContext** — single-actor public API: append, buildWindow, checkpoint
- **4 memory types** — working, episodic, semantic, procedural
- **5 Metal compute shaders** — relevance, recency, attention, compression, consolidation
- **Automatic consolidation** — background dedup, fact promotion, contradiction detection
- **Progressive compression** — extractive fallback, no LLM required
- **Session persistence** — atomic checkpoints with full state restore

## Performance

| Operation | 500 turns, M2 |
|---|---|
| buildWindow p99 | Xms |
| consolidate p99 | Xms |
| GPU scoring | Xx faster than CPU |
| GPU memory | ~1 MB |

## Requirements

- Swift 6.2+
- iOS 17.0+ / macOS 14.0+ / visionOS 1.0+
- [MetalANNS](https://github.com/chriskarani/MetalANNS)

## Installation

```swift
dependencies: [
    .package(url: "https://github.com/chriskarani/ContextCore.git", from: "1.0.0")
]
```
```

4. **Swift Package Index**: Submit URL at [swiftpackageindex.com/add-a-package](https://swiftpackageindex.com/add-a-package).

**Commit:** `chore(phase7): 7.4 — Release preparation`

---

## Final Verification

After all 4 sub-tasks:

```bash
swift build 2>&1                    # zero errors, zero warnings
swift test 2>&1                     # all 184+ tests green
swift build -c release 2>&1         # optimized build clean
swift test --sanitize=thread 2>&1   # zero data races
swift package generate-documentation --target ContextCore 2>&1  # zero warnings
```

## Phase 7 Is Done When

- Every public symbol has a doc comment with summary, parameters, returns, throws, and example
- 5 DocC articles build and link correctly from the landing page
- README contains problem statement, quick start, architecture, before/after, parameter table, and badges
- Benchmark suite runs as an executable target and produces BENCHMARKS.md
- `buildWindow` p99 < 20ms for 500 turns confirmed on M-series Mac
- `consolidate` p99 < 500ms for 2000 chunks confirmed on M-series Mac
- Thread Sanitizer and Address Sanitizer pass with zero issues
- LICENSE file present
- Git tag `v1.0.0` created
- GitHub release notes written
- Phase 1–6 tests still pass (no regressions)
- 4 clean atomic commits in git history

---

## The Full Stack — Delivered

```
┌─────────────────────────────────────┐
│         Your App / Bebop            │  ← domain logic, UI, business rules
├─────────────────────────────────────┤
│           ContextCore               │  ← this framework
│  AgentContext · WindowPacker        │
│  ConsolidationEngine · Scoring      │
│  Metal kernels (5 shaders)          │
├─────────────────────────────────────┤
│            MetalANNS                │  ← vector index dependency
│  Fixed out-degree graph · NN-Descent│
│  Metal kernels (5 shaders)          │
├─────────────────────────────────────┤
│         Apple Frameworks            │
│  Metal · CoreML · Accelerate · ANE  │
└─────────────────────────────────────┘
```

ContextCore — Intelligent context for agents that think on-device.
