# CLAUDE.md

Guidance for Claude Code (and other AI assistants) when working in this repository.

## Project Overview

**ContextCore** is a Swift package that provides GPU-accelerated context memory for on-device AI agents on Apple Silicon. It sits between an agent's reasoning loop and the model prompt, scoring/ranking/compressing/curating memory in real time before each model call.

- **Language:** Swift 6.2 (strict concurrency)
- **Platforms:** iOS 17+, macOS 14+, visionOS 1+
- **Build system:** Swift Package Manager (`Package.swift`)
- **Acceleration:** Metal compute shaders, CoreML (embeddings), Accelerate
- **External dependency:** [`MetalANNS`](https://github.com/christopherkarani/MetalANNS) (vector index), `swift-docc-plugin`

## Repository Layout

```
Package.swift                   # Swift package manifest (5 targets)
README.md                       # Project overview and quick start
BENCHMARKS.md                   # Latest measured performance numbers
LICENSE                         # MIT

Sources/
├── ContextCoreTypes/           # Pure value types & protocols (no Metal)
│   ├── ContextCoreTypes.swift
│   ├── Errors.swift            # ContextCoreError enum
│   ├── Turn.swift              # Turn, TurnRole, ToolCall
│   ├── Memory/MemoryChunk.swift
│   └── Protocols/              # EmbeddingProvider, TokenCounter,
│                               # CompressionDelegate, ConsolidationStores
│
├── ContextCoreShaders/         # Metal shader library (resource-only)
│   ├── ContextCoreShaders.swift
│   └── Shaders/
│       ├── Relevance.metal
│       ├── Recency.metal
│       ├── Attention.metal
│       ├── Compression.metal
│       └── Consolidation.metal
│
├── ContextCoreEngine/          # GPU-backed engines & CPU reference
│   ├── MetalContext.swift      # Device/queue/library bootstrap
│   ├── ScoringEngine.swift     # GPU relevance + recency scoring
│   ├── AttentionEngine.swift   # GPU attention-centrality reranking
│   ├── CompressionEngine.swift # Sentence ranking + abstractive hook
│   ├── ConsolidationEngine.swift # Episodic → semantic promotion
│   ├── EmbeddingCache.swift
│   ├── CPUReference.swift      # Reference impls for parity tests
│   └── ExtractiveFallbackDelegate.swift
│
├── ContextCore/                # Public library (high-level API)
│   ├── AgentContext.swift      # Main actor API — entry point
│   ├── ContextConfiguration.swift
│   ├── ContextWindow.swift     # ContextChunk, ContextWindow, FormatStyle
│   ├── ContextStats.swift
│   ├── WindowPacker.swift
│   ├── ProgressiveCompressor.swift
│   ├── ChunkOrderer.swift
│   ├── EmbeddingProviders.swift # CoreML + Caching providers
│   ├── CompressionDelegate.swift
│   ├── Turn.swift / Errors.swift # Re-export typealiases
│   ├── Memory/                 # EpisodicStore, SemanticStore, ProceduralStore
│   ├── Persistence/            # SessionStore, ContextCheckpoint
│   ├── Protocols/              # Public protocol re-exports
│   ├── Resources/Embeddings/   # minilm-l6-v2.mlpackage (shipped model)
│   └── ContextCore.docc/       # DocC catalog (Getting Started, Tuning, …)
│
└── ContextCoreBenchmarks/      # Executable: BenchmarkRunner + suites
    ├── BenchmarkRunner.swift
    ├── BuildWindowBenchmark.swift
    ├── ConsolidationBenchmark.swift
    ├── ScoringBenchmark.swift
    ├── RecallQualityBenchmark.swift
    ├── BenchmarkDataFactory.swift
    ├── BenchmarkHarness.swift
    └── BenchmarkMarkdownWriter.swift

Tests/ContextCoreTests/         # swift-testing (`@Test`, `@Suite`) tests
docs-site/                      # Docusaurus site (separate Node project)
```

### Target Dependency Graph

```
ContextCoreTypes  ← pure values/protocols; no dependencies
      ↑
ContextCoreShaders  ← Metal resources only
      ↑
ContextCoreEngine  ← depends on Types + Shaders + MetalANNS
      ↑
ContextCore  ← public API; depends on Engine + Types + MetalANNS
      ↑
ContextCoreBenchmarks  ← executable; depends on ContextCore + Engine
```

Keep this layering strict:
- `ContextCoreTypes` must stay framework-free (no Metal, no CoreML).
- Anything touching Metal goes in `ContextCoreEngine`.
- `ContextCore` re-exports engine symbols via typealiases in `ContextCore.swift` and `Turn.swift` / `Errors.swift` — prefer adding to the existing re-export pattern instead of leaking engine types directly.

## Build, Test, Benchmark

All commands run from the repo root.

```bash
# Build the library
swift build

# Release build
swift build -c release

# Run the test suite (uses swift-testing, not XCTest)
swift test

# Run a single suite or test
swift test --filter AgentContextTests
swift test --filter "AgentContext Integration Tests/buildWindow respects effective budget and includes guaranteed items"

# Build and run the benchmark executable
swift run -c release ContextCoreBenchmarks
# (writes updated tables into BENCHMARKS.md)
```

Metal shaders are compiled by SwiftPM via `.process("Shaders")` in `Sources/ContextCoreShaders`. On platforms where the default metallib is not available, `MetalContext.library(device:)` falls back to compiling the `.metal` files from bundle resources at runtime — preserve both code paths when changing shader packaging.

The shipped CoreML embedding model lives at `Sources/ContextCore/Resources/Embeddings/minilm-l6-v2.mlpackage`. On the iOS Simulator, `CoreMLEmbeddingProvider` intentionally falls back to a deterministic hash-based vector to avoid simulator-specific CoreML issues — tests rely on this determinism.

## Public API Surface

The high-level entry point is the `AgentContext` actor (`Sources/ContextCore/AgentContext.swift`). Its lifecycle:

```swift
let context = try AgentContext()                              // or .init(configuration:)
try await context.beginSession(systemPrompt: "...")           // start a session
try await context.append(turn: Turn(role: .user, content: …)) // record turns
let window = try await context.buildWindow(                   // pack a prompt
    currentTask: "...",
    maxTokens: 4096
)
let prompt = window.formatted(style: .chatML)                 // .raw/.chatML/.alpaca/.custom
try await context.remember("A durable fact")                  // direct semantic insert
try await context.forget(id: chunkID)                         // soft-demote
let hits = try await context.recall(query: "...", k: 5)       // ad-hoc retrieval
try await context.consolidate()                               // manual consolidation
try await context.endSession()                                // ends session + consolidates
try await context.checkpoint(to: url)                         // persist state (atomic)
let restored = try await AgentContext.load(from: url)         // restore
```

Key supporting types (all `Sendable`, most `Codable`):
- `ContextConfiguration` — runtime tuning knobs; see `.default` and the Tuning Guide below.
- `ContextWindow` / `ContextChunk` / `CompressionLevel` / `FormatStyle`
- `Turn` / `TurnRole` / `ToolCall`
- `MemoryChunk` / `MemoryType` (`.episodic`, `.semantic`, `.procedural`)
- `ContextStats` — nonisolated snapshot read via `context.stats`
- `ContextCoreError` — the single public error type
- Protocols: `EmbeddingProvider`, `TokenCounter`, `CompressionDelegate`

### The Four Memory Tiers

1. **Working memory** — the packed `ContextWindow` itself.
2. **Episodic** — per-turn history (`EpisodicStore`), decays with `episodicHalfLifeDays`.
3. **Semantic** — consolidated long-lived facts (`SemanticStore`), decays with `semanticHalfLifeDays`.
4. **Procedural** — tool-usage patterns keyed by task type (`ProceduralStore`).

### `buildWindow` Pipeline

1. Embed task query via `EmbeddingProvider` (cached).
2. Score episodic + semantic candidates in parallel on GPU (`ScoringEngine`): relevance × recency × retention.
3. Gather procedural tool candidates from `ProceduralStore`.
4. Attention-based rerank (`AttentionEngine`) blending centrality (`centralityWeight`) with base score.
5. Pack under budget via `WindowPacker` (guaranteed recent turns + system prompt first).
6. Optional progressive compression (`ProgressiveCompressor` → `CompressionEngine`).
7. Order chunks for model attention via `ChunkOrderer` (`.typeGrouped` by default).

`buildWindow` reserves a `tokenBudgetSafetyMargin` fraction of `maxTokens` as headroom. If the guaranteed content (system prompt + recent turns) exceeds the effective budget, it throws `ContextCoreError.tokenBudgetTooSmall`.

### Configuration Defaults

See `ContextConfiguration.default` and `Sources/ContextCore/ContextCore.docc/TuningGuide.md`. Highlights:

| Parameter | Default | Purpose |
|---|---:|---|
| `maxTokens` | 4096 | Hard token budget |
| `tokenBudgetSafetyMargin` | 0.10 | Headroom fraction |
| `episodicMemoryK` / `semanticMemoryK` | 8 / 4 | Candidates per build |
| `recentTurnsGuaranteed` | 3 | Always-included recent turns |
| `episodicHalfLifeDays` / `semanticHalfLifeDays` | 7 / 90 | Recency decay |
| `consolidationThreshold` | 200 | Auto-consolidate trigger |
| `similarityMergeThreshold` | 0.92 | Duplicate merge bar |
| `relevanceWeight` / `centralityWeight` | 0.7 / 0.4 | Score blending |
| `efSearch` | 64 | ANN search breadth |

## Coding Conventions

### Swift style
- **Swift 6.2**, strict concurrency; annotate types `Sendable` and public APIs `public` explicitly.
- Long-running stateful components are **actors** (`AgentContext`, `ScoringEngine`, `WindowPacker`, stores, `CompressionEngine`, `ConsolidationEngine`, `EmbeddingCache`). Don't add locks to actors — use actor isolation. The one lock in `AgentContext` (`OSAllocatedUnfairLock` around `ContextStats`) exists specifically to let `stats` stay `nonisolated`.
- Use `async let` when scoring/retrieval work can run in parallel, as in `AgentContext.buildWindow` and `recall`.
- Prefer `Logger(subsystem: "com.contextcore", category: …)` from `os.Logger`. Existing categories: `AgentContext`, `Consolidation`, `Scoring`. Log errors with `privacy: .public` for user-visible strings only (not raw content).
- Throw typed `ContextCoreError` cases at API boundaries. Wrap unknown errors with `.embeddingFailed(String)` / `.compressionFailed(String)` — don't re-throw arbitrary errors from public methods.
- Keep `@Sendable` closures small; prefer pure functions (e.g. `mutateStats`).

### Documentation
- **Every public symbol must have DocC-style `///` comments** with a one-line summary, `- Parameter(s)`, `- Returns`, `- Throws`, and a `- Complexity` note for algorithmic methods. Match the style in `AgentContext.swift`.
- User-facing narrative lives in `Sources/ContextCore/ContextCore.docc/*.md`. When you add a public symbol, consider whether it needs a topic entry in `ContextCore.md`.
- Don't add casual comments inside method bodies unless the logic is genuinely non-obvious.

### Files & layout
- One primary type per file; file name matches the type.
- Public type re-exports live in `ContextCore/Turn.swift`, `ContextCore/Errors.swift`, and `ContextCore/ContextCore.swift`. Follow the existing `public typealias` pattern when exposing an engine symbol through the public library.
- Keep Metal shader changes (`*.metal`) paired with pipeline updates in the corresponding engine file and with a CPU reference in `CPUReference.swift` for parity tests.

## Testing

- Tests use **swift-testing** (`import Testing`, `@Suite`, `@Test`, `#expect`), **not** XCTest.
- Each feature has a dedicated test file (`ScoringTests`, `AttentionTests`, `WindowPackerTests`, …). `AgentContextTestSupport.swift` and `TestHelpers.swift` provide shared fixtures; reuse them instead of reinventing setup.
- `TestHelpers.randomVector(dim:seed:)` uses a seeded LCG so embedding tests are deterministic. Use it, or the simulator fallback in `CoreMLEmbeddingProvider`, whenever you need reproducible vectors.
- CPU/GPU parity: new GPU kernels should have a matching CPU reference in `CPUReference.swift` and a parity test asserting `TestHelpers.maxAbsError` is within tolerance.
- Whenever you touch public behavior, run `swift test` before committing.

## Benchmarks

The benchmark executable (`swift run -c release ContextCoreBenchmarks`) updates `BENCHMARKS.md` with measured p50/p95/p99 latencies for `buildWindow`, consolidation, GPU vs CPU scoring, and recall quality. Targets to preserve on M2:
- `buildWindow` p99 < 20 ms for 500 turns
- `consolidate(2000)` p99 < 500 ms
- GPU math-only speedup ≥ 2× CPU at 50k chunks

If a change risks regressing any of these, run the benchmarks and update `BENCHMARKS.md` in the same commit.

## Persistence

- `AgentContext.checkpoint(to:)` writes JSON atomically via a temp file + `moveItem`. Never write directly to the destination URL.
- `ContextCheckpoint.version` must currently be `1`; bump and branch the decoder if the on-disk schema changes, and throw `ContextCoreError.checkpointCorrupt` for unsupported versions.
- `SessionStore` owns in-memory session state; all mutations go through it rather than scattered state on `AgentContext`.

## Git & Workflow

- `.gitignore` already excludes build products (`.build/`, `.swiftpm/`, `DerivedData/`), `Package.resolved`, release notes drafts, `tasks/`, `prompts/`, and Docusaurus build artifacts. Don't commit these.
- The `docs-site/` directory is a separate Docusaurus (Node) project; changes there don't affect the Swift build.
- Commit messages in this repo use conventional-style prefixes: `feat(...)`, `fix(...)`, `perf(...)`, `docs(...)`, `chore(...)`, sometimes with a phase tag like `perf(phase7):`. Match this when committing.
- **This session is working on the `claude/add-claude-documentation-vYL8D` branch.** Develop, commit, and push there.
- Do not open a pull request unless the user explicitly asks for one.
- GitHub access in this session is restricted to `christopherkarani/contextcore` via the MCP tools — never `gh`.

## When Making Changes

1. **Respect the target layering.** Don't pull Metal/CoreML into `ContextCoreTypes`; don't pull engine internals into public API without a typealias re-export.
2. **Don't break `Sendable`.** If you need shared mutable state, use an actor or `OSAllocatedUnfairLock`, matching existing patterns.
3. **Document new public symbols with DocC comments** and add them to `ContextCore.docc/ContextCore.md` if they deserve top-level discoverability.
4. **Add/extend tests** in `Tests/ContextCoreTests/` using swift-testing. Reuse `TestHelpers` and `AgentContextTestSupport`.
5. **For GPU work**, update the Metal shader, the engine actor, the CPU reference, and a parity test together.
6. **For API changes**, update `AgentContext.swift`, the relevant DocC page, and `README.md` quick start if the example changes.
7. **Before committing**, run `swift build` and `swift test`. If performance-sensitive code changed, run `swift run -c release ContextCoreBenchmarks` and refresh `BENCHMARKS.md`.

## Things Not To Do

- Don't add XCTest — this repo is swift-testing only.
- Don't add `print` statements for diagnostics — use `Logger`.
- Don't hand-roll token budgeting logic in new places — go through `WindowPacker`/`ProgressiveCompressor`.
- Don't introduce new error types at public boundaries — extend `ContextCoreError`.
- Don't modify `docs-site/` when asked about Swift code; it's a separate artifact.
- Don't delete or regenerate `Package.resolved` — it's gitignored by design.
