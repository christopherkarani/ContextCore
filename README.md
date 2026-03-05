# ContextCore

GPU-accelerated context management for on-device AI agents.

![Swift 6.2](https://img.shields.io/badge/Swift-6.2-orange.svg) ![iOS 17+](https://img.shields.io/badge/iOS-17%2B-blue.svg) ![macOS 14+](https://img.shields.io/badge/macOS-14%2B-blue.svg) ![visionOS 1+](https://img.shields.io/badge/visionOS-1%2B-blue.svg) ![MIT License](https://img.shields.io/badge/License-MIT-green.svg)

## The Problem

Naive append-only agent loops eventually overflow the model context window, lose early high-value user intent, and produce inconsistent answers. ContextCore solves this before every model call by deciding what to retrieve, what to compress, and how to arrange context under a strict token budget.

## Quick Start

```swift
import ContextCore

let context = try AgentContext()
try await context.beginSession(systemPrompt: "You are a helpful coding assistant.")
try await context.append(turn: Turn(role: .user, content: "Help me debug a Swift actor issue."))

let window = try await context.buildWindow(
    currentTask: "Debug a Swift actor isolation issue",
    maxTokens: 4096
)

let prompt = window.formatted(style: .chatML)
try await context.endSession()
```

## Architecture

```text
┌─────────────────────────────────────┐
│         Your App / Bebop            │
├─────────────────────────────────────┤
│           ContextCore               │
│  AgentContext · WindowPacker        │
│  ConsolidationEngine · Scoring      │
│  Metal kernels (5 shaders)          │
├─────────────────────────────────────┤
│            MetalANNS                │
│  Fixed out-degree graph · NN-Descent│
│  Metal kernels (5 shaders)          │
├─────────────────────────────────────┤
│         Apple Frameworks            │
│  Metal · CoreML · Accelerate · ANE  │
└─────────────────────────────────────┘
```

### Four Memory Types

| Memory | Description |
|---|---|
| Working | The final packed `ContextWindow` sent to the model. |
| Episodic | Turn-level history retrieved by semantic similarity. |
| Semantic | Consolidated high-retention facts promoted from episodic memory. |
| Procedural | Tool usage patterns keyed by task type. |

## Before & After

### Naive Agent (append-only)

```text
[TRUNCATED — first 12 turns lost to front truncation]
Turn 13: User: "What about error handling?"
Turn 14: Assistant: "You should use Result type or throws..."
Turn 15: User: "Can you show me an example?"
Turn 16: Assistant: "Here's a simple example: func fetch()..."
Turn 17: User: "That doesn't compile. I'm getting an error."
Turn 18: Assistant: "Let me fix that. The issue is..."
Turn 19: User: "Actually, go back to the original question about actors."
Turn 20: User: "How do actors prevent data races?"
```

### With ContextCore

```text
[System] You are a Swift programming assistant.

[Semantic — promoted fact] The user is building an SPM library targeting iOS 17+
with strict concurrency checking enabled.

[Semantic — promoted fact] The user prefers code examples over explanations.

[Episodic — retrieved] Turn 3: User asked how Swift actors provide thread safety.
[Episodic — retrieved] Turn 5: Assistant explained actor isolation and Sendable.

[Recent — guaranteed] Turn 18: Assistant: "Let me fix that..."
[Recent — guaranteed] Turn 19: User: "Go back to the original question about actors."
[Recent — guaranteed] Turn 20: User: "How do actors prevent data races?"

Total: 1,847 tokens (within 2,048 budget with margin)
```

## Installation

```swift
dependencies: [
    .package(url: "https://github.com/christopherkarani/ContextCore.git", from: "1.0.0")
]
```

### Dependencies

- [MetalANNS](https://github.com/christopherkarani/MetalANNS) — GPU-accelerated vector index

## Configuration

| Parameter | Default | Effect |
|---|---:|---|
| `maxTokens` | `4096` | Hard context budget |
| `tokenBudgetSafetyMargin` | `0.10` | Headroom for tokenizer drift |
| `episodicMemoryK` | `8` | Episodic candidates retrieved |
| `semanticMemoryK` | `4` | Semantic candidates retrieved |
| `recentTurnsGuaranteed` | `3` | Always-included latest turns |
| `episodicHalfLifeDays` | `7` | Recency decay for episodic memory |
| `semanticHalfLifeDays` | `90` | Recency decay for semantic memory |
| `consolidationThreshold` | `200` | Auto-consolidation trigger |
| `similarityMergeThreshold` | `0.92` | Duplicate merge threshold |
| `relevanceWeight` | `0.7` | Similarity vs recency blend |
| `centralityWeight` | `0.4` | Attention centrality weight |
| `efSearch` | `64` | ANN search depth/quality tradeoff |
| `embeddingProvider` | `CoreMLEmbeddingProvider + cache` | Embedding backend |
| `tokenCounter` | `ApproximateTokenCounter` | Token estimator |
| `compressionDelegate` | `nil` | Optional abstractive compression |

## Memory Footprint

For a 500-turn session (`dim=384`, graph degree `32`):

- Episodic index: ~0.9 MB
- Semantic index: ~0.1 MB
- Scoring buffers: ~0.01 MB per call
- Total GPU memory: ~1 MB

## Known Limitations

- Token counting is approximate (10% safety margin by default).
- Embedding model behavior differs on simulator (deterministic fallback in tests).
- Contradiction detection is heuristic, not semantic contradiction understanding.
- No built-in abstractive LLM compression; inject `CompressionDelegate` for higher-quality summaries.
- GPU scoring throughput is currently launch-overhead bound for small `n` in this benchmark harness.

## Performance

See [BENCHMARKS.md](BENCHMARKS.md) for measured results.

| Operation | 500 turns, M2 | Target |
|---|---|---|
| `buildWindow` p99 | 6.54ms | < 20ms |
| `consolidate` p99 | 19.71ms | < 500ms |
| GPU scoring speedup (`n=2000`) | 0.02x | > 10x |

## License

MIT
