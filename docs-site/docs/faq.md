---
sidebar_position: 100
title: "FAQ"
---

# FAQ

## Does ContextCore require a GPU?

Yes. ContextCore uses Metal compute shaders for scoring, so a Metal-capable device is required. This includes all Apple Silicon Macs (M1 and later) and A-series iOS devices. The iOS Simulator has limited Metal support and may not work reliably.

## What embedding model is used by default?

The default `EmbeddingProvider` is `CoreMLEmbeddingProvider`, which runs a CoreML-optimized MiniLM model producing 384-dimensional embeddings. Results are cached in a 512-entry LRU cache to avoid redundant inference.

## Can I use my own embedding model?

Yes. Implement the `EmbeddingProvider` protocol and pass it via `ContextConfiguration.embeddingProvider`:

```swift
struct MyEmbedder: EmbeddingProvider {
    func embed(_ text: String) async throws -> [Float] {
        // Your implementation
    }
}

var config = ContextConfiguration.default
config.embeddingProvider = MyEmbedder()
```

## How accurate is the default token counter?

`ApproximateTokenCounter` uses a word-count heuristic with approximately 10% variance compared to model-specific tokenizers. For precise budgeting, implement the `TokenCounter` protocol with a tokenizer matching your target model (e.g., tiktoken for GPT-4).

## How does consolidation work?

Consolidation triggers automatically when the episodic chunk count exceeds `consolidationThreshold` (default: 200), or when a session ends via `endSession()`. The consolidation engine:

1. Promotes high-value episodic facts to semantic memory.
2. Merges chunks with cosine similarity above `similarityMergeThreshold` (default: 0.92).
3. Evicts stale chunks based on recency decay and attention centrality scores.

You can also trigger it manually with `consolidate()`.

## Can I use ContextCore without sessions?

No. Sessions are required. You must call `beginSession()` before appending turns or building context windows. Calling `append(turn:)` or `buildWindow(currentTask:)` without an active session throws `ContextError.sessionNotStarted`.

## What format styles are supported?

The `ContextWindow.formatted(style:)` method supports four styles:

- `.raw` -- plain text with newline-separated chunks.
- `.chatML` -- ChatML tags for models that expect them.
- `.alpaca` -- Alpaca instruction format.
- `.custom(template:)` -- user-defined template with `{role}` and `{content}` placeholders.

## How do I persist and restore state?

Use `checkpoint(to:)` to save the full state and `AgentContext.load(from:)` to restore it:

```swift
// Save
try await context.checkpoint(to: fileURL)

// Restore
let restored = try await AgentContext.load(from: fileURL)
```

## What platforms are supported?

- iOS 17+
- macOS 14+
- visionOS 1+
- Swift 6.2+
