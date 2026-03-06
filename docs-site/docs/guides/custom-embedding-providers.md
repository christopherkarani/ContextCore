---
sidebar_position: 6
title: "Custom Embedding Providers"
---

# Custom Embedding Providers

ContextCore uses embeddings for retrieval, similarity scoring, compression, and deduplication. You can replace the default on-device provider with any embedding backend.

## The `EmbeddingProvider` Protocol

```swift
public protocol EmbeddingProvider: Sendable {
    func embed(_ text: String) async throws -> [Float]
}
```

Any type that conforms to `EmbeddingProvider` and returns a fixed-dimension `[Float]` vector can be used.

## Default Provider

The default embedding stack is a `CachingEmbeddingProvider` wrapping a `CoreMLEmbeddingProvider`:

- **`CoreMLEmbeddingProvider`** runs an on-device MiniLM model via Core ML. No network calls, no API keys, low latency.
- **`CachingEmbeddingProvider`** adds an LRU cache (capacity 512) in front of any provider, avoiding redundant embedding computations for repeated text.

## Implementing a Custom Provider

Conform to `EmbeddingProvider` and pass your implementation in the configuration:

```swift
struct OpenAIEmbeddingProvider: EmbeddingProvider {
    func embed(_ text: String) async throws -> [Float] {
        // Call OpenAI embeddings API
        // Return [Float] vector
    }
}

let config = ContextConfiguration(
    embeddingProvider: OpenAIEmbeddingProvider()
)
let context = try AgentContext(configuration: config)
```

## Wrapping with Caching

`CachingEmbeddingProvider` can wrap any provider to add LRU caching:

```swift
let base = OpenAIEmbeddingProvider()
let cached = CachingEmbeddingProvider(wrapping: base, capacity: 1024)

let config = ContextConfiguration(embeddingProvider: cached)
```

This is especially valuable for remote providers where each call has network latency and API cost.

## Important Constraints

**Consistent dimensions.** All embeddings within a single `AgentContext` session must have the same vector dimension. Mixing providers that produce different dimensions (e.g., 384 vs 1536) will cause retrieval failures.

**Checkpoint compatibility.** When restoring from a checkpoint, the embedding provider must produce vectors of the same dimension as those stored in the checkpoint. Switching providers between sessions requires re-embedding all stored chunks.
