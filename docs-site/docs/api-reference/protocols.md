---
sidebar_position: 6
title: "Protocols"
---

# Protocols

ContextCore defines three public protocols for dependency injection. Each has a built-in default implementation, but can be replaced with custom types.

---

## EmbeddingProvider

Converts text into a vector embedding for similarity search.

```swift
public protocol EmbeddingProvider: Sendable {
    func embed(_ text: String) async throws -> [Float]
}
```

### Default Implementation

`CoreMLEmbeddingProvider` uses a bundled CoreML MiniLM model (384 dimensions) with a 512-entry LRU cache to avoid redundant inference.

### Usage

```swift
struct OpenAIEmbeddingProvider: EmbeddingProvider {
    func embed(_ text: String) async throws -> [Float] {
        // Call OpenAI embeddings API
    }
}

var config = ContextConfiguration.default
config.embeddingProvider = OpenAIEmbeddingProvider()
```

### Notes

- All embeddings in a single `AgentContext` must share the same dimensionality.
- The provider must be thread-safe (`Sendable`).

---

## TokenCounter

Estimates or computes the token count for a string.

```swift
public protocol TokenCounter: Sendable {
    func count(_ text: String) -> Int
}
```

### Default Implementation

`ApproximateTokenCounter` uses a word-count heuristic (roughly 0.75 tokens per character boundary). This has approximately 10% variance compared to model-specific tokenizers.

### Usage

```swift
struct TikTokenCounter: TokenCounter {
    func count(_ text: String) -> Int {
        // Use tiktoken for precise GPT-4 token counts
    }
}

var config = ContextConfiguration.default
config.tokenCounter = TikTokenCounter()
```

### Notes

- Token counting is called frequently during window packing. Keep implementations fast.
- The synchronous signature is intentional to avoid async overhead in hot paths.

---

## CompressionDelegate

Compresses text and extracts facts for consolidation.

```swift
public protocol CompressionDelegate: Sendable {
    func compress(text: String, targetTokens: Int) async throws -> String
    func extractFacts(from text: String) async throws -> [String]
}
```

### Methods

| Method | Description |
|---|---|
| `compress(text:targetTokens:)` | Reduces the text to fit within `targetTokens`. Called during window packing when chunks exceed the budget. |
| `extractFacts(from:)` | Extracts discrete facts from a text block. Called during consolidation to promote episodic content to semantic memory. |

### Default Implementation

None. When `ContextConfiguration.compressionDelegate` is `nil`, chunks that exceed the budget are dropped rather than compressed, and fact extraction is skipped during consolidation.

### Usage

```swift
struct LLMCompressor: CompressionDelegate {
    func compress(text: String, targetTokens: Int) async throws -> String {
        // Call an LLM to summarize the text
    }

    func extractFacts(from text: String) async throws -> [String] {
        // Call an LLM to extract key facts
    }
}

var config = ContextConfiguration.default
config.compressionDelegate = LLMCompressor()
```

### Notes

- Both methods are `async` to support LLM-backed implementations.
- `compress` should return text that is at or below `targetTokens` in length as measured by the configured `TokenCounter`.
