---
sidebar_position: 4
title: "Progressive Compression"
---

# Progressive Compression

When the token budget is tight and all retrieved candidates cannot fit, ContextCore applies progressive compression to maximize the information density of the context window. This is handled automatically during `buildWindow` -- no manual invocation is needed.

## How It Works

The `ProgressiveCompressor` processes candidates in ascending score order, compressing the lowest-value chunks first. This preserves the most relevant content at full fidelity while aggressively compressing less important material.

### Compression Levels

Each candidate is assigned one of three compression levels based on available budget:

| Level | Target | Effect |
|---|---|---|
| `.light` | 50% of original tokens | Keeps the most important half of the content. |
| `.heavy` | 25% of original tokens | Retains only the most critical sentences. |
| `.dropped` | 0% | Chunk is removed entirely from the window. |

The compressor works through candidates from lowest to highest score, escalating compression levels as needed until the remaining content fits within the token budget.

## Extractive Compression

By default, compression is purely extractive. The `CompressionEngine` ranks sentences within a chunk by their embedding similarity to the chunk's overall meaning, then keeps the top sentences that fit within the target token count.

This approach requires no LLM calls and runs entirely on-device using the same embedding infrastructure as the rest of the framework.

## Custom Abstractive Compression

For higher-quality compression, you can provide a `CompressionDelegate` that performs abstractive summarization:

```swift
public protocol CompressionDelegate: Sendable {
    func compress(_ text: String, targetTokens: Int) async throws -> String
}
```

Example implementation using an LLM:

```swift
struct LLMCompressionDelegate: CompressionDelegate {
    func compress(_ text: String, targetTokens: Int) async throws -> String {
        // Call your LLM to summarize the text
        // within the target token count
        return summarizedText
    }
}

let config = ContextConfiguration(
    compressionDelegate: LLMCompressionDelegate()
)
```

When a `CompressionDelegate` is provided, it is used for `.light` and `.heavy` compression levels instead of the default extractive approach. The `.dropped` level still removes the chunk entirely.

## Compression Results

The `ProgressiveCompressionResult` returned internally tracks compression statistics:

- Original token count before compression
- Compressed token count after compression
- Total token savings
- Number of chunks at each compression level

These statistics surface in the `ContextWindow` via the `compressedChunks` property, which reports how many chunks were compressed to fit the budget.
