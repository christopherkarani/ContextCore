---
sidebar_position: 7
title: "Custom Token Counters"
---

# Custom Token Counters

Token counting drives budget allocation, packing decisions, and compression targeting in ContextCore. You can replace the default heuristic counter with a precise tokenizer for your target model.

## The `TokenCounter` Protocol

```swift
public protocol TokenCounter: Sendable {
    func count(_ text: String) -> Int
}
```

The protocol is synchronous by design -- token counting is called frequently and must be fast.

## Default Counter

`ApproximateTokenCounter` uses a word-count heuristic to estimate token counts. It is fast and requires no external dependencies, with roughly 10% variance from actual BPE token counts.

This is sufficient for most use cases where exact budget adherence is not critical.

## Implementing a Precise Counter

For exact token budgeting, implement `TokenCounter` with a real tokenizer:

```swift
struct TiktokenCounter: TokenCounter {
    func count(_ text: String) -> Int {
        // Use tiktoken encoding (e.g., cl100k_base for GPT-4)
        return tokenize(text).count
    }
}

let config = ContextConfiguration(
    tokenCounter: TiktokenCounter()
)
let context = try AgentContext(configuration: config)
```

## Where Token Counting Is Used

The token counter is invoked in several parts of the pipeline:

- **Budget calculation** -- Determines how much of the `maxTokens` budget remains after the safety margin and guaranteed recent turns.
- **Packing decisions** -- During context window assembly, each candidate's token count determines whether it fits in the remaining budget.
- **Compression targeting** -- Progressive compression uses token counts to calculate the target size for `.light` (50%) and `.heavy` (25%) compression levels.

Because token counting is called on every chunk during every `buildWindow` call, keep your implementation fast. If using a tokenizer with startup cost, initialize it once and reuse across calls.
