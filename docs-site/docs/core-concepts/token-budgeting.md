---
sidebar_position: 2
title: "Token Budgeting"
---

# Token Budgeting

Every call to `buildWindow()` must fit within a token budget. ContextCore provides fine-grained control over how that budget is allocated and what happens when space runs out.

## Budget Configuration

Two parameters control the budget:

- **`maxTokens`** (default: 4096) — The raw token budget for the context window.
- **`tokenBudgetSafetyMargin`** (default: 0.10) — A fraction of the budget reserved as headroom.

The effective budget is:

```
effectiveBudget = floor(maxTokens * (1 - tokenBudgetSafetyMargin))
```

With defaults, that gives `floor(4096 * 0.90) = 3686` usable tokens.

```swift
let config = ContextConfig(
    maxTokens: 8192,
    tokenBudgetSafetyMargin: 0.05  // 5% reserved
)
let agent = try await AgentContext(config: config)
```

The safety margin ensures the model always has room for its reply and avoids truncation at the boundary.

## Allocation Order

When building a window, the budget is consumed in a strict priority order:

1. **System prompt** — Allocated first. If you pin a system prompt via `beginSession(id:systemPrompt:)`, it always occupies its full token cost.
2. **Guaranteed recent turns** — The last N turns (controlled by `recentTurnsGuaranteed`, default 3) are included unconditionally.
3. **Scored memory** — Remaining budget is filled with chunks from episodic, semantic, and procedural memory, ranked by their composite score (relevance, recency, importance).

If the system prompt and guaranteed turns alone exceed the budget, scored memory receives no allocation.

## Progressive Compression

When the budget is tight, ContextCore applies progressive compression before dropping content entirely:

1. **Light compression** — Minor summarization that preserves most detail.
2. **Heavy compression** — Aggressive summarization that retains only key facts.
3. **Drop** — The chunk is excluded from the window.

This graduated approach keeps the context as rich as possible under pressure.

## Custom Token Counting

Token counting is abstracted behind the `TokenCounter` protocol:

```swift
public protocol TokenCounter: Sendable {
    func count(_ text: String) -> Int
}
```

The built-in `ApproximateTokenCounter` uses a word-count heuristic (roughly 1 token per 0.75 words). It is fast and requires no external dependencies, making it suitable for most use cases.

For exact counts, provide your own implementation — for example, wrapping a tiktoken-based tokenizer:

```swift
struct TiktokenCounter: TokenCounter {
    let encoding: Encoding

    func count(_ text: String) -> Int {
        encoding.encode(text).count
    }
}

let config = ContextConfig(
    maxTokens: 8192,
    tokenCounter: TiktokenCounter(encoding: .cl100kBase)
)
```

Using an exact counter improves budget utilization, especially at larger context sizes where the heuristic error compounds.
