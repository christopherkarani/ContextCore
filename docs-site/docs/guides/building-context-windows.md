---
sidebar_position: 2
title: "Building Context Windows"
---

# Building Context Windows

The primary output of ContextCore is a `ContextWindow` -- a token-budgeted, relevance-ordered collection of chunks ready to be sent to an LLM. You build one by calling `buildWindow` on an `AgentContext` instance.

## The `buildWindow` Method

```swift
let window = try await context.buildWindow(
    currentTask: "Summarize the user's recent purchase history",
    maxTokens: 4096  // optional override; defaults to config.maxTokens
)
```

**Parameters:**

- `currentTask` -- A natural-language description of the current task. This is embedded and used as the query vector for retrieval.
- `maxTokens` -- An optional token budget override. When omitted, the value from `ContextConfiguration.maxTokens` is used.

## The Retrieval Pipeline

When you call `buildWindow`, the following pipeline executes:

1. **Embed query** -- The `currentTask` string is embedded using the configured `EmbeddingProvider`.
2. **Parallel scoring** -- Episodic and semantic memory stores are searched concurrently. Each candidate is scored using a combination of embedding similarity, temporal decay, and access frequency.
3. **Retrieve procedural** -- Procedural memory patterns (tool schemas, instructions) are retrieved based on task relevance.
4. **Merge** -- Results from all three memory types are merged into a single candidate list, with duplicates removed using the `similarityMergeThreshold`.
5. **Attention rerank** -- Candidates are reranked using attention-based centrality scoring, weighted by `relevanceWeight` and `centralityWeight`.
6. **Pack** -- Candidates are packed into the token budget (minus the safety margin). If the budget is tight, progressive compression is applied to fit more content.
7. **Order** -- Final chunks are ordered for coherent presentation: system prompt first, then procedural, semantic, episodic (by recency), and guaranteed recent turns last.

## The `ContextWindow` Type

The returned `ContextWindow` exposes the following properties:

| Property | Type | Description |
|---|---|---|
| `chunks` | `[ContextChunk]` | The ordered list of chunks included in the window. |
| `totalTokens` | `Int` | Total token count of all included chunks. |
| `budgetUsed` | `Double` | Fraction of the token budget consumed (0.0 to 1.0). |
| `retrievedFromMemory` | `Int` | Number of chunks pulled from episodic or semantic memory. |
| `compressedChunks` | `Int` | Number of chunks that were compressed to fit the budget. |

## Formatting for LLMs

Use `formatted(style:)` to render the window into a string suitable for your target model:

```swift
// ChatML format
let prompt = window.formatted(style: .chatML)

// Alpaca instruction format
let prompt = window.formatted(style: .alpaca)

// Raw concatenation
let prompt = window.formatted(style: .raw)

// Custom template
let prompt = window.formatted(style: .custom(template: "### Context\n{chunks}\n### Task\n{task}"))
```

## Full Example

```swift
let context = try AgentContext(configuration: ContextConfiguration(
    maxTokens: 8192,
    episodicMemoryK: 12,
    semanticMemoryK: 6
))

// Add some history
try await context.append(turn: Turn(role: .user, content: "What did I order last week?"))
try await context.append(turn: Turn(role: .assistant, content: "You ordered a mechanical keyboard and a monitor stand."))

// Store a fact
try await context.remember("User prefers next-day delivery when available.")

// Build a context window for a new task
let window = try await context.buildWindow(
    currentTask: "Help the user track their recent orders"
)

print("Tokens used: \(window.totalTokens) (\(Int(window.budgetUsed * 100))% of budget)")
print("Retrieved from memory: \(window.retrievedFromMemory)")
print("Compressed chunks: \(window.compressedChunks)")

let prompt = window.formatted(style: .chatML)
// Send prompt to your LLM
```
