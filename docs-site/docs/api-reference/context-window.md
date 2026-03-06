---
sidebar_position: 3
title: "ContextWindow"
---

# ContextWindow

`ContextWindow` is the output of `AgentContext.buildWindow(currentTask:maxTokens:)`. It contains an ordered list of chunks packed within a token budget, ready for injection into a model prompt.

```swift
public struct ContextWindow: Codable, Sendable, Hashable
```

## Properties

| Property | Type | Description |
|---|---|---|
| `chunks` | `[ContextChunk]` | Ordered chunks included in the window. |
| `totalTokens` | `Int` | Total tokens consumed by all chunks. |
| `budgetUsed` | `Float` | Fraction of the token budget used, in the range [0, 1]. |
| `budget` | `Int` | Effective token budget (after safety margin). |
| `retrievedFromMemory` | `Int` | Number of chunks sourced from episodic or semantic memory. |
| `compressedChunks` | `Int` | Number of chunks that were compressed to fit the budget. |

## Methods

### `formatted(style:)`

```swift
public func formatted(style: FormatStyle) -> String
```

Serializes the context window into a string suitable for model injection.

## FormatStyle

```swift
public enum FormatStyle: Sendable {
    case raw
    case chatML
    case alpaca
    case custom(template: String)
}
```

| Style | Description |
|---|---|
| `.raw` | Plain text, chunks separated by newlines. |
| `.chatML` | ChatML tags (`<\|im_start\|>role`, `<\|im_end\|>`). |
| `.alpaca` | Alpaca-style instruction format. |
| `.custom(template:)` | User-defined template with `{role}` and `{content}` placeholders. |

### Example

```swift
let window = try await context.buildWindow(currentTask: "Summarize recent activity")
let prompt = window.formatted(style: .chatML)
```

---

## ContextChunk

A single chunk within a `ContextWindow`.

```swift
public struct ContextChunk: Codable, Sendable, Hashable, Identifiable
```

### Properties

| Property | Type | Description |
|---|---|---|
| `id` | `UUID` | Unique identifier. |
| `content` | `String` | Text content of the chunk. |
| `role` | `TurnRole` | The role that produced this chunk. |
| `tokenCount` | `Int` | Token count of the content. |
| `score` | `Float` | Combined relevance and recency score. |
| `source` | `MemoryType` | Where this chunk originated (episodic, semantic, etc.). |
| `compressionLevel` | `CompressionLevel` | How much compression was applied. |
| `timestamp` | `Date` | When the original content was created. |
| `isGuaranteedRecent` | `Bool` | Whether this chunk was included as a guaranteed recent turn. |
| `isSystemPrompt` | `Bool` | Whether this chunk is the session system prompt. |

---

## CompressionLevel

```swift
public enum CompressionLevel: Codable, Sendable, Hashable {
    case none
    case light
    case heavy
    case dropped
}
```

| Case | Description |
|---|---|
| `.none` | Original content, unmodified. |
| `.light` | Minor compression applied (e.g., removing filler). |
| `.heavy` | Aggressive compression to fit budget constraints. |
| `.dropped` | Content was dropped entirely due to budget limits. |
