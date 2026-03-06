---
sidebar_position: 2
---

# Quick Start

This guide walks through the core ContextCore flow: initialize, manage a session, build a context window, and interact with long-term memory.

## The 5-Step Flow

```swift
import ContextCore

// 1. Initialize
let context = try AgentContext()

// 2. Start a session
try await context.beginSession(systemPrompt: "You are a senior Swift engineer.")

// 3. Append turns
try await context.append(turn: Turn(role: .user, content: "How do I fix this actor leak?"))

// 4. Build a packed window (Metal-accelerated)
let window = try await context.buildWindow(
    currentTask: "Debug actor isolation",
    maxTokens: 4096
)

// 5. Format for your model
let prompt = window.formatted(style: .chatML)
```

`buildWindow` uses Metal-accelerated scoring to select and rank the most relevant turns for your token budget. The result is a packed context window ready to send to any LLM.

## Long-Term Memory

### Storing Facts

Persist important information across sessions:

```swift
try await context.remember("User prefers async/await over callbacks")
```

### Recalling Memory

Query stored knowledge with semantic search:

```swift
let results = try await context.recall(query: "user preferences", k: 5)
```

## Session Lifecycle

### Ending a Session

When you end a session, ContextCore automatically consolidates the conversation into long-term memory:

```swift
try await context.endSession()
```

### Checkpointing

Save a snapshot of the current context state to disk for later restoration:

```swift
try await context.checkpoint(to: checkpointURL)
```

## Next Steps

With these primitives you can build agents that maintain coherent, token-efficient context across arbitrarily long interactions. Explore the API reference for advanced configuration of scoring strategies, memory backends, and window packing policies.
