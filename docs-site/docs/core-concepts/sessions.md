---
sidebar_position: 3
title: "Sessions"
---

# Sessions

A session represents the lifecycle of a single conversation. It provides the structure within which turns are appended, context windows are built, and knowledge is consolidated.

## Session Lifecycle

The typical flow is:

```swift
// 1. Start a session
try await agent.beginSession(id: "chat-42", systemPrompt: "You are a helpful assistant.")

// 2. Append turns and build windows
try await agent.appendTurn(.user("Summarize the Q3 report."))
let window = try await agent.buildWindow()
// Send window.messages to the LLM, get a response
try await agent.appendTurn(.assistant(response))

// 3. End the session
try await agent.endSession()
```

## Starting a Session

```swift
func beginSession(id: String? = nil, systemPrompt: String? = nil) async throws
```

- **`id`** — An optional identifier. If omitted, one is generated automatically.
- **`systemPrompt`** — An optional system prompt that is pinned to the top of every context window for the duration of the session. It is always allocated budget first.

If you call `beginSession` while a session is already active, ContextCore **automatically ends the previous session** (including running consolidation) before starting the new one. You do not need to call `endSession()` manually in this case.

## Ending a Session

```swift
try await agent.endSession()
```

Ending a session triggers **consolidation**, which:

1. **Promotes** high-value episodic content to semantic memory.
2. **Deduplicates** facts that overlap with existing semantic knowledge.
3. **Evicts** stale chunks that have decayed past their usefulness.

Consolidation is the primary mechanism by which short-term conversational history becomes durable long-term knowledge. See [Four-Tier Memory](./four-tier-memory.md) for details on the promotion flow.

## Guaranteed Recent Turns

The `recentTurnsGuaranteed` configuration parameter (default: 3) controls how many of the most recent turns are **always included** in the context window, regardless of their relevance score.

```swift
let config = ContextConfig(
    recentTurnsGuaranteed: 5  // Always include the last 5 turns
)
```

This ensures conversational coherence. Even if older turns score higher on relevance, the model always sees the immediate context of the conversation.

## Persistence

Sessions and their associated state can be saved to disk and restored later.

### Checkpointing

```swift
let url = URL(fileURLWithPath: "/path/to/checkpoint.ctx")
try await agent.checkpoint(to: url)
```

This writes the full agent state — including all memory stores and session metadata — to the specified file.

### Loading

```swift
let agent = try await AgentContext.load(from: url)
```

This restores the agent to the exact state it was in at checkpoint time, including any active session. You can resume appending turns and building windows immediately.
