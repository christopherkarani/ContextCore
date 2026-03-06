---
sidebar_position: 1
title: "Four-Tier Memory"
---

# Four-Tier Memory

ContextCore organizes an agent's knowledge into four distinct memory tiers, each with its own storage semantics, retrieval strategy, and lifecycle. These tiers are represented by the `MemoryType` enum.

## Working Memory

Working memory is the immediate context window sent to the language model. It is **not persisted** — it is rebuilt fresh on every call to `buildWindow()`.

```swift
let window = try await agent.buildWindow()
// window.messages contains the packed context ready for the LLM
```

The window is assembled by scoring and ranking chunks from the other three tiers, then packing them within the token budget. Think of working memory as the final output of the context management pipeline, not a store you write to directly.

## Episodic Memory

Episodic memory holds the turn-by-turn conversational history. Each `Turn` — whether from the user, assistant, tool, or system — is stored with its embedding vector and a timestamp.

```swift
try await agent.appendTurn(.user("What is the capital of France?"))
try await agent.appendTurn(.assistant("The capital of France is Paris."))
```

Retrieval is by **semantic similarity**: when building a context window, episodic chunks are scored against the current query embedding so that the most relevant past exchanges surface first.

Episodic memory is managed by `EpisodicStore` and has a default half-life of **7 days**. Chunks that are not accessed or promoted will decay and eventually be evicted.

## Semantic Memory

Semantic memory stores long-term facts — durable knowledge that should persist well beyond a single conversation. It has a default half-life of **90 days**.

Facts reach semantic memory in two ways:

1. **Promotion from episodic memory** during consolidation (see below).
2. **Direct storage** via the `remember()` API.

```swift
try await agent.remember("The user prefers metric units.")
```

Semantic memory is managed by `SemanticStore`. Because its contents have already been distilled or explicitly provided, semantic chunks tend to be more concise and higher-signal than raw episodic turns.

## Procedural Memory

Procedural memory records tool-usage patterns and task-specific logic. When an agent uses a tool to accomplish a task, ContextCore can record which tools were invoked and in what sequence, building a library of learned procedures.

This allows the agent to recall effective tool strategies for similar tasks in the future. Procedural memory is managed by `ProceduralStore`.

## Promotion Flow: Episodic to Semantic

The primary flow of knowledge through the system is:

```
Episodic  -->  (consolidation)  -->  Semantic
```

Consolidation is triggered when a session ends (via `endSession()`). During consolidation, ContextCore:

1. **Deduplicates** — identifies episodic chunks that express the same fact and merges them into a single entry.
2. **Merges** — combines similar or overlapping facts into more concise representations.
3. **Promotes** — elevates high-value content from episodic memory into semantic memory, where it gains the longer 90-day half-life.

Content that is low-value or redundant with existing semantic memory is left to decay naturally in the episodic store.
