---
sidebar_position: 3
title: "Memory Management"
---

# Memory Management

`AgentContext` provides a simple API for storing, retrieving, and managing memories across episodic and semantic stores.

## Adding Conversation Turns

Use `append(turn:)` to add conversation turns to episodic memory:

```swift
try await context.append(turn: Turn(role: .user, content: "Book a flight to Tokyo for next Friday."))
try await context.append(turn: Turn(role: .assistant, content: "I found 3 flights to Tokyo on March 14th."))
```

Each appended turn is automatically:
- Embedded using the configured `EmbeddingProvider`
- Token-counted for budget calculations
- Timestamped for temporal decay scoring
- Indexed in the HNSW vector index for fast retrieval

## Storing Semantic Facts

Use `remember(_:)` to store long-lived facts in semantic memory:

```swift
try await context.remember("User is allergic to peanuts.")
try await context.remember("Preferred language: Japanese for travel-related queries.")
```

Semantic memories have a longer decay half-life (`semanticHalfLifeDays`, default 90 days) compared to episodic memories (default 7 days), making them suitable for persistent knowledge.

## Retrieving Memories

Use `recall(query:k:)` to retrieve the top-k most relevant chunks across both episodic and semantic stores:

```swift
let results = try await context.recall(query: "dietary restrictions", k: 5)
for chunk in results {
    print("\(chunk.source): \(chunk.content)")
}
```

Retrieval is based on cosine similarity between the query embedding and stored chunk embeddings. Results are returned in descending relevance order.

## Forgetting

Use `forget(id:)` to soft-forget a memory by reducing its retention score:

```swift
try await context.forget(id: chunkID)
```

Soft-forgetting does not delete the chunk. Instead, it significantly reduces the chunk's retention score, making it unlikely to be retrieved in future context window builds. The chunk is searched in both episodic and semantic stores.

## Consolidation

Consolidation compacts episodic memory by merging similar chunks and removing low-value entries.

### Auto-Consolidation

When the episodic memory count exceeds `consolidationThreshold` (default 200), consolidation runs automatically during the next `buildWindow` or `append` call. This keeps memory bounded without manual intervention.

### Explicit Consolidation

You can trigger consolidation manually at any time:

```swift
try await context.consolidate()
```

During consolidation:
- Chunks with cosine similarity above `similarityMergeThreshold` (default 0.92) are merged
- Merged chunks combine their content and average their embeddings
- Very low retention chunks may be dropped entirely
- The HNSW index is rebuilt after consolidation
