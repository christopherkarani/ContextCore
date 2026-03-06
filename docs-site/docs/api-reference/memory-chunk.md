---
sidebar_position: 5
title: "MemoryChunk"
---

# MemoryChunk

`MemoryChunk` represents a stored piece of memory in ContextCore's episodic or semantic memory indices. Returned by `AgentContext.recall(query:k:)`.

```swift
public struct MemoryChunk: Identifiable, Codable, Sendable, Hashable
```

## Properties

| Property | Type | Description |
|---|---|---|
| `id` | `UUID` | Unique identifier. |
| `content` | `String` | Text content of the memory. |
| `embedding` | `[Float]` | Vector embedding of the content. |
| `type` | `MemoryType` | Classification of the memory. |
| `createdAt` | `Date` | When the memory was first stored. |
| `lastAccessedAt` | `Date` | When the memory was last retrieved or scored. |
| `accessCount` | `Int` | Total number of times this memory has been accessed. |
| `retentionScore` | `Float` | Current retention score used for eviction decisions. |
| `sourceSessionID` | `UUID?` | The session that produced this memory, if any. |
| `metadata` | `[String: String]` | Arbitrary key-value metadata. |

## Usage

```swift
let results = try await context.recall(query: "deployment process", k: 3)
for chunk in results {
    print("\(chunk.type): \(chunk.content) (score: \(chunk.retentionScore))")
}
```

---

## MemoryType

```swift
public enum MemoryType: String, Codable, Sendable, Hashable {
    case episodic
    case semantic
    case procedural
    case working
}
```

| Case | Description |
|---|---|
| `.episodic` | Derived from conversation turns. Subject to recency decay. |
| `.semantic` | Promoted facts and knowledge. Long half-life. |
| `.procedural` | Learned procedures and patterns. |
| `.working` | Temporary working memory for the current session. |
