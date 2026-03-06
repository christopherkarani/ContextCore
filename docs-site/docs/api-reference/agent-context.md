---
sidebar_position: 1
title: "AgentContext"
---

# AgentContext

`AgentContext` is the primary public actor and entry point for ContextCore. It manages sessions, memory, context window construction, and state persistence.

```swift
public actor AgentContext
```

## Initializers

### `init(configuration:)`

```swift
public init(configuration: ContextConfiguration = .default) throws
```

Creates a new `AgentContext` and initializes all internal engines (Metal scoring, embedding, HNSW indices).

**Throws:** `ContextError.metalDeviceUnavailable` if no Metal device is found.

### `load(from:)`

```swift
public static func load(from url: URL) async throws -> AgentContext
```

Restores an `AgentContext` from a previously saved checkpoint.

**Throws:** `ContextError.checkpointCorrupt` if the checkpoint data is invalid.

## Properties

### `configuration`

```swift
public let configuration: ContextConfiguration
```

The immutable runtime configuration provided at initialization.

### `stats`

```swift
public nonisolated var stats: ContextStats
```

A thread-safe snapshot of runtime statistics (chunk counts, memory usage, session info). Safe to read from any isolation domain.

## Methods

### `beginSession(id:systemPrompt:)`

```swift
public func beginSession(id: UUID = UUID(), systemPrompt: String? = nil) async throws
```

Starts a new session. An optional system prompt is injected as a guaranteed chunk in every context window built during this session.

### `endSession()`

```swift
public func endSession() async throws
```

Ends the current session and triggers consolidation (deduplication, promotion to semantic memory, eviction).

**Throws:** `ContextError.sessionNotStarted` if no session is active.

### `append(turn:)`

```swift
public func append(turn: Turn) async throws
```

Appends a conversation turn to episodic memory. Embeddings and token counts are computed automatically if not already set on the turn.

**Throws:** `ContextError.sessionNotStarted` if no session is active.

### `buildWindow(currentTask:maxTokens:)`

```swift
public func buildWindow(currentTask: String, maxTokens: Int? = nil) async throws -> ContextWindow
```

Builds a packed context window optimized for the given task description. The window respects the token budget (from `maxTokens` or `configuration.maxTokens`) and includes guaranteed recent turns, top-k episodic and semantic chunks, and the system prompt.

**Throws:**
- `ContextError.sessionNotStarted` if no session is active.
- `ContextError.tokenBudgetTooSmall` if the budget cannot fit the guaranteed chunks.

### `remember(_:)`

```swift
public func remember(_ fact: String) async throws
```

Stores a semantic fact in long-term memory. The fact is embedded and indexed for future retrieval.

### `forget(id:)`

```swift
public func forget(id: UUID) async throws
```

Soft-forgets a memory chunk by ID. The chunk is marked for eviction and excluded from future context windows.

**Throws:** `ContextError.chunkNotFound` if no chunk exists with the given ID.

### `recall(query:k:)`

```swift
public func recall(query: String, k: Int = 5) async throws -> [MemoryChunk]
```

Retrieves the top-k most relevant memory chunks for the given query string.

### `consolidate()`

```swift
public func consolidate() async throws
```

Explicitly triggers consolidation: promotes high-value episodic chunks to semantic memory, merges similar chunks, and evicts stale entries.

**Throws:** `ContextError.sessionNotStarted` if no session is active.

### `checkpoint(to:)`

```swift
public func checkpoint(to url: URL) async throws
```

Persists the full state (memory indices, session data, configuration) to the given file URL.

**Throws:** `ContextError.checkpointCorrupt` if serialization fails.
