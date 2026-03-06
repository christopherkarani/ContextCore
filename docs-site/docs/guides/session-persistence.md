---
sidebar_position: 5
title: "Session Persistence"
---

# Session Persistence

ContextCore supports checkpointing and restoring full agent state, allowing sessions to survive app restarts, device reboots, or migration between devices.

## Saving a Checkpoint

Use `checkpoint(to:)` to persist the complete agent state as JSON:

```swift
let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    .appendingPathComponent("context.checkpoint")
try await context.checkpoint(to: url)
```

The write is atomic: data is first written to a temporary file, then moved into place. This prevents corruption if the app is terminated mid-write.

## Restoring from a Checkpoint

Use the static `load(from:)` method to restore a previously saved context:

```swift
let restored = try await AgentContext.load(from: url)
```

This rebuilds all internal engines -- episodic memory, semantic memory, procedural store, HNSW index, and scoring state -- from the checkpoint data.

## What Gets Saved

The `ContextCheckpoint` structure stores:

| Field | Description |
|---|---|
| Episodic chunks | All conversation turn memories with embeddings, timestamps, and retention scores. |
| Semantic chunks | All stored facts with embeddings and metadata. |
| Procedural patterns | Tool schemas, instructions, and other procedural knowledge. |
| Session ID | The unique identifier for the current session. |
| System prompt | The active system prompt, if set. |
| Recent turns | The most recent conversation turns (used for guaranteed inclusion). |
| Total sessions | Counter tracking how many sessions have been checkpointed. |
| Configuration | The full `ContextConfiguration` used to create the context. |
| Version | Schema version number for forward-compatible deserialization. |

## Schema Versioning

Each checkpoint includes a `version` field. Future versions of ContextCore will use this to migrate older checkpoint formats automatically, ensuring that saved sessions remain loadable across framework updates.

## Example: Session Lifecycle

```swift
let checkpointURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    .appendingPathComponent("agent.checkpoint")

// Try to restore a previous session
let context: AgentContext
if FileManager.default.fileExists(atPath: checkpointURL.path) {
    context = try await AgentContext.load(from: checkpointURL)
} else {
    context = try AgentContext()
}

// Use the context normally
try await context.append(turn: Turn(role: .user, content: "Hello"))
let window = try await context.buildWindow(currentTask: "Greet the user")

// Save before exiting
try await context.checkpoint(to: checkpointURL)
```
