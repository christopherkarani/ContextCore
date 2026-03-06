---
sidebar_position: 4
title: "Turn"
---

# Turn

`Turn` represents a single conversation turn appended to an `AgentContext` session.

```swift
public struct Turn: Identifiable, Codable, Sendable, Hashable
```

## Properties

| Property | Type | Description |
|---|---|---|
| `id` | `UUID` | Unique identifier. Defaults to a new UUID. |
| `role` | `TurnRole` | The role that produced this turn. |
| `content` | `String` | Text content of the turn. |
| `timestamp` | `Date` | When the turn was created. Defaults to `Date()`. |
| `tokenCount` | `Int` | Token count. Defaults to `0`; auto-computed on append if zero. |
| `embedding` | `[Float]?` | Pre-computed embedding. Defaults to `nil`; auto-computed on append if nil. |
| `metadata` | `[String: String]` | Arbitrary key-value metadata. Defaults to empty. |

## Initializer

```swift
public init(
    id: UUID = UUID(),
    role: TurnRole,
    content: String,
    timestamp: Date = Date(),
    tokenCount: Int = 0,
    embedding: [Float]? = nil,
    metadata: [String: String] = [:]
)
```

## Equality and Hashing

`Turn` conforms to `Equatable` and `Hashable` based solely on `id`.

## Usage

```swift
let turn = Turn(role: .user, content: "What happened in yesterday's meeting?")
try await context.append(turn: turn)
```

---

## TurnRole

```swift
public enum TurnRole: String, Codable, Sendable, Hashable {
    case user
    case assistant
    case tool
    case system
}
```

| Case | Description |
|---|---|
| `.user` | A message from the user. |
| `.assistant` | A response from the assistant. |
| `.tool` | Output from a tool invocation. |
| `.system` | A system-level instruction. |

---

## ToolCall

`ToolCall` represents metadata about a tool invocation within a turn.

```swift
public struct ToolCall: Codable, Sendable, Hashable
```

### Properties

| Property | Type | Description |
|---|---|---|
| `name` | `String` | Name of the tool that was called. |
| `input` | `String` | Input passed to the tool. |
| `output` | `String` | Output returned by the tool. |
| `durationMs` | `Double` | Execution time in milliseconds. |
