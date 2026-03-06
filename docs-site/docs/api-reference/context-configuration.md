---
sidebar_position: 2
title: "ContextConfiguration"
---

# ContextConfiguration

`ContextConfiguration` controls all tunable parameters for an `AgentContext` instance.

```swift
public struct ContextConfiguration: Sendable
```

## Static Properties

### `default`

```swift
public static var `default`: ContextConfiguration
```

A configuration with sensible defaults for general-purpose use.

## Properties

| Property | Type | Default | Description |
|---|---|---|---|
| `maxTokens` | `Int` | `4096` | Maximum token budget for context windows. |
| `tokenBudgetSafetyMargin` | `Float` | `0.10` | Fraction of budget reserved as safety margin (0.0 to 1.0). |
| `episodicMemoryK` | `Int` | `8` | Number of episodic memory chunks retrieved per window build. |
| `semanticMemoryK` | `Int` | `4` | Number of semantic memory chunks retrieved per window build. |
| `recentTurnsGuaranteed` | `Int` | `3` | Number of most recent turns always included in the window. |
| `episodicHalfLifeDays` | `Double` | `7` | Half-life in days for episodic memory recency decay. |
| `semanticHalfLifeDays` | `Double` | `90` | Half-life in days for semantic memory recency decay. |
| `consolidationThreshold` | `Int` | `200` | Episodic chunk count that triggers automatic consolidation. |
| `similarityMergeThreshold` | `Float` | `0.92` | Cosine similarity threshold above which chunks are merged. |
| `relevanceWeight` | `Float` | `0.7` | Weight of relevance score in the final scoring formula. |
| `centralityWeight` | `Float` | `0.4` | Weight of attention centrality in eviction decisions. |
| `efSearch` | `Int` | `64` | HNSW search beam width. Higher values improve recall at the cost of latency. |
| `embeddingProvider` | `any EmbeddingProvider` | CoreML MiniLM | Provider used to compute text embeddings. |
| `tokenCounter` | `any TokenCounter` | Approximate counter | Counter used to estimate token counts for text. |
| `compressionDelegate` | `(any CompressionDelegate)?` | `nil` | Optional delegate for compressing chunks that exceed budget. |

## Usage

```swift
var config = ContextConfiguration.default
config.maxTokens = 8192
config.episodicMemoryK = 12
config.relevanceWeight = 0.8

let context = try AgentContext(configuration: config)
```

## Notes

- `embeddingProvider` and `tokenCounter` have built-in defaults. Override them only if you need a specific model or precise tokenization.
- Setting `compressionDelegate` to `nil` disables chunk compression. Chunks that do not fit the budget are dropped instead.
- `similarityMergeThreshold` at `0.92` is conservative. Lower it to merge more aggressively during consolidation.
