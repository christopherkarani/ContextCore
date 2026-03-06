---
sidebar_position: 1
title: "Configuring AgentContext"
---

# Configuring AgentContext

`AgentContext` is configured through a `ContextConfiguration` value passed at initialization. Every parameter has a sensible default, so you only need to override what matters for your use case.

## Configuration Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `maxTokens` | `Int` | `4096` | Maximum token budget for built context windows. |
| `tokenBudgetSafetyMargin` | `Double` | `0.10` | Reserve 10% of the budget as a safety margin to avoid overflows. |
| `episodicMemoryK` | `Int` | `8` | Number of episodic memory candidates retrieved per window build. |
| `semanticMemoryK` | `Int` | `4` | Number of semantic memory candidates retrieved per window build. |
| `recentTurnsGuaranteed` | `Int` | `3` | Number of most recent turns always included in the context window. |
| `episodicHalfLifeDays` | `Double` | `7` | Half-life in days for episodic memory decay scoring. |
| `semanticHalfLifeDays` | `Double` | `90` | Half-life in days for semantic memory decay scoring. |
| `consolidationThreshold` | `Int` | `200` | Episodic memory count that triggers auto-consolidation. |
| `similarityMergeThreshold` | `Double` | `0.92` | Cosine similarity threshold above which chunks are merged during deduplication. |
| `relevanceWeight` | `Double` | `0.7` | Weight applied to embedding similarity when scoring candidates. |
| `centralityWeight` | `Double` | `0.4` | Weight applied to attention centrality when scoring candidates. |
| `efSearch` | `Int` | `64` | Breadth parameter for approximate nearest neighbor search. Higher values improve recall at the cost of latency. |
| `embeddingProvider` | `EmbeddingProvider` | `CachingEmbeddingProvider` wrapping `CoreMLEmbeddingProvider` | The embedding backend used for all vector operations. |
| `tokenCounter` | `TokenCounter` | `ApproximateTokenCounter` | Token counting strategy used for budget calculations. |
| `compressionDelegate` | `CompressionDelegate?` | `nil` | Optional delegate for abstractive compression. When nil, only extractive compression is used. |

## Creating a Custom Configuration

```swift
let config = ContextConfiguration(
    maxTokens: 8192,
    tokenBudgetSafetyMargin: 0.05,
    episodicMemoryK: 12,
    semanticMemoryK: 6,
    recentTurnsGuaranteed: 5,
    episodicHalfLifeDays: 14,
    semanticHalfLifeDays: 180,
    consolidationThreshold: 500,
    similarityMergeThreshold: 0.90,
    relevanceWeight: 0.8,
    centralityWeight: 0.3,
    efSearch: 128,
    embeddingProvider: MyCustomProvider(),
    tokenCounter: TiktokenCounter()
)
let context = try AgentContext(configuration: config)
```

## Using the Default Configuration

If the defaults work for your application, initialization is a single line:

```swift
let context = try AgentContext()
```

## Tuning Guidelines

- **Small models (under 4K context):** Lower `maxTokens` and reduce `episodicMemoryK` / `semanticMemoryK` to keep windows compact.
- **Long-running agents:** Increase `consolidationThreshold` if you want to batch consolidation less frequently, or decrease it for more aggressive memory compaction.
- **High-precision retrieval:** Raise `efSearch` (e.g., 128 or 256) for better recall from the HNSW index. This trades latency for accuracy.
- **Deduplication sensitivity:** Lower `similarityMergeThreshold` (e.g., 0.85) to merge more aggressively, or raise it (e.g., 0.95) to keep near-duplicates separate.
