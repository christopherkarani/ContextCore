# ContextCore 1.0.0

GPU-accelerated context management for on-device AI agents.

## What's Included

- **AgentContext** — single-actor public API: append, buildWindow, checkpoint
- **4 memory types** — working, episodic, semantic, procedural
- **5 Metal compute shaders** — relevance, recency, attention, compression, consolidation
- **Automatic consolidation** — background dedup, fact promotion, contradiction detection
- **Progressive compression** — extractive fallback, no LLM required
- **Session persistence** — atomic checkpoints with full state restore

## Performance

Measured on `chriskarani.local` (macOS 26.0, Build 25A354) on March 5, 2026.

| Operation | 500 turns, M2-class Mac |
|---|---|
| buildWindow p99 (4096 budget) | 6.54ms |
| consolidate p99 (2000 chunks) | 19.71ms |
| GPU scoring (`n=2000`) | 0.02x of CPU baseline |
| GPU memory | ~1 MB |

## Requirements

- Swift 6.2+
- iOS 17.0+ / macOS 14.0+ / visionOS 1.0+
- [MetalANNS](https://github.com/christopherkarani/MetalANNS)

## Installation

```swift
dependencies: [
    .package(url: "https://github.com/christopherkarani/ContextCore.git", from: "1.0.0")
]
```
