# ContextCore 🧠

<div align="center">
  <img src="https://raw.githubusercontent.com/christopherkarani/ContextCore/main/Assets/banner.png" alt="ContextCore Banner" width="100%">
  <br>
  <h1><b>Aura ⚡️ ContextCore</b></h1>
  <h3><b>High-precision, low-latency memory scout for Apple Silicon.</b></h3>
  
  <p>
    <a href="https://swift.org"><img src="https://img.shields.io/badge/Swift-6.2-000000.svg?style=for-the-badge&logo=swift&logoColor=white" alt="Swift 6.2"></a>
    <a href="https://developer.apple.com/ios/"><img src="https://img.shields.io/badge/iOS-17%2B-000000.svg?style=for-the-badge&logo=apple&logoColor=white" alt="iOS 17+"></a>
    <a href="https://developer.apple.com/macos/"><img src="https://img.shields.io/badge/macOS-14%2B-000000.svg?style=for-the-badge&logo=apple&logoColor=white" alt="macOS 14+"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-000000.svg?style=for-the-badge" alt="MIT License"></a>
  </p>
</div>

---

## ⚡️ The Strong Elements

*   **Metal-Accelerated Scoring:** Parallelized relevance & recency scoring using custom Metal shaders. Verified at **63.36M chunks/sec** and **2.45x GPU math speedup** on large workloads.
*   **Four-Tier Memory:** Working, Episodic, Semantic, and Procedural memory tiers.
*   **Progressive Compression:** Automatically applies light or heavy extractive compression to lower-signal chunks.
*   **Sub-5ms Window Builds:** `buildWindow(500, 4096)` now measures **4.89ms p99** on the latest full release run.
*   **Fast Background Consolidation:** `consolidate(2000)` now measures **15.61ms p99**.
*   **Attention-Aware Reranking:** Re-orders context chunks based on attention centrality.

## 🏗️ Architecture

```mermaid
flowchart TB
    subgraph Client ["Your Application"]
        Input([User Input])
    end

    subgraph Core ["ContextCore Engine"]
        direction TB
        Orch[AgentContext]
        
        subgraph Metal ["Metal Acceleration ⚡️"]
            Scoring[Scoring Kernel]
            Attn[Attention Kernel]
        end
        
        subgraph Mem ["Memory Tiers"]
            Episodic[(Episodic)]
            Semantic[(Semantic)]
            Procedural[(Procedural)]
        end
        
        Packer[Window Packer]
    end

    Input --> Orch
    Orch -->|Query| Mem
    Mem -->|Candidates| Scoring
    Scoring -->|Ranked Chunks| Attn
    Attn -->|Reranked| Packer
    Packer -->|Final Prompt| Model([LLM Inference])

    style Core fill:#fff,stroke:#000,stroke-width:2px,color:#000
    style Metal fill:#000,stroke:#fff,stroke-width:1px,color:#fff
    style Scoring fill:#000,stroke:#fff,stroke-width:1px,color:#fff
    style Attn fill:#000,stroke:#fff,stroke-width:1px,color:#fff
    style Client fill:#fff,stroke:#000,stroke-dasharray: 5 5
    style Model fill:#000,color:#fff
```

## ⚖️ The ContextCore Advantage

| Feature | ❌ Standard LLM Usage | ✅ With ContextCore |
| :--- | :--- | :--- |
| **Recall** | Forgets early conversation turns as context fills. | **Perfect Recall**: Retrieves relevant turns from days ago using semantic search. |
| **Speed** | Slows down linearly as context grows. | **GPU-Tuned**: Window building stays under **5ms p99**, consolidation stays under **16ms p99**, and GPU math reaches **2.45x** CPU speedup at scale. |
| **Cost** | Wastes tokens re-sending irrelevant history. | **Cost Efficient**: Packs only high-value tokens; compresses the rest. |
| **Coherence** | Loses track of long-running tasks. | **Goal Oriented**: "Procedural Memory" tracks tool usage and task patterns. |

## 📊 Performance

ContextCore is designed to run locally on Apple Silicon.

```mermaid
xychart-beta
    title "Window Build Latency (p99) - Lower is Better"
    x-axis [Target Limit, ContextCore (M2)]
    y-axis "Milliseconds (ms)" 0 --> 25
    bar [20.0, 4.89]
```

```mermaid
xychart-beta
    title "Consolidation Time (2000 chunks) - Lower is Better"
    x-axis [Target Limit, ContextCore (M2)]
    y-axis "Milliseconds (ms)" 0 --> 500
    bar [500.0, 15.61]
```

```mermaid
xychart-beta
    title "GPU Math Speedup (50000 chunks) - Higher is Better"
    x-axis [CPU Baseline, ContextCore GPU]
    y-axis "Relative Speed" 0 --> 3
    bar [1.0, 2.45]
```

## 🚀 Quick Start

```swift
import ContextCore

// 1. Initialize Cortex
let context = try AgentContext()

// 2. Start a session
try await context.beginSession(systemPrompt: "You are a senior Swift engineer.")

// 3. Append turns
try await context.append(turn: Turn(role: .user, content: "How do I fix this actor leak?"))

// 4. Build a packed window (Metal-accelerated)
let window = try await context.buildWindow(
    currentTask: "Debug actor isolation",
    maxTokens: 4096
)

// 5. Format for your model
let prompt = window.formatted(style: .chatML)
```

## 🛠 Installation

```swift
dependencies: [
    .package(url: "https://github.com/christopherkarani/ContextCore.git", from: "1.0.0")
]
```

## 📜 License
ContextCore is available under the MIT license. See [LICENSE](LICENSE) for details.
