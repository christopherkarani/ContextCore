# ContextCore Tweet Thread

**1/**
Introducing ContextCore — a GPU-accelerated context management framework for on-device AI agents on Apple Silicon.

Your AI agent forgets what you said 10 minutes ago? ContextCore fixes that.

Open source. Swift 6.2. Sub-5ms latency. Let me explain 🧵

**2/**
The problem: LLMs have fixed context windows. As conversations grow, you lose early context. You waste tokens resending irrelevant history. Latency scales linearly.

ContextCore sits between your app and the model, deciding exactly what context matters — in real time.

**3/**
It uses a four-tier memory architecture:

→ Working Memory: what's in the current prompt
→ Episodic Memory: conversation turns stored as vectors
→ Semantic Memory: durable facts promoted over time
→ Procedural Memory: learned tool-usage patterns

Your agent remembers like a human does.

**4/**
Your agent had a debugging session 3 days ago? ContextCore retrieves exactly the relevant parts using semantic search — not just "last N messages."

Perfect recall from days or weeks of history, packed into a tight token budget.

**5/**
Performance? We benchmarked it:

• Window builds: 4.89ms p99 (500 candidates, 4096 tokens)
• Consolidation: 15.61ms p99 (2000 chunks)
• GPU scoring: 63.36M chunks/sec
• 2.45x speedup over CPU on large workloads

All on an M2. All on-device.

**6/**
Progressive compression keeps you under budget without losing signal:

• High-relevance chunks → sent in full
• Medium → light compression (50%)
• Low → heavy compression (75%)
• Noise → evicted entirely

No wasted tokens. No hallucination from missing context.

**7/**
The attention-aware reranker prevents a subtle failure mode: dropping context that's low-relevance individually but crucial for model coherence.

It blends relevance scores with centrality scores — keeping the connective tissue of your conversation intact.

**8/**
Everything runs locally on Apple Silicon via Metal shaders:

• Relevance scoring
• Recency weighting
• Pairwise similarity
• Contradiction detection
• Centrality computation

Zero cloud calls. Full privacy. Native Swift concurrency with actor isolation.

**9/**
Auto-consolidation runs in the background:

→ Deduplicates near-identical memories (0.92 similarity threshold)
→ Promotes durable facts from episodic → semantic
→ Detects contradictions via antipodal scoring
→ Evicts low-value chunks

Your memory stays lean without manual curation.

**10/**
Integration is simple — here's the core loop:

```swift
let context = try AgentContext()
try await context.beginSession(systemPrompt: "...")

// Append turns
try await context.append(turn: userTurn)

// Build optimized window
let window = try await context.buildWindow(
    currentTask: query,
    maxTokens: 4096
)

// Send to model
let prompt = window.formatted(style: .chatML)
```

**11/**
13+ tunable parameters let you optimize for your workload:

• Recency half-life (episodic vs semantic)
• Compression thresholds
• Guaranteed recent turns
• Consolidation triggers
• Similarity thresholds

One config struct. Full control.

**12/**
Built for real use cases:

• Code assistants that remember your patterns across sessions
• Research agents that retain facts from weeks of exploration
• Support bots that learn which tools solve which problems
• Any long-running agent on iOS, macOS, or visionOS

**13/**
ContextCore is open source and ready to use:

→ Swift Package Manager
→ iOS 17+ / macOS 14+ / visionOS 1+
→ Swift 6.2 with strict concurrency
→ Full DocC documentation
→ Benchmarks included

github.com/nickthedude/ContextCore

Build agents that actually remember. ⚡
