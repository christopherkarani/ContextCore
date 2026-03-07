# ContextCore Twitter Thread

---

**1/**
We just open-sourced ContextCore — a GPU-accelerated context management system for AI agents on Apple Silicon.

Your LLM forgets what happened 20 turns ago? ContextCore doesn't.

Sub-5ms latency. Perfect recall. Native Swift.

Here's how it works 🧵

---

**2/**
The problem: as conversations grow longer, LLMs fall apart.

- Early turns get forgotten
- Performance degrades linearly
- Irrelevant history wastes your token budget
- Long-running tasks lose coherence

Every AI agent developer has hit this wall.

---

**3/**
ContextCore fixes this with a four-tier memory architecture:

- Episodic Memory → turn-level conversation history
- Semantic Memory → durable facts promoted over time
- Procedural Memory → learned tool usage patterns
- Working Memory → the optimized context window sent to the model

---

**4/**
The magic is in how it builds that working memory.

Every call, ContextCore:
→ Embeds your query
→ Retrieves relevant memory across all tiers
→ Scores each chunk by relevance + recency (on the GPU)
→ Packs the best context under your token budget
→ Orders chunks for optimal model attention

All in under 5ms.

---

**5/**
We went all-in on Metal compute shaders for Apple Silicon.

The result:
- 63M+ chunks/sec scoring throughput
- 2.45x GPU speedup over CPU at scale
- 3.29ms window builds (p99)
- 15.61ms consolidation of 2,000 chunks (p99)

This isn't a proof of concept. These are M2 MacBook benchmarks.

---

**6/**
Consolidation runs automatically in the background.

As your session grows, ContextCore:
- Detects duplicate episodic memories
- Promotes high-value facts to long-term semantic memory (90-day half-life)
- Evicts low-retention noise

Your context self-optimizes. No manual pruning.

---

**7/**
Progressive compression kicks in only when you're budget-constrained.

Low-signal chunks get extractively compressed. High-signal chunks stay intact. Sentence importance is scored on the GPU.

You keep the information that matters and spend tokens where they count.

---

**8/**
Beyond cosine similarity — ContextCore uses attention-aware reranking.

Metal shaders compute token centrality (how connected a chunk is to others) and cross-attention scores (how relevant to the current task).

The result: context windows that are semantically coherent, not just individually relevant.

---

**9/**
Built for real production use:

- Full checkpoint/restore persistence
- 15+ configurable tuning parameters
- Session tracking and runtime stats
- Configurable half-lives per memory tier
- Token safety margins
- ChatML, Alpaca, and custom formatting

---

**10/**
Getting started is simple:

```swift
let context = AgentContext(configuration: .default)
await context.ingest(turn: userMessage)
let window = await context.buildWindow(
    maxTurns: 500,
    tokenBudget: 4096
)
// window.messages → ready for your LLM
```

That's it. Five lines to production-grade memory.

---

**11/**
Who is this for?

- AI agent developers building on macOS/iOS
- Teams optimizing token costs at scale
- Anyone tired of naive sliding-window context
- Swift developers who want native Apple Silicon performance

---

**12/**
ContextCore is open source and ready to use today.

Check it out, star the repo, and let us know what you build with it.

GitHub: github.com/christopherkarani/ContextCore

We're just getting started.

---
