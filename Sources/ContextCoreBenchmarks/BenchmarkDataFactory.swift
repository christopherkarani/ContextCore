import ContextCore
import Foundation

struct BenchmarkEmbeddingProvider: EmbeddingProvider {
    let dimension: Int = 384

    func embed(_ text: String) async throws -> [Float] {
        Self.vector(for: text, dimension: dimension)
    }

    func embedBatch(_ texts: [String]) async throws -> [[Float]] {
        texts.map { Self.vector(for: $0, dimension: dimension) }
    }

    private static func vector(for text: String, dimension: Int) -> [Float] {
        var state = stableSeed(from: text)
        var values = [Float](repeating: 0, count: dimension)

        for index in values.indices {
            state &*= 6364136223846793005
            state &+= 1442695040888963407
            let component = Float(Int64(bitPattern: state & 0x0000_FFFF_FFFF_FFFF) % 10_000) / 5_000.0 - 1.0
            values[index] = component
        }

        return l2Normalize(values)
    }

    private static func stableSeed(from text: String) -> UInt64 {
        var hash: UInt64 = 1469598103934665603
        for byte in text.utf8 {
            hash ^= UInt64(byte)
            hash &*= 1099511628211
        }
        return hash
    }

    private static func l2Normalize(_ vector: [Float]) -> [Float] {
        let norm = vector.reduce(0) { partial, value in
            partial + (value * value)
        }.squareRoot()

        guard norm > 0 else {
            return vector
        }

        return vector.map { $0 / norm }
    }
}

enum BenchmarkDataFactory {
    static let realisticTurnTemplates: [String] = [
        "User asks for help debugging a Swift concurrency crash around actor isolation and Sendable conformance.",
        "Assistant proposes an async/await refactor and provides a short code snippet with TaskGroup usage.",
        "Tool call output: build logs indicate a non-Sendable capture inside a detached task in NetworkService.swift.",
        "User shares a failing unit test and asks for a deterministic fix that keeps API behavior unchanged.",
        "Assistant suggests a minimal patch with MainActor isolation and updated protocol constraints.",
        "Tool call output: integration tests pass on macOS but fail on iOS simulator due token-count drift.",
        "User asks for benchmark guidance and wants p99 latency and throughput metrics for production readiness.",
        "Assistant explains retrieval vs recency weighting and proposes tuning episodicMemoryK and semanticMemoryK.",
        "Tool call output: profiler shows scoring kernel is fast but consolidation spikes when duplicates accumulate.",
        "User asks for a release checklist including docs, sanitizer runs, and platform compatibility validation."
    ]

    static let relevantConcurrencyTemplates: [String] = [
        "Swift actors serialize access to mutable state and prevent unsynchronized cross-thread mutation.",
        "Actor-isolated methods require await from outside the actor boundary.",
        "Sendable types are required for values crossing concurrency domains.",
        "Structured concurrency encourages task cancellation propagation.",
        "Use MainActor for UI-bound state transitions in Swift applications."
    ]

    static let irrelevantTemplates: [String] = [
        "Recipe note: roast vegetables at 220C and season after resting.",
        "Travel plan: local weather looks clear with light winds in the afternoon.",
        "Coffee guide: coarse grind improves extraction for cold brew concentrate.",
        "Home maintenance reminder: replace HVAC filter every three months.",
        "Fitness tip: progressive overload improves strength adaptation over time."
    ]

    static func makeConfiguration() -> ContextConfiguration {
        var config = ContextConfiguration.default
        config.embeddingProvider = BenchmarkEmbeddingProvider()
        config.consolidationThreshold = Int.max
        return config
    }

    static func realisticTurnContent(index: Int) -> String {
        let template = realisticTurnTemplates[index % realisticTurnTemplates.count]
        return "[Turn \(index)] \(template)"
    }

    static func duplicateProneContent(index: Int) -> String {
        if index % 10 == 0 {
            return realisticTurnContent(index: index / 10)
        }
        return realisticTurnContent(index: index)
    }

    static func relevantConcurrencyContent(index: Int) -> String {
        let template = relevantConcurrencyTemplates[index % relevantConcurrencyTemplates.count]
        return "[Relevant \(index)] \(template)"
    }

    static func irrelevantContent(index: Int) -> String {
        let template = irrelevantTemplates[index % irrelevantTemplates.count]
        return "[Irrelevant \(index)] \(template)"
    }

    static func randomVector(dim: Int, seed: UInt64) -> [Float] {
        var rng = SeededGenerator(seed: seed)
        return randomVector(dim: dim, using: &rng)
    }

    static func randomVectors(n: Int, dim: Int, seed: UInt64) -> [[Float]] {
        var rng = SeededGenerator(seed: seed)
        return (0..<n).map { _ in randomVector(dim: dim, using: &rng) }
    }

    private static func randomVector(dim: Int, using rng: inout SeededGenerator) -> [Float] {
        var values = [Float](repeating: 0, count: dim)
        for index in values.indices {
            let raw = Float(rng.next() & 0xFFFF) / Float(0xFFFF)
            values[index] = raw * 2 - 1
        }
        return normalize(values)
    }

    private static func normalize(_ values: [Float]) -> [Float] {
        let norm = values.reduce(Float.zero) { $0 + ($1 * $1) }.squareRoot()
        guard norm > 0 else {
            return values
        }
        return values.map { $0 / norm }
    }

    struct SeededGenerator: RandomNumberGenerator {
        private var state: UInt64

        init(seed: UInt64) {
            self.state = seed
        }

        mutating func next() -> UInt64 {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return state
        }
    }
}
