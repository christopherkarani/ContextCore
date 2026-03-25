import Foundation
@testable import ContextCore

struct AgentTestEmbeddingProvider: EmbeddingProvider, Sendable {
    let dimension: Int

    init(dimension: Int = 384) {
        self.dimension = dimension
    }

    func embed(_ text: String) async throws -> [Float] {
        Self.vector(for: text, dimension: dimension)
    }

    func embedBatch(_ texts: [String]) async throws -> [[Float]] {
        texts.map { Self.vector(for: $0, dimension: dimension) }
    }

    private static func vector(for text: String, dimension: Int) -> [Float] {
        var hash: UInt64 = 1469598103934665603
        for byte in text.utf8 {
            hash ^= UInt64(byte)
            hash &*= 1099511628211
        }

        var state = hash
        var values = [Float](repeating: 0, count: dimension)
        for index in values.indices {
            state &*= 6364136223846793005
            state &+= 1442695040888963407
            let component = Float(Int64(bitPattern: state & 0x0000_FFFF_FFFF_FFFF) % 10_000) / 5_000.0 - 1.0
            values[index] = component
        }

        let norm = values.reduce(Float.zero) { $0 + ($1 * $1) }.squareRoot()
        guard norm > 0 else {
            return values
        }
        return values.map { $0 / norm }
    }
}

struct FailingEmbeddingProvider: EmbeddingProvider, Sendable {
    let dimension: Int = 384

    func embed(_ text: String) async throws -> [Float] {
        throw NSError(domain: "test.embedding", code: 1, userInfo: [NSLocalizedDescriptionKey: "Embedding failed"])
    }

    func embedBatch(_ texts: [String]) async throws -> [[Float]] {
        throw NSError(domain: "test.embedding", code: 2, userInfo: [NSLocalizedDescriptionKey: "Batch embedding failed"])
    }
}

func makeAgentConfiguration(
    provider: any EmbeddingProvider = AgentTestEmbeddingProvider(),
    maxTokens: Int = 4096,
    tokenSafetyMargin: Float = 0.10,
    consolidationThreshold: Int = 200
) -> ContextConfiguration {
    var config = ContextConfiguration.default
    config.embeddingProvider = provider
    config.tokenCounter = ApproximateTokenCounter()
    config.maxTokens = maxTokens
    config.tokenBudgetSafetyMargin = tokenSafetyMargin
    config.consolidationThreshold = consolidationThreshold
    config.episodicMemoryK = max(config.episodicMemoryK, 8)
    config.semanticMemoryK = max(config.semanticMemoryK, 4)
    config.recentTurnsGuaranteed = 3
    config.similarityMergeThreshold = 0.92
    return config
}

func makeDiverseTurns() -> [Turn] {
    [
        Turn(role: .user, content: "Help me set up a Swift Package Manager project"),
        Turn(role: .assistant, content: "Run swift package init --type library to start a package."),
        Turn(role: .user, content: "How do I add a dependency in Package.swift?"),
        Turn(role: .assistant, content: "Use .package(url: ..., from: \"1.0.0\") in dependencies."),
        Turn(role: .user, content: "Explain actors versus classes in Swift."),
        Turn(role: .assistant, content: "Actors isolate mutable state and prevent data races."),
        Turn(role: .user, content: "Can you explain async await?"),
        Turn(role: .assistant, content: "Mark functions async and call them with await."),
        Turn(role: .user, content: "What is Metal used for?"),
        Turn(role: .assistant, content: "Metal accelerates graphics and compute workloads on Apple GPUs."),
    ]
}
