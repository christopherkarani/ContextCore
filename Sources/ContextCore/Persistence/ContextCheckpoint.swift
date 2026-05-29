import Foundation

/// Serializable snapshot of ContextCore runtime state.
public struct ContextCheckpoint: Codable, Sendable {
    /// Serializable subset of ``ContextConfiguration`` persisted in checkpoints.
    public struct ConfigurationSnapshot: Codable, Sendable {
        /// Persisted `maxTokens`.
        public let maxTokens: Int
        /// Persisted `tokenBudgetSafetyMargin`.
        public let tokenBudgetSafetyMargin: Float
        /// Persisted `episodicMemoryK`.
        public let episodicMemoryK: Int
        /// Persisted `semanticMemoryK`.
        public let semanticMemoryK: Int
        /// Persisted `recentTurnsGuaranteed`.
        public let recentTurnsGuaranteed: Int
        /// Persisted `episodicHalfLifeDays`.
        public let episodicHalfLifeDays: Double
        /// Persisted `semanticHalfLifeDays`.
        public let semanticHalfLifeDays: Double
        /// Persisted `consolidationThreshold`.
        public let consolidationThreshold: Int
        /// Persisted `similarityMergeThreshold`.
        public let similarityMergeThreshold: Float
        /// Persisted `relevanceWeight`.
        public let relevanceWeight: Float
        /// Persisted `centralityWeight`.
        public let centralityWeight: Float
        /// Persisted `minimumRetentionScore`.
        public let minimumRetentionScore: Float
        /// Persisted `maxCheckpointBytes`.
        public let maxCheckpointBytes: Int
        /// Persisted `maxEmbeddingTextLength`.
        public let maxEmbeddingTextLength: Int
        /// Persisted `efSearch`.
        public let efSearch: Int

        /// Creates a configuration snapshot from runtime config.
        ///
        /// - Parameter configuration: Source configuration.
        public init(configuration: ContextConfiguration) {
            self.maxTokens = configuration.maxTokens
            self.tokenBudgetSafetyMargin = configuration.tokenBudgetSafetyMargin
            self.episodicMemoryK = configuration.episodicMemoryK
            self.semanticMemoryK = configuration.semanticMemoryK
            self.recentTurnsGuaranteed = configuration.recentTurnsGuaranteed
            self.episodicHalfLifeDays = configuration.episodicHalfLifeDays
            self.semanticHalfLifeDays = configuration.semanticHalfLifeDays
            self.consolidationThreshold = configuration.consolidationThreshold
            self.similarityMergeThreshold = configuration.similarityMergeThreshold
            self.relevanceWeight = configuration.relevanceWeight
            self.centralityWeight = configuration.centralityWeight
            self.minimumRetentionScore = configuration.minimumRetentionScore
            self.maxCheckpointBytes = configuration.maxCheckpointBytes
            self.maxEmbeddingTextLength = configuration.maxEmbeddingTextLength
            self.efSearch = configuration.efSearch
        }

        // Provide defaults for fields added after schema version 1.
        enum CodingKeys: String, CodingKey {
            case maxTokens, tokenBudgetSafetyMargin, episodicMemoryK, semanticMemoryK
            case recentTurnsGuaranteed, episodicHalfLifeDays, semanticHalfLifeDays
            case consolidationThreshold, similarityMergeThreshold, relevanceWeight
            case centralityWeight, minimumRetentionScore, maxCheckpointBytes
            case maxEmbeddingTextLength, efSearch
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            maxTokens = try container.decode(Int.self, forKey: .maxTokens)
            tokenBudgetSafetyMargin = try container.decode(Float.self, forKey: .tokenBudgetSafetyMargin)
            episodicMemoryK = try container.decode(Int.self, forKey: .episodicMemoryK)
            semanticMemoryK = try container.decode(Int.self, forKey: .semanticMemoryK)
            recentTurnsGuaranteed = try container.decode(Int.self, forKey: .recentTurnsGuaranteed)
            episodicHalfLifeDays = try container.decode(Double.self, forKey: .episodicHalfLifeDays)
            semanticHalfLifeDays = try container.decode(Double.self, forKey: .semanticHalfLifeDays)
            consolidationThreshold = try container.decode(Int.self, forKey: .consolidationThreshold)
            similarityMergeThreshold = try container.decode(Float.self, forKey: .similarityMergeThreshold)
            relevanceWeight = try container.decode(Float.self, forKey: .relevanceWeight)
            centralityWeight = try container.decode(Float.self, forKey: .centralityWeight)
            minimumRetentionScore = try container.decodeIfPresent(Float.self, forKey: .minimumRetentionScore) ?? 0.01
            maxCheckpointBytes = try container.decodeIfPresent(Int.self, forKey: .maxCheckpointBytes) ?? 0
            maxEmbeddingTextLength = try container.decodeIfPresent(Int.self, forKey: .maxEmbeddingTextLength) ?? 0
            efSearch = try container.decode(Int.self, forKey: .efSearch)
        }

        /// Applies persisted scalar values onto a base configuration.
        ///
        /// - Parameter base: Base configuration providing runtime dependencies.
        /// - Returns: Reconstructed configuration.
        public func apply(to base: ContextConfiguration) -> ContextConfiguration {
            ContextConfiguration(
                maxTokens: maxTokens,
                tokenBudgetSafetyMargin: tokenBudgetSafetyMargin,
                episodicMemoryK: episodicMemoryK,
                semanticMemoryK: semanticMemoryK,
                recentTurnsGuaranteed: recentTurnsGuaranteed,
                episodicHalfLifeDays: episodicHalfLifeDays,
                semanticHalfLifeDays: semanticHalfLifeDays,
                consolidationThreshold: consolidationThreshold,
                similarityMergeThreshold: similarityMergeThreshold,
                relevanceWeight: relevanceWeight,
                centralityWeight: centralityWeight,
                minimumRetentionScore: minimumRetentionScore,
                maxCheckpointBytes: maxCheckpointBytes,
                maxEmbeddingTextLength: maxEmbeddingTextLength,
                efSearch: efSearch,
                embeddingProvider: base.embeddingProvider,
                tokenCounter: base.tokenCounter,
                compressionDelegate: base.compressionDelegate
            )
        }
    }

    /// Checkpoint schema version.
    public let version: Int
    /// Checkpoint creation timestamp.
    public let createdAt: Date
    /// Persisted episodic chunks.
    public let episodicChunks: [MemoryChunk]
    /// Persisted semantic chunks.
    public let semanticChunks: [MemoryChunk]
    /// Persisted procedural patterns.
    public let proceduralPatterns: [String: [ToolCall]]
    /// Last active session identifier.
    public let lastSessionID: UUID?
    /// Persisted system prompt.
    public let systemPrompt: String?
    /// Recent turn buffer.
    public let recentTurns: [Turn]
    /// Total session counter.
    public let totalSessions: Int
    /// Persisted configuration snapshot.
    public let configuration: ConfigurationSnapshot

    /// Creates a checkpoint snapshot.
    ///
    /// - Parameters:
    ///   - version: Checkpoint schema version.
    ///   - createdAt: Creation timestamp.
    ///   - episodicChunks: Episodic chunks to persist.
    ///   - semanticChunks: Semantic chunks to persist.
    ///   - proceduralPatterns: Procedural patterns to persist.
    ///   - lastSessionID: Last active session identifier.
    ///   - systemPrompt: Active system prompt.
    ///   - recentTurns: Recent turn buffer.
    ///   - totalSessions: Total sessions started.
    ///   - configuration: Configuration snapshot.
    public init(
        version: Int = 1,
        createdAt: Date = .now,
        episodicChunks: [MemoryChunk],
        semanticChunks: [MemoryChunk],
        proceduralPatterns: [String: [ToolCall]],
        lastSessionID: UUID?,
        systemPrompt: String?,
        recentTurns: [Turn],
        totalSessions: Int,
        configuration: ConfigurationSnapshot
    ) {
        self.version = version
        self.createdAt = createdAt
        self.episodicChunks = episodicChunks
        self.semanticChunks = semanticChunks
        self.proceduralPatterns = proceduralPatterns
        self.lastSessionID = lastSessionID
        self.systemPrompt = systemPrompt
        self.recentTurns = recentTurns
        self.totalSessions = totalSessions
        self.configuration = configuration
    }
}
