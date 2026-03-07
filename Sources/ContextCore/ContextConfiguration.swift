/// Runtime tuning parameters for ``AgentContext``.
public struct ContextConfiguration: Sendable {
    /// Maximum token budget before safety margin is applied.
    public var maxTokens: Int
    /// Reserved fraction of budget to absorb token-count drift.
    public var tokenBudgetSafetyMargin: Float
    /// Number of episodic candidates retained per build.
    public var episodicMemoryK: Int
    /// Number of semantic candidates retained per build.
    public var semanticMemoryK: Int
    /// Number of latest turns guaranteed in every window.
    public var recentTurnsGuaranteed: Int
    /// Half-life for episodic recency decay in days.
    public var episodicHalfLifeDays: Double
    /// Half-life for semantic recency decay in days.
    public var semanticHalfLifeDays: Double
    /// Episodic-count threshold that triggers auto-consolidation.
    public var consolidationThreshold: Int
    /// Similarity threshold used for duplicate merge detection.
    public var similarityMergeThreshold: Float
    /// Weight assigned to semantic similarity scoring.
    public var relevanceWeight: Float
    /// Weight assigned to attention centrality scoring.
    public var centralityWeight: Float
    /// Minimum retention score for chunks to be included in recall results.
    public var minimumRetentionScore: Float
    /// Maximum allowed checkpoint file size in bytes (0 = unlimited).
    public var maxCheckpointBytes: Int
    /// Maximum text length (in characters) accepted by the embedding provider.
    public var maxEmbeddingTextLength: Int
    /// ANN search breadth parameter.
    public var efSearch: Int
    /// Embedding backend used for query and chunk vectors.
    public var embeddingProvider: any EmbeddingProvider
    /// Token counter used for packing decisions.
    public var tokenCounter: any TokenCounter
    /// Optional delegate for abstractive compression and fact extraction.
    public var compressionDelegate: (any CompressionDelegate)?

    /// Creates a fully specified configuration.
    ///
    /// - Parameters:
    ///   - maxTokens: Maximum token budget before margin.
    ///   - tokenBudgetSafetyMargin: Reserved budget fraction.
    ///   - episodicMemoryK: Episodic retrieval count.
    ///   - semanticMemoryK: Semantic retrieval count.
    ///   - recentTurnsGuaranteed: Number of guaranteed recent turns.
    ///   - episodicHalfLifeDays: Episodic half-life in days.
    ///   - semanticHalfLifeDays: Semantic half-life in days.
    ///   - consolidationThreshold: Auto-consolidation threshold.
    ///   - similarityMergeThreshold: Duplicate merge threshold.
    ///   - relevanceWeight: Similarity weight.
    ///   - centralityWeight: Centrality weight.
    ///   - minimumRetentionScore: Minimum retention score for recall filtering.
    ///   - maxCheckpointBytes: Maximum checkpoint file size in bytes (0 = unlimited).
    ///   - maxEmbeddingTextLength: Maximum text length for embedding (0 = unlimited).
    ///   - efSearch: ANN search breadth.
    ///   - embeddingProvider: Embedding backend.
    ///   - tokenCounter: Token counter backend.
    ///   - compressionDelegate: Optional compression delegate.
    public init(
        maxTokens: Int,
        tokenBudgetSafetyMargin: Float,
        episodicMemoryK: Int,
        semanticMemoryK: Int,
        recentTurnsGuaranteed: Int,
        episodicHalfLifeDays: Double,
        semanticHalfLifeDays: Double,
        consolidationThreshold: Int,
        similarityMergeThreshold: Float,
        relevanceWeight: Float,
        centralityWeight: Float,
        minimumRetentionScore: Float = 0.01,
        maxCheckpointBytes: Int = 0,
        maxEmbeddingTextLength: Int = 0,
        efSearch: Int,
        embeddingProvider: any EmbeddingProvider,
        tokenCounter: any TokenCounter,
        compressionDelegate: (any CompressionDelegate)? = nil
    ) {
        self.maxTokens = maxTokens
        self.tokenBudgetSafetyMargin = tokenBudgetSafetyMargin
        self.episodicMemoryK = episodicMemoryK
        self.semanticMemoryK = semanticMemoryK
        self.recentTurnsGuaranteed = recentTurnsGuaranteed
        self.episodicHalfLifeDays = episodicHalfLifeDays
        self.semanticHalfLifeDays = semanticHalfLifeDays
        self.consolidationThreshold = consolidationThreshold
        self.similarityMergeThreshold = similarityMergeThreshold
        self.relevanceWeight = relevanceWeight
        self.centralityWeight = centralityWeight
        self.minimumRetentionScore = minimumRetentionScore
        self.maxCheckpointBytes = maxCheckpointBytes
        self.maxEmbeddingTextLength = maxEmbeddingTextLength
        self.efSearch = efSearch
        self.embeddingProvider = embeddingProvider
        self.tokenCounter = tokenCounter
        self.compressionDelegate = compressionDelegate
    }

    /// Production-oriented default configuration.
    public static var `default`: ContextConfiguration {
        ContextConfiguration(
            maxTokens: 4096,
            tokenBudgetSafetyMargin: 0.10,
            episodicMemoryK: 8,
            semanticMemoryK: 4,
            recentTurnsGuaranteed: 3,
            episodicHalfLifeDays: 7,
            semanticHalfLifeDays: 90,
            consolidationThreshold: 200,
            similarityMergeThreshold: 0.92,
            relevanceWeight: 0.7,
            centralityWeight: 0.4,
            minimumRetentionScore: 0.01,
            maxCheckpointBytes: 100_000_000,
            maxEmbeddingTextLength: 100_000,
            efSearch: 64,
            embeddingProvider: CachingEmbeddingProvider(
                base: CoreMLEmbeddingProvider(),
                cache: EmbeddingCache(capacity: 512)
            ),
            tokenCounter: ApproximateTokenCounter(),
            compressionDelegate: nil
        )
    }
}
