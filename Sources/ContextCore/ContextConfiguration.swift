public struct ContextConfiguration: Sendable {
    public var maxTokens: Int
    public var tokenBudgetSafetyMargin: Float
    public var episodicMemoryK: Int
    public var semanticMemoryK: Int
    public var recentTurnsGuaranteed: Int
    public var episodicHalfLifeDays: Double
    public var semanticHalfLifeDays: Double
    public var consolidationThreshold: Int
    public var similarityMergeThreshold: Float
    public var relevanceWeight: Float
    public var centralityWeight: Float
    public var efSearch: Int
    public var embeddingProvider: any EmbeddingProvider
    public var tokenCounter: any TokenCounter
    public var compressionDelegate: (any CompressionDelegate)?

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
        self.efSearch = efSearch
        self.embeddingProvider = embeddingProvider
        self.tokenCounter = tokenCounter
        self.compressionDelegate = compressionDelegate
    }

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
