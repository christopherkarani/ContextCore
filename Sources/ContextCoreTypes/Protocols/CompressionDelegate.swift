public protocol CompressionDelegate: Sendable {
    /// Compress text to fit within targetTokens.
    /// Implementations can be extractive or abstractive.
    func compress(_ text: String, targetTokens: Int) async throws -> String

    /// Extract standalone factual statements from text.
    func extractFacts(from text: String) async throws -> [String]
}
