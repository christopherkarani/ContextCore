public protocol CompressionDelegate: Sendable {
    func compress(_ text: String, targetTokens: Int) async throws -> String
    func extractFacts(from text: String) async throws -> [String]
}
