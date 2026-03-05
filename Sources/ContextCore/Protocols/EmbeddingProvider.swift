public protocol EmbeddingProvider: Sendable {
    func embed(_ text: String) async throws -> [Float]
    func embedBatch(_ texts: [String]) async throws -> [[Float]]
    var dimension: Int { get }
}
