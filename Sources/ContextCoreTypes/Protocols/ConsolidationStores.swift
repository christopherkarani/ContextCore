import Foundation

public protocol ConsolidationEpisodicStore: Sendable {
    var count: Int { get async }
    func allChunks() async -> [MemoryChunk]
    func updateRetentionScore(id: UUID, delta: Float) async throws
    func evict(id: UUID) async throws
    func markConsolidated(id: UUID) async throws
    func isConsolidated(id: UUID) async -> Bool
}

public protocol ConsolidationSemanticStore: Sendable {
    var count: Int { get async }
    func allChunks() async -> [MemoryChunk]
    func upsert(fact: String, embedding: [Float]) async throws
}
