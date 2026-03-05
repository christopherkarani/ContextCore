import Foundation
import MetalANNS

public actor EpisodicStore: ConsolidationEpisodicStore {
    private let index: Advanced.StreamingIndex
    private var chunksByID: [String: MemoryChunk] = [:]
    private let sourceSessionID: UUID
    private var embeddingDimension: Int?
    private let consolidatedMarkerKey = "consolidated.phase4"

    public init(sourceSessionID: UUID = UUID()) {
        self.sourceSessionID = sourceSessionID
        let config = StreamingConfiguration(
            deltaCapacity: 1_024,
            mergeStrategy: .blocking,
            indexConfiguration: IndexConfiguration(metric: .cosine)
        )
        self.index = Advanced.StreamingIndex(config: config)
    }

    public var count: Int {
        chunksByID.count
    }

    public func insert(turn: Turn) async throws {
        guard let embedding = turn.embedding else {
            throw ContextCoreError.embeddingFailed("Turn embedding is nil")
        }

        try validateDimension(embedding)

        var metadata = turn.metadata
        metadata["turnRole"] = turn.role.rawValue
        metadata["turnID"] = turn.id.uuidString

        let chunk = MemoryChunk(
            id: turn.id,
            content: turn.content,
            embedding: embedding,
            type: .episodic,
            createdAt: turn.timestamp,
            lastAccessedAt: .now,
            accessCount: 1,
            retentionScore: 0.5,
            sourceSessionID: sourceSessionID,
            metadata: metadata
        )

        let chunkID = chunk.id.uuidString
        try await index.insert(embedding, id: chunkID)
        chunksByID[chunkID] = chunk
    }

    public func retrieve(query: [Float], k: Int) async throws -> [MemoryChunk] {
        guard !chunksByID.isEmpty else {
            return []
        }
        guard k > 0 else {
            return []
        }

        try validateDimension(query)
        let results = try await index.search(query: query, k: k)

        var retrieved: [MemoryChunk] = []
        for result in results {
            guard let chunk = chunksByID[result.id] else {
                continue
            }
            retrieved.append(chunk)
        }

        return retrieved
    }

    public func allChunks() async -> [MemoryChunk] {
        chunksByID.values.sorted { lhs, rhs in
            if lhs.createdAt != rhs.createdAt {
                return lhs.createdAt < rhs.createdAt
            }
            return lhs.id.uuidString < rhs.id.uuidString
        }
    }

    public func updateRetentionScore(id: UUID, delta: Float) async throws {
        let key = id.uuidString
        guard var chunk = chunksByID[key] else {
            throw ContextCoreError.chunkNotFound(id: id)
        }
        let updated = max(0, min(1, chunk.retentionScore + delta))
        chunk.retentionScore = updated
        chunksByID[key] = chunk
    }

    public func evict(id: UUID) async throws {
        let key = id.uuidString
        guard chunksByID[key] != nil else {
            throw ContextCoreError.chunkNotFound(id: id)
        }
        try await index.delete(id: key)
        chunksByID.removeValue(forKey: key)
    }

    public func markConsolidated(id: UUID) async throws {
        let key = id.uuidString
        guard var chunk = chunksByID[key] else {
            throw ContextCoreError.chunkNotFound(id: id)
        }
        chunk.metadata[consolidatedMarkerKey] = "true"
        chunksByID[key] = chunk
    }

    public func isConsolidated(id: UUID) async -> Bool {
        let key = id.uuidString
        guard let chunk = chunksByID[key] else {
            return false
        }
        return chunk.metadata[consolidatedMarkerKey] == "true"
    }

    private func validateDimension(_ embedding: [Float]) throws {
        if let expected = embeddingDimension, expected != embedding.count {
            throw ContextCoreError.dimensionMismatch(expected: expected, got: embedding.count)
        }
        if embeddingDimension == nil {
            embeddingDimension = embedding.count
        }
    }
}
