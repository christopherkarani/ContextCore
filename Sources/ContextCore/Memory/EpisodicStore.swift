import Foundation
import MetalANNS

public actor EpisodicStore {
    private let index: Advanced.StreamingIndex
    private var chunksByID: [String: MemoryChunk] = [:]
    private let sourceSessionID: UUID
    private var embeddingDimension: Int?

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
            throw NSError(
                domain: "ContextCore.EpisodicStore",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Turn embedding is nil"]
            )
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

    private func validateDimension(_ embedding: [Float]) throws {
        if let expected = embeddingDimension, expected != embedding.count {
            throw NSError(
                domain: "ContextCore.EpisodicStore",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Dimension mismatch. expected: \(expected), got: \(embedding.count)"]
            )
        }
        if embeddingDimension == nil {
            embeddingDimension = embedding.count
        }
    }
}
