import Foundation
import MetalANNS

public actor SemanticStore {
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

    public func insert(
        content: String,
        embedding: [Float],
        metadata: [String: String] = [:]
    ) async throws {
        try validateDimension(embedding)

        let chunk = MemoryChunk(
            content: content,
            embedding: embedding,
            type: .semantic,
            createdAt: .now,
            lastAccessedAt: .now,
            accessCount: 1,
            retentionScore: 1.0,
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

    public func upsert(fact: String, embedding: [Float]) async throws {
        try validateDimension(embedding)

        if let existingID = bestMatchingChunkID(for: embedding, threshold: 0.9),
           var existing = chunksByID[existingID]
        {
            existing.lastAccessedAt = .now
            existing.accessCount += 1
            chunksByID[existingID] = existing
            return
        }

        try await insert(
            content: fact,
            embedding: embedding,
            metadata: ["kind": "fact"]
        )
    }

    private func validateDimension(_ embedding: [Float]) throws {
        if let expected = embeddingDimension, expected != embedding.count {
            throw NSError(
                domain: "ContextCore.SemanticStore",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Dimension mismatch. expected: \(expected), got: \(embedding.count)"]
            )
        }
        if embeddingDimension == nil {
            embeddingDimension = embedding.count
        }
    }

    private func bestMatchingChunkID(for embedding: [Float], threshold: Float) -> String? {
        var bestID: String?
        var bestSimilarity: Float = -1

        for (id, chunk) in chunksByID {
            let similarity = cosineSimilarity(embedding, chunk.embedding)
            if similarity > bestSimilarity {
                bestSimilarity = similarity
                bestID = id
            }
        }

        guard let bestID, bestSimilarity > threshold else {
            return nil
        }
        return bestID
    }

    private func cosineSimilarity(_ lhs: [Float], _ rhs: [Float]) -> Float {
        guard lhs.count == rhs.count else {
            return -1
        }

        var dot: Float = 0
        var lhsNorm: Float = 0
        var rhsNorm: Float = 0

        for index in lhs.indices {
            dot += lhs[index] * rhs[index]
            lhsNorm += lhs[index] * lhs[index]
            rhsNorm += rhs[index] * rhs[index]
        }

        let denominator = (lhsNorm.squareRoot() * rhsNorm.squareRoot())
        guard denominator > 0 else {
            return -1
        }
        return dot / denominator
    }
}
