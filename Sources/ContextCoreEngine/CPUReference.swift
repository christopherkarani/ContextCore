import Accelerate
import ContextCoreTypes
import Foundation

public enum CPUReference {
    public static func relevanceScores(
        query: [Float],
        chunks: [[Float]],
        recencyWeights: [Float],
        relevanceWeight: Float = 0.7,
        recencyWeight: Float = 0.3
    ) -> [Float] {
        guard !chunks.isEmpty else {
            return []
        }
        precondition(query.count == chunks[0].count, "Query dimension must match chunk dimension")
        precondition(chunks.count == recencyWeights.count, "Chunk count must match recency weight count")

        var queryNormSquared: Float = 0
        query.withUnsafeBufferPointer { ptr in
            vDSP_svesq(ptr.baseAddress!, 1, &queryNormSquared, vDSP_Length(query.count))
        }
        let queryNorm = sqrt(queryNormSquared)

        return zip(chunks, recencyWeights).map { chunk, recency in
            var dot: Float = 0
            var chunkNormSquared: Float = 0

            query.withUnsafeBufferPointer { queryPtr in
                chunk.withUnsafeBufferPointer { chunkPtr in
                    vDSP_dotpr(queryPtr.baseAddress!, 1, chunkPtr.baseAddress!, 1, &dot, vDSP_Length(query.count))
                    vDSP_svesq(chunkPtr.baseAddress!, 1, &chunkNormSquared, vDSP_Length(chunk.count))
                }
            }

            let chunkNorm = sqrt(chunkNormSquared)
            let denom = queryNorm * chunkNorm
            let cosine: Float = denom > 0 ? dot / denom : 0
            return cosine * relevanceWeight + recency * recencyWeight
        }
    }

    public static func recencyWeights(
        timestamps: [Date],
        currentTime: Date,
        halfLife: TimeInterval
    ) -> [Float] {
        guard !timestamps.isEmpty else {
            return []
        }
        precondition(halfLife > 0, "halfLife must be positive")

        let now = Float(currentTime.timeIntervalSince1970)
        let halfLifeF = Float(halfLife)
        let ln2: Float = 0.693147

        var exponents = timestamps.map { timestamp -> Float in
            let age = now - Float(timestamp.timeIntervalSince1970)
            return -ln2 * age / halfLifeF
        }

        var values = [Float](repeating: 0, count: exponents.count)
        var count = Int32(exponents.count)
        vvexpf(&values, &exponents, &count)

        return values.map { min(max($0, 0), 1) }
    }

    public static func centrality(embeddings: [[Float]]) -> [Float] {
        guard !embeddings.isEmpty else {
            return []
        }

        let n = embeddings.count
        if n == 1 {
            return [0]
        }

        return (0..<n).map { i in
            var sum: Float = 0
            for j in 0..<n where j != i {
                sum += cosineSimilarity(embeddings[i], embeddings[j])
            }
            return sum / Float(n - 1)
        }
    }

    public static func crossAttentionScores(
        query: [Float],
        embeddings: [[Float]],
        centrality: [Float],
        relevanceWeight: Float,
        centralityWeight: Float
    ) -> [Float] {
        guard !embeddings.isEmpty else {
            return []
        }
        precondition(embeddings.count == centrality.count, "Embedding and centrality count mismatch")

        return embeddings.enumerated().map { index, embedding in
            let relevance = cosineSimilarity(query, embedding)
            return relevance * relevanceWeight + centrality[index] * centralityWeight
        }
    }

    public static func sentenceImportance(
        sentenceEmbeddings: [[Float]],
        chunkEmbedding: [Float]
    ) -> [Float] {
        sentenceEmbeddings.map { sentence in
            cosineSimilarity(sentence, chunkEmbedding)
        }
    }

    public static func pairwiseSimilarity(embeddings: [[Float]]) -> [[Float]] {
        guard !embeddings.isEmpty else {
            return []
        }
        let n = embeddings.count
        if n == 1 {
            return [[1.0]]
        }

        let dim = embeddings[0].count
        precondition(embeddings.allSatisfy { $0.count == dim }, "Embedding dimension mismatch")

        var matrix = Array(repeating: Array(repeating: Float.zero, count: n), count: n)
        for i in 0..<n {
            matrix[i][i] = 1.0
            guard i + 1 < n else { continue }
            for j in (i + 1)..<n {
                let value = cosineSimilarity(embeddings[i], embeddings[j])
                matrix[i][j] = value
                matrix[j][i] = value
            }
        }
        return matrix
    }

    public static func findMergeCandidates(
        similarities: [[Float]],
        threshold: Float
    ) -> [(Int, Int)] {
        guard !similarities.isEmpty else {
            return []
        }
        let n = similarities.count
        precondition(similarities.allSatisfy { $0.count == n }, "Similarity matrix must be square")

        var pairs: [(Int, Int)] = []
        for i in 0..<n {
            guard i + 1 < n else { continue }
            for j in (i + 1)..<n where similarities[i][j] > threshold {
                pairs.append((i, j))
            }
        }
        return pairs
    }

    public static func antipodalFraction(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count, "Vectors must match dimensions")
        guard !a.isEmpty else {
            return 0
        }

        var signDiffCount = 0
        for idx in a.indices where (a[idx] >= 0) != (b[idx] >= 0) {
            signDiffCount += 1
        }

        return Float(signDiffCount) / Float(a.count)
    }

    private static func cosineSimilarity(_ lhs: [Float], _ rhs: [Float]) -> Float {
        guard lhs.count == rhs.count else {
            return 0
        }

        var dot: Float = 0
        var lhsNormSquared: Float = 0
        var rhsNormSquared: Float = 0

        lhs.withUnsafeBufferPointer { lhsPtr in
            rhs.withUnsafeBufferPointer { rhsPtr in
                vDSP_dotpr(lhsPtr.baseAddress!, 1, rhsPtr.baseAddress!, 1, &dot, vDSP_Length(lhs.count))
                vDSP_svesq(lhsPtr.baseAddress!, 1, &lhsNormSquared, vDSP_Length(lhs.count))
                vDSP_svesq(rhsPtr.baseAddress!, 1, &rhsNormSquared, vDSP_Length(rhs.count))
            }
        }

        let denom = sqrt(lhsNormSquared) * sqrt(rhsNormSquared)
        guard denom > 0 else {
            return 0
        }
        return dot / denom
    }
}
