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
}
