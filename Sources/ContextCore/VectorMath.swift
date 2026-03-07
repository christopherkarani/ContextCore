import Accelerate

/// Shared vector math utilities using Accelerate for SIMD performance.
enum VectorMath {
    /// Computes cosine similarity between two equal-length vectors.
    ///
    /// Returns 0 for empty or mismatched vectors, and 0 when either vector
    /// has zero magnitude.
    ///
    /// - Parameters:
    ///   - lhs: First vector.
    ///   - rhs: Second vector.
    /// - Returns: Cosine similarity in `[-1, 1]`, or 0 for degenerate inputs.
    static func cosineSimilarity(_ lhs: [Float], _ rhs: [Float]) -> Float {
        guard lhs.count == rhs.count, !lhs.isEmpty else {
            return 0
        }

        var dot: Float = 0
        var lhsNormSq: Float = 0
        var rhsNormSq: Float = 0

        vDSP_dotpr(lhs, 1, rhs, 1, &dot, vDSP_Length(lhs.count))
        vDSP_dotpr(lhs, 1, lhs, 1, &lhsNormSq, vDSP_Length(lhs.count))
        vDSP_dotpr(rhs, 1, rhs, 1, &rhsNormSq, vDSP_Length(rhs.count))

        let denominator = lhsNormSq.squareRoot() * rhsNormSq.squareRoot()
        guard denominator > 0 else {
            return 0
        }

        return dot / denominator
    }
}
