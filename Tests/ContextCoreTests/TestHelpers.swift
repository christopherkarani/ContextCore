import Foundation

enum TestHelpers {
    struct SeededGenerator: RandomNumberGenerator {
        private var state: UInt64

        init(seed: UInt64) {
            self.state = seed
        }

        mutating func next() -> UInt64 {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return state
        }
    }

    static func randomVector(dim: Int, seed: UInt64) -> [Float] {
        var rng = SeededGenerator(seed: seed)
        return randomVector(dim: dim, using: &rng)
    }

    static func randomVectors(n: Int, dim: Int, seed: UInt64) -> [[Float]] {
        var rng = SeededGenerator(seed: seed)
        return (0..<n).map { _ in randomVector(dim: dim, using: &rng) }
    }

    static func maxAbsError(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count, "Array sizes must match")
        var maxValue: Float = 0
        for index in a.indices {
            let delta = abs(a[index] - b[index])
            if delta > maxValue {
                maxValue = delta
            }
        }
        return maxValue
    }

    private static func randomVector(
        dim: Int,
        using rng: inout SeededGenerator
    ) -> [Float] {
        var values = [Float](repeating: 0, count: dim)
        for index in values.indices {
            let raw = Float(rng.next() & 0xFFFF) / Float(0xFFFF)
            values[index] = raw * 2 - 1
        }
        return l2Normalize(values)
    }

    static func l2Normalize(_ values: [Float]) -> [Float] {
        let norm = values.reduce(Float.zero) { $0 + ($1 * $1) }.squareRoot()
        guard norm > 0 else {
            return values
        }
        return values.map { $0 / norm }
    }
}
