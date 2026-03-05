import Foundation
import Testing
@testable import ContextCore

@Suite("Embedding Tests")
struct EmbeddingTests {
    @Test("Caching provider returns identical vector and fast second hit")
    func cachingProviderSecondHitIsFast() async throws {
        let provider = CachingEmbeddingProvider(base: MockEmbeddingProvider())

        let first = try await provider.embed("hello world")
        var second: [Float] = []
        let clock = ContinuousClock()
        let elapsed = try await clock.measure {
            second = try await provider.embed("hello world")
        }

        #expect(bitwiseEqual(first, second))
        #expect(elapsed < .milliseconds(1))
    }

    @Test("embedBatch returns unique 384-dimensional vectors")
    func embedBatchReturnsExpectedShape() async throws {
        let provider = CachingEmbeddingProvider(base: MockEmbeddingProvider())
        let texts = (0..<10).map { "text-\($0)" }

        let vectors = try await provider.embedBatch(texts)

        #expect(vectors.count == 10)
        #expect(vectors.allSatisfy { $0.count == 384 })

        let uniqueCount = Set(vectors.map(vectorFingerprint)).count
        #expect(uniqueCount == 10)
    }

    @Test("EmbeddingProvider dimension is 384")
    func embeddingDimensionIs384() {
        let provider = MockEmbeddingProvider()
        #expect(provider.dimension == 384)
    }

    @Test("EmbeddingCache evicts least-recently-used entry at capacity")
    func embeddingCacheEvictsAtCapacity() async {
        let cache = EmbeddingCache(capacity: 512)

        for i in 0..<513 {
            await cache.set("key-\(i)", value: [Float(i)])
        }

        #expect(await cache.count == 512)
        #expect(await cache.get("key-0") == nil)
        #expect(await cache.get("key-512") == [512])
    }

    private func bitwiseEqual(_ lhs: [Float], _ rhs: [Float]) -> Bool {
        guard lhs.count == rhs.count else {
            return false
        }
        for idx in lhs.indices where lhs[idx].bitPattern != rhs[idx].bitPattern {
            return false
        }
        return true
    }

    private func vectorFingerprint(_ vector: [Float]) -> String {
        vector.prefix(8).map { String($0.bitPattern) }.joined(separator: ":")
    }
}

private struct MockEmbeddingProvider: EmbeddingProvider, Sendable {
    let dimension: Int = 384

    func embed(_ text: String) async throws -> [Float] {
        deterministicVector(for: text)
    }

    func embedBatch(_ texts: [String]) async throws -> [[Float]] {
        texts.map(deterministicVector)
    }

    private func deterministicVector(for text: String) -> [Float] {
        var state = stableSeed(from: text)
        var values = [Float](repeating: 0, count: dimension)

        for idx in values.indices {
            state &*= 6364136223846793005
            state &+= 1442695040888963407
            let component = Float(Int64(bitPattern: state & 0x0000_FFFF_FFFF_FFFF) % 10_000) / 5_000.0 - 1.0
            values[idx] = component
        }

        let norm: Float = values.reduce(0) { $0 + ($1 * $1) }.squareRoot()
        if norm > 0 {
            for idx in values.indices {
                values[idx] /= norm
            }
        }
        return values
    }

    private func stableSeed(from text: String) -> UInt64 {
        var hash: UInt64 = 1469598103934665603
        for byte in text.utf8 {
            hash ^= UInt64(byte)
            hash &*= 1099511628211
        }
        return hash
    }
}
