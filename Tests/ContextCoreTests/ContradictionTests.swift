import Foundation
import Testing
@testable import ContextCore

@Suite("Contradiction Tests")
struct ContradictionTests {
    @Test("Antipodal fraction for exact negation is one")
    func exactNegation() {
        let vector = TestHelpers.randomVector(dim: 384, seed: 9_001)
        let negated = vector.map { -$0 }
        let fraction = CPUReference.antipodalFraction(vector, negated)
        #expect(abs(fraction - 1.0) < 1e-6)
    }

    @Test("Antipodal fraction for identical vectors is zero")
    func identicalEmbeddings() {
        let vector = TestHelpers.randomVector(dim: 384, seed: 9_002)
        let fraction = CPUReference.antipodalFraction(vector, vector)
        #expect(abs(fraction) < 1e-6)
    }

    @Test("Antipodal fraction for partial sign flip is near half")
    func partialNegation() {
        let vector = TestHelpers.randomVector(dim: 384, seed: 9_003)
        var partial = vector
        for idx in stride(from: 0, to: partial.count, by: 2) {
            partial[idx] = -partial[idx]
        }
        let fraction = CPUReference.antipodalFraction(vector, partial)
        #expect(abs(fraction - 0.5) < 0.01)
    }

#if !targetEnvironment(simulator)
    @Test("GPU antipodal fractions match CPU reference")
    func gpuVsCpuParity() async throws {
        var embeddingsA: [[Float]] = []
        var embeddingsB: [[Float]] = []
        for idx in 0..<50 {
            let a = TestHelpers.randomVector(dim: 384, seed: UInt64(9_100 + idx))
            let b = idx % 2 == 0 ? a.map { -$0 } : a
            embeddingsA.append(a)
            embeddingsB.append(b)
        }

        let cpu = zip(embeddingsA, embeddingsB).map { CPUReference.antipodalFraction($0, $1) }
        let engine = try ConsolidationEngine(embeddingProvider: StubEmbeddingProvider())
        let gpu = try await engine.antipodalFractions(embeddingsA: embeddingsA, embeddingsB: embeddingsB)

        let maxError = TestHelpers.maxAbsError(cpu, gpu)
        #expect(maxError < 1e-6)
    }

    @Test("Contradiction candidates include crafted contradictory pair")
    func contradictionCandidatesFound() async throws {
        let semanticStore = SemanticStore()
        let pair = makeContradictoryPair(dim: 384, seed: 9_200)
        let similarity = cosine(pair.0, pair.1)
        let antipodal = CPUReference.antipodalFraction(pair.0, pair.1)

        #expect(similarity > 0.75)
        #expect(antipodal > 0.30)

        try await semanticStore.insert(content: "user prefers dark mode", embedding: pair.0)
        try await semanticStore.insert(content: "user switched to light mode", embedding: pair.1)

        for idx in 0..<18 {
            let embedding = TestHelpers.randomVector(dim: 384, seed: UInt64(9_300 + idx))
            try await semanticStore.insert(content: "fact-\(idx)", embedding: embedding)
        }

        let engine = try ConsolidationEngine(embeddingProvider: StubEmbeddingProvider())
        let candidates = try await engine.contradictionCandidates(in: semanticStore)

        let contents = Set(candidates.flatMap { [$0.0.content, $0.1.content] })
        #expect(contents.contains("user prefers dark mode"))
        #expect(contents.contains("user switched to light mode"))
    }

    @Test("Contradiction candidates are empty when facts are consistent")
    func contradictionCandidatesNone() async throws {
        let semanticStore = SemanticStore()
        for idx in 0..<20 {
            let embedding = TestHelpers.randomVector(dim: 384, seed: UInt64(9_400 + idx))
            try await semanticStore.insert(content: "consistent-\(idx)", embedding: embedding)
        }

        let engine = try ConsolidationEngine(embeddingProvider: StubEmbeddingProvider())
        let candidates = try await engine.contradictionCandidates(in: semanticStore)
        #expect(candidates.isEmpty)
    }

    @Test("Similarity gate excludes low-similarity antipodal pairs")
    func similarityFilter() async throws {
        let semanticStore = SemanticStore()
        let a = TestHelpers.randomVector(dim: 384, seed: 9_500)
        let b = makeLowSimilarityHighAntipodalPair(from: a, seed: 9_501)

        #expect(cosine(a, b) < 0.75)
        #expect(CPUReference.antipodalFraction(a, b) > 0.30)

        try await semanticStore.insert(content: "pair-a", embedding: a)
        try await semanticStore.insert(content: "pair-b", embedding: b)

        let engine = try ConsolidationEngine(embeddingProvider: StubEmbeddingProvider())
        let candidates = try await engine.contradictionCandidates(in: semanticStore)
        #expect(candidates.isEmpty)
    }

    @Test("Antipodal gate excludes low-antipodal high-similarity pairs")
    func antipodalFilter() async throws {
        let semanticStore = SemanticStore()
        let a = TestHelpers.randomVector(dim: 384, seed: 9_600)
        let b = makeHighSimilarityLowAntipodalPair(from: a, seed: 9_601)

        #expect(cosine(a, b) > 0.75)
        #expect(CPUReference.antipodalFraction(a, b) < 0.30)

        try await semanticStore.insert(content: "pair-c", embedding: a)
        try await semanticStore.insert(content: "pair-d", embedding: b)

        let engine = try ConsolidationEngine(embeddingProvider: StubEmbeddingProvider())
        let candidates = try await engine.contradictionCandidates(in: semanticStore)
        #expect(candidates.isEmpty)
    }
#endif

    @Test("Contradiction candidates handles empty store")
    func emptyStore() async throws {
        let semanticStore = SemanticStore()
        let engine = try ConsolidationEngine(embeddingProvider: StubEmbeddingProvider())
        let candidates = try await engine.contradictionCandidates(in: semanticStore)
        #expect(candidates.isEmpty)
    }

    @Test("Contradiction candidates handles single fact")
    func singleFact() async throws {
        let semanticStore = SemanticStore()
        try await semanticStore.insert(
            content: "single",
            embedding: TestHelpers.randomVector(dim: 384, seed: 9_700)
        )
        let engine = try ConsolidationEngine(embeddingProvider: StubEmbeddingProvider())
        let candidates = try await engine.contradictionCandidates(in: semanticStore)
        #expect(candidates.isEmpty)
    }

    private func makeContradictoryPair(dim: Int, seed: UInt64) -> ([Float], [Float]) {
        var base = [Float](repeating: 0, count: dim)
        var rng = TestHelpers.SeededGenerator(seed: seed)

        for idx in 0..<dim {
            let raw = Float(rng.next() & 0xFFFF) / Float(0xFFFF)
            let centered = raw * 2 - 1
            base[idx] = idx < 64 ? centered : centered * 0.03
        }

        base = TestHelpers.l2Normalize(base)
        var contra = base
        for idx in 64..<(64 + 160) where idx < contra.count {
            contra[idx] = -contra[idx]
        }
        contra = TestHelpers.l2Normalize(contra)

        return (base, contra)
    }

    private func makeLowSimilarityHighAntipodalPair(from base: [Float], seed: UInt64) -> [Float] {
        var orth = TestHelpers.randomVector(dim: base.count, seed: seed)
        let dot = zip(orth, base).reduce(Float.zero) { $0 + ($1.0 * $1.1) }
        orth = zip(orth, base).map { $0.0 - dot * $0.1 }
        orth = TestHelpers.l2Normalize(orth)

        var lowSim = zip(base, orth).map { (0.4 * $0.0) + (0.9165 * $0.1) }
        lowSim = TestHelpers.l2Normalize(lowSim)
        for idx in stride(from: 0, to: lowSim.count, by: 3) {
            lowSim[idx] = -lowSim[idx]
        }
        return TestHelpers.l2Normalize(lowSim)
    }

    private func makeHighSimilarityLowAntipodalPair(from base: [Float], seed: UInt64) -> [Float] {
        var orth = TestHelpers.randomVector(dim: base.count, seed: seed)
        let dot = zip(orth, base).reduce(Float.zero) { $0 + ($1.0 * $1.1) }
        orth = zip(orth, base).map { $0.0 - dot * $0.1 }
        orth = TestHelpers.l2Normalize(orth)

        var highSim = zip(base, orth).map { (0.92 * $0.0) + (0.392 * $0.1) }
        highSim = TestHelpers.l2Normalize(highSim)
        for idx in stride(from: 0, to: min(40, highSim.count), by: 2) {
            highSim[idx] = -highSim[idx]
        }
        return TestHelpers.l2Normalize(highSim)
    }

    private func cosine(_ a: [Float], _ b: [Float]) -> Float {
        var dot: Float = 0
        var normA: Float = 0
        var normB: Float = 0
        for idx in a.indices {
            dot += a[idx] * b[idx]
            normA += a[idx] * a[idx]
            normB += b[idx] * b[idx]
        }
        return dot / (normA.squareRoot() * normB.squareRoot())
    }
}

private struct StubEmbeddingProvider: EmbeddingProvider {
    let dimension: Int = 384

    func embed(_ text: String) async throws -> [Float] {
        TestHelpers.randomVector(dim: dimension, seed: stableSeed(text))
    }

    func embedBatch(_ texts: [String]) async throws -> [[Float]] {
        texts.map { TestHelpers.randomVector(dim: dimension, seed: stableSeed($0)) }
    }

    private func stableSeed(_ value: String) -> UInt64 {
        value.utf8.reduce(UInt64(1469598103934665603)) { partial, byte in
            let mixed = partial ^ UInt64(byte)
            return mixed &* 1099511628211
        }
    }
}
