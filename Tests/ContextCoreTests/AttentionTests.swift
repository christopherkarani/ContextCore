import Foundation
import Testing
@testable import ContextCore

#if !targetEnvironment(simulator)
@Suite("Attention Tests")
struct AttentionTests {
    @Test("Centrality identifies outlier as lowest")
    func centralityOutlier() async throws {
        let dim = 384
        let base = TestHelpers.randomVector(dim: dim, seed: 6001)
        let outlier = makeOrthogonalVector(to: base, seed: 6002)

        var embeddings = (0..<9).map { index in
            perturb(base: base, seed: UInt64(7000 + index), magnitude: 0.02)
        }
        embeddings.append(outlier)

        let engine = try AttentionEngine()
        let scores = try await engine.computeCentrality(embeddings: embeddings)

        let minIndex = scores.enumerated().min(by: { $0.element < $1.element })?.offset
        #expect(minIndex == 9)
    }

    @Test("Centrality for identical embeddings is ~1")
    func centralityUniform() async throws {
        let embedding = TestHelpers.randomVector(dim: 384, seed: 6101)
        let embeddings = Array(repeating: embedding, count: 10)

        let engine = try AttentionEngine()
        let scores = try await engine.computeCentrality(embeddings: embeddings)

        #expect(scores.count == 10)
        #expect(scores.allSatisfy { abs($0 - 1.0) < 1e-4 })
    }

    @Test("Centrality for single chunk is zero")
    func centralitySingle() async throws {
        let embedding = TestHelpers.randomVector(dim: 384, seed: 6201)

        let engine = try AttentionEngine()
        let scores = try await engine.computeCentrality(embeddings: [embedding])

        #expect(scores == [0.0])
    }

    @Test("Relevant and central chunks score highest")
    func relevantAndCentralChunks() async throws {
        let data = semanticLikeDataset()

        let engine = try AttentionEngine()
        let scored = try await engine.scoreWindowForEviction(
            taskQuery: data.query,
            windowChunks: data.chunks,
            relevanceWeight: 0.6,
            centralityWeight: 0.4
        )

        let topThree = Array(scored.suffix(3)).map(\.chunk.content)
        #expect(topThree.allSatisfy { $0.contains("Swift") })
    }

    @Test("Weight changes alter scores")
    func weightSensitivity() async throws {
        let data = semanticLikeDataset()
        let engine = try AttentionEngine()

        let first = try await engine.scoreWindowForEviction(
            taskQuery: data.query,
            windowChunks: data.chunks,
            relevanceWeight: 0.8,
            centralityWeight: 0.2
        )

        let second = try await engine.scoreWindowForEviction(
            taskQuery: data.query,
            windowChunks: data.chunks,
            relevanceWeight: 0.6,
            centralityWeight: 0.4
        )

        let firstByID = Dictionary(uniqueKeysWithValues: first.map { ($0.chunk.id, $0.evictionScore) })
        let changed = second.contains { item in
            guard let baseline = firstByID[item.chunk.id] else { return false }
            return abs(item.evictionScore - baseline) > 1e-6
        }

        #expect(changed)
    }

    @Test("GPU vs CPU parity for centrality")
    func centralityParity() async throws {
        let embeddings = TestHelpers.randomVectors(n: 50, dim: 384, seed: 6301)

        let cpu = CPUReference.centrality(embeddings: embeddings)

        let engine = try AttentionEngine()
        let gpu = try await engine.computeCentrality(embeddings: embeddings)

        let maxError = TestHelpers.maxAbsError(cpu, gpu)
        #expect(maxError < 1e-4)
    }

    @Test("GPU vs CPU parity for cross-attention")
    func crossAttentionParity() async throws {
        let query = TestHelpers.randomVector(dim: 384, seed: 6401)
        let embeddings = TestHelpers.randomVectors(n: 50, dim: 384, seed: 6402)
        let chunks = embeddings.enumerated().map { makeChunk(content: "chunk-\($0.offset)", embedding: $0.element, index: $0.offset) }

        let centrality = CPUReference.centrality(embeddings: embeddings)
        let cpuScores = CPUReference.crossAttentionScores(
            query: query,
            embeddings: embeddings,
            centrality: centrality,
            relevanceWeight: 0.6,
            centralityWeight: 0.4
        )

        let cpuByID = Dictionary(uniqueKeysWithValues: chunks.enumerated().map { ($0.element.id, cpuScores[$0.offset]) })

        let engine = try AttentionEngine()
        let gpu = try await engine.scoreWindowForEviction(
            taskQuery: query,
            windowChunks: chunks,
            relevanceWeight: 0.6,
            centralityWeight: 0.4
        )

        let gpuScores = gpu.map(\.evictionScore)
        let expected = gpu.compactMap { cpuByID[$0.chunk.id] }
        let maxError = TestHelpers.maxAbsError(gpuScores, expected)
        #expect(maxError < 1e-4)
    }

    @Test("Eviction scores are sorted ascending")
    func evictionOrdering() async throws {
        let data = semanticLikeDataset()

        let engine = try AttentionEngine()
        let scored = try await engine.scoreWindowForEviction(
            taskQuery: data.query,
            windowChunks: data.chunks
        )

        for index in 0..<(scored.count - 1) {
            #expect(scored[index].evictionScore <= scored[index + 1].evictionScore)
        }
    }

    private func semanticLikeDataset() -> (query: [Float], chunks: [MemoryChunk]) {
        let dim = 384
        let swiftBase = TestHelpers.randomVector(dim: dim, seed: 6501)
        let cookingBase = makeOrthogonalVector(to: swiftBase, seed: 6502)
        let fillerBase = makeOrthogonalVector(to: cookingBase, seed: 6503)

        let query = perturb(base: swiftBase, seed: 6504, magnitude: 0.01)

        let swiftChunks = [
            makeChunk(content: "Swift actors provide data race safety.", embedding: perturb(base: swiftBase, seed: 6510, magnitude: 0.01), index: 0),
            makeChunk(content: "Swift async await simplifies concurrency.", embedding: perturb(base: swiftBase, seed: 6511, magnitude: 0.01), index: 1),
            makeChunk(content: "Swift Sendable catches race bugs.", embedding: perturb(base: swiftBase, seed: 6512, magnitude: 0.01), index: 2),
        ]

        let cookingChunks = [
            makeChunk(content: "Preheat oven before baking.", embedding: perturb(base: cookingBase, seed: 6520, magnitude: 0.01), index: 3),
            makeChunk(content: "Whisk eggs until fluffy.", embedding: perturb(base: cookingBase, seed: 6521, magnitude: 0.01), index: 4),
        ]

        let fillerChunks = [
            makeChunk(content: "Generic placeholder text one.", embedding: perturb(base: fillerBase, seed: 6530, magnitude: 0.01), index: 5),
            makeChunk(content: "Generic placeholder text two.", embedding: perturb(base: fillerBase, seed: 6531, magnitude: 0.01), index: 6),
        ]

        return (query: query, chunks: swiftChunks + cookingChunks + fillerChunks)
    }

    private func perturb(base: [Float], seed: UInt64, magnitude: Float) -> [Float] {
        var rng = TestHelpers.SeededGenerator(seed: seed)
        let noisy = base.map { value -> Float in
            let delta = (Float(rng.next() & 0xFFFF) / Float(0xFFFF)) * 2 - 1
            return value + (delta * magnitude)
        }
        return TestHelpers.l2Normalize(noisy)
    }

    private func makeOrthogonalVector(to base: [Float], seed: UInt64) -> [Float] {
        var candidate = TestHelpers.randomVector(dim: base.count, seed: seed)

        let dot = zip(candidate, base).reduce(Float.zero) { $0 + ($1.0 * $1.1) }
        candidate = zip(candidate, base).map { $0.0 - dot * $0.1 }
        return TestHelpers.l2Normalize(candidate)
    }

    private func makeChunk(content: String, embedding: [Float], index: Int) -> MemoryChunk {
        MemoryChunk(
            id: UUID(uuidString: String(format: "BBBBBBBB-BBBB-BBBB-BBBB-%012d", index)) ?? UUID(),
            content: content,
            embedding: embedding,
            type: .semantic,
            retentionScore: 1.0,
            sourceSessionID: UUID(uuidString: "CCCCCCCC-CCCC-CCCC-CCCC-CCCCCCCCCCCC")!
        )
    }
}
#endif
