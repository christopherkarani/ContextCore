import Foundation
import Testing
@testable import ContextCore

@Suite("Scoring Tests")
struct ScoringTests {
    @Test("CPU reference correctness")
    func cpuReferenceCorrectness() {
        let query: [Float] = [1, 0]
        let chunks: [[Float]] = [[1, 0]]
        let recency: [Float] = [0.5]

        let scores = CPUReference.relevanceScores(
            query: query,
            chunks: chunks,
            recencyWeights: recency,
            relevanceWeight: 0.7,
            recencyWeight: 0.3
        )

        #expect(abs(scores[0] - 0.85) < 1e-6)
    }

#if !targetEnvironment(simulator)
    @Test("GPU vs CPU parity")
    func gpuVsCpuParity() async throws {
        let dim = 384
        let n = 500
        let query = TestHelpers.randomVector(dim: dim, seed: 123)
        let embeddings = TestHelpers.randomVectors(n: n, dim: dim, seed: 456)

        var rng = TestHelpers.SeededGenerator(seed: 789)
        let recency = (0..<n).map { _ in Float(rng.next() & 0xFFFF) / Float(0xFFFF) }

        let chunks = makeChunks(from: embeddings)
        let cpuScores = CPUReference.relevanceScores(
            query: query,
            chunks: embeddings,
            recencyWeights: recency,
            relevanceWeight: 0.7,
            recencyWeight: 0.3
        )

        let engine = try ScoringEngine()
        let gpuResult = try await engine.scoreChunks(
            query: query,
            chunks: chunks,
            recencyWeights: recency,
            relevanceWeight: 0.7,
            recencyWeight: 0.3
        )

        var expectedByID: [UUID: Float] = [:]
        for (index, chunk) in chunks.enumerated() {
            expectedByID[chunk.id] = cpuScores[index]
        }

        let gpuScores = gpuResult.map(\.score)
        let expectedScores = gpuResult.compactMap { expectedByID[$0.chunk.id] }

        #expect(gpuScores.count == expectedScores.count)
        let maxError = TestHelpers.maxAbsError(gpuScores, expectedScores)
        #expect(maxError < 1e-4)
    }

    @Test("Top-k correctness")
    func topKCorrectness() async throws {
        var rng = TestHelpers.SeededGenerator(seed: 901)
        let scores = (0..<500).map { _ in Float(rng.next() & 0xFFFF) / Float(0xFFFF) }

        let engine = try ScoringEngine()
        let top = try await engine.topKIndices(scores: scores, k: 10)

        let expected = scores.indices
            .sorted { scores[$0] > scores[$1] }
            .prefix(10)
            .map { $0 }

        #expect(top == expected)
    }

    @Test("Custom weights produce different scores")
    func customWeightsDiffer() async throws {
        let dim = 384
        let n = 64
        let query = TestHelpers.randomVector(dim: dim, seed: 1001)
        let embeddings = TestHelpers.randomVectors(n: n, dim: dim, seed: 1002)
        let recency = (0..<n).map { index in Float(index) / Float(n) }
        let chunks = makeChunks(from: embeddings)

        let engine = try ScoringEngine()

        let baseline = try await engine.scoreChunks(
            query: query,
            chunks: chunks,
            recencyWeights: recency,
            relevanceWeight: 0.7,
            recencyWeight: 0.3
        )

        let custom = try await engine.scoreChunks(
            query: query,
            chunks: chunks,
            recencyWeights: recency,
            relevanceWeight: 0.5,
            recencyWeight: 0.5
        )

        var baselineByID: [UUID: Float] = [:]
        for item in baseline {
            baselineByID[item.chunk.id] = item.score
        }

        let changed = custom.contains { item in
            guard let baselineScore = baselineByID[item.chunk.id] else {
                return false
            }
            return abs(baselineScore - item.score) > 1e-6
        }

        #expect(changed)
    }

    @Test("Single chunk scoring")
    func singleChunk() async throws {
        let query = TestHelpers.randomVector(dim: 384, seed: 2001)
        let embedding = TestHelpers.randomVector(dim: 384, seed: 2002)
        let chunk = makeChunk(embedding: embedding, index: 0)

        let engine = try ScoringEngine()
        let scored = try await engine.scoreChunks(
            query: query,
            chunks: [chunk],
            recencyWeights: [1.0]
        )

        #expect(scored.count == 1)
        #expect(scored[0].chunk.id == chunk.id)
    }

    @Test("Zero recency weights")
    func zeroRecencyWeights() async throws {
        let dim = 384
        let n = 16
        let query = TestHelpers.randomVector(dim: dim, seed: 3001)
        let embeddings = TestHelpers.randomVectors(n: n, dim: dim, seed: 3002)
        let chunks = makeChunks(from: embeddings)
        let recency = [Float](repeating: 0, count: n)

        let cpu = CPUReference.relevanceScores(
            query: query,
            chunks: embeddings,
            recencyWeights: recency,
            relevanceWeight: 0.7,
            recencyWeight: 0.3
        )

        let engine = try ScoringEngine()
        let gpu = try await engine.scoreChunks(
            query: query,
            chunks: chunks,
            recencyWeights: recency,
            relevanceWeight: 0.7,
            recencyWeight: 0.3
        )

        var cpuByID: [UUID: Float] = [:]
        for (index, chunk) in chunks.enumerated() {
            cpuByID[chunk.id] = cpu[index]
        }

        let orderedCPU = gpu.compactMap { cpuByID[$0.chunk.id] }
        let orderedGPU = gpu.map(\.score)
        let maxError = TestHelpers.maxAbsError(orderedGPU, orderedCPU)

        #expect(maxError < 1e-4)
    }

    @Test("Sorted output descending")
    func sortedOutput() async throws {
        let dim = 384
        let n = 32
        let query = TestHelpers.randomVector(dim: dim, seed: 4001)
        let embeddings = TestHelpers.randomVectors(n: n, dim: dim, seed: 4002)
        let chunks = makeChunks(from: embeddings)
        var rng = TestHelpers.SeededGenerator(seed: 4003)
        let recency = (0..<n).map { _ in Float(rng.next() & 0xFFFF) / Float(0xFFFF) }

        let engine = try ScoringEngine()
        let scored = try await engine.scoreChunks(
            query: query,
            chunks: chunks,
            recencyWeights: recency
        )

        #expect(scored.count == n)
        for idx in 0..<(scored.count - 1) {
            #expect(scored[idx].score >= scored[idx + 1].score)
        }
    }
#endif

    private func makeChunks(from embeddings: [[Float]]) -> [MemoryChunk] {
        embeddings.enumerated().map { makeChunk(embedding: $0.element, index: $0.offset) }
    }

    private func makeChunk(embedding: [Float], index: Int) -> MemoryChunk {
        MemoryChunk(
            id: UUID(uuidString: String(format: "00000000-0000-0000-0000-%012d", index)) ?? UUID(),
            content: "chunk-\(index)",
            embedding: embedding,
            type: .semantic,
            retentionScore: 1.0,
            sourceSessionID: UUID(uuidString: "AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA")!
        )
    }
}
