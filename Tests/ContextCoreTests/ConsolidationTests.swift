import Foundation
import Testing
@testable import ContextCore

@Suite("Consolidation Tests")
struct ConsolidationTests {
    @Test("CPU pairwise similarity is symmetric")
    func cpuPairwiseSymmetry() {
        let embeddings = TestHelpers.randomVectors(n: 20, dim: 384, seed: 8_001)
        let similarity = CPUReference.pairwiseSimilarity(embeddings: embeddings)

        for i in 0..<similarity.count {
            for j in 0..<similarity.count {
                #expect(abs(similarity[i][j] - similarity[j][i]) < 1e-6)
            }
        }
    }

    @Test("CPU pairwise similarity diagonal is one")
    func cpuPairwiseDiagonal() {
        let embeddings = TestHelpers.randomVectors(n: 20, dim: 384, seed: 8_002)
        let similarity = CPUReference.pairwiseSimilarity(embeddings: embeddings)

        for i in 0..<similarity.count {
            #expect(abs(similarity[i][i] - 1.0) < 1e-6)
        }
    }

#if !targetEnvironment(simulator)
    @Test("GPU pairwise similarity matches CPU reference")
    func gpuPairwiseParity() async throws {
        let embeddings = TestHelpers.randomVectors(n: 100, dim: 384, seed: 8_003)
        let cpu = CPUReference.pairwiseSimilarity(embeddings: embeddings)

        let engine = try ConsolidationEngine(embeddingProvider: StubEmbeddingProvider())
        let gpu = try await engine.pairwiseSimilarity(embeddings: embeddings)

        var maxAbsError: Float = 0
        for i in 0..<cpu.count {
            for j in (i + 1)..<cpu.count {
                let delta = abs(cpu[i][j] - gpu[i][j])
                if delta > maxAbsError {
                    maxAbsError = delta
                }
            }
        }

        #expect(maxAbsError < 1e-4)
    }

    @Test("Merge candidates include all expected near-duplicate pairs")
    func mergeCandidatesTruePositives() async throws {
        let dataset = makeNearDuplicateDataset(uniqueCount: 90, pairCount: 10, dim: 384)
        let store = EpisodicStore()
        try await insert(turns: dataset.turns, into: store)

        let engine = try ConsolidationEngine(embeddingProvider: StubEmbeddingProvider())
        let duplicates = try await engine.findDuplicates(in: store, threshold: 0.92)
        let found = Set(duplicates.map { UUIDPair($0.0, $0.1) })

        for expected in dataset.expectedPairs {
            #expect(found.contains(expected))
        }
    }

    @Test("Merge candidates have no false positives for unique vectors")
    func mergeCandidatesNoFalsePositives() async throws {
        let turns = makeTurns(
            embeddings: TestHelpers.randomVectors(n: 100, dim: 384, seed: 8_005),
            contentPrefix: "unique"
        )

        let store = EpisodicStore()
        try await insert(turns: turns, into: store)

        let engine = try ConsolidationEngine(embeddingProvider: StubEmbeddingProvider())
        let duplicates = try await engine.findDuplicates(in: store, threshold: 0.92)

        #expect(duplicates.isEmpty)
    }

    @Test("Duplicate detection threshold sensitivity")
    func thresholdSensitivity() async throws {
        let dataset = makeNearDuplicateDataset(uniqueCount: 90, pairCount: 10, dim: 384)
        let store = EpisodicStore()
        try await insert(turns: dataset.turns, into: store)

        let engine = try ConsolidationEngine(embeddingProvider: StubEmbeddingProvider())
        let loose = try await engine.findDuplicates(in: store, threshold: 0.80)
        let baseline = try await engine.findDuplicates(in: store, threshold: 0.92)
        let strict = try await engine.findDuplicates(in: store, threshold: 0.99)

        #expect(loose.count >= baseline.count)
        #expect(baseline.count >= dataset.expectedPairs.count)
        #expect(strict.count <= baseline.count)
    }

    @Test("Duplicate detection handles n equals one")
    func duplicateDetectionSingleChunk() async throws {
        let embeddings = [TestHelpers.randomVector(dim: 384, seed: 8_006)]
        let turns = makeTurns(embeddings: embeddings, contentPrefix: "single")

        let store = EpisodicStore()
        try await insert(turns: turns, into: store)

        let engine = try ConsolidationEngine(embeddingProvider: StubEmbeddingProvider())
        let duplicates = try await engine.findDuplicates(in: store)

        #expect(duplicates.isEmpty)
    }

    @Test("Duplicate detection for two identical chunks")
    func duplicateDetectionTwoIdentical() async throws {
        let embedding = TestHelpers.randomVector(dim: 384, seed: 8_007)
        let turns = makeTurns(
            embeddings: [embedding, embedding],
            contentPrefix: "identical"
        )

        let store = EpisodicStore()
        try await insert(turns: turns, into: store)

        let engine = try ConsolidationEngine(embeddingProvider: StubEmbeddingProvider())
        let duplicates = try await engine.findDuplicates(in: store, threshold: 0.92)

        #expect(duplicates.count == 1)
        #expect(duplicates[0].0 != duplicates[0].1)
    }
#endif

    @Test("EpisodicStore allChunks returns all inserted chunks")
    func episodicAllChunks() async throws {
        let turns = makeTurns(
            embeddings: TestHelpers.randomVectors(n: 15, dim: 64, seed: 8_101),
            contentPrefix: "all"
        )
        let store = EpisodicStore()
        try await insert(turns: turns, into: store)

        let chunks = await store.allChunks()
        #expect(chunks.count == 15)
    }

    @Test("EpisodicStore updateRetentionScore adjusts and clamps")
    func episodicUpdateRetentionScore() async throws {
        let embedding = TestHelpers.randomVector(dim: 64, seed: 8_102)
        var turn = Turn(role: .assistant, content: "retention")
        turn.embedding = embedding

        let store = EpisodicStore()
        try await store.insert(turn: turn)
        try await store.updateRetentionScore(id: turn.id, delta: -0.2)

        let chunks = await store.allChunks()
        let updated = try #require(chunks.first(where: { $0.id == turn.id }))
        #expect(abs(updated.retentionScore - 0.3) < 1e-6)
    }

    @Test("EpisodicStore evict removes chunk from store")
    func episodicEvict() async throws {
        let turns = makeTurns(
            embeddings: TestHelpers.randomVectors(n: 5, dim: 64, seed: 8_103),
            contentPrefix: "evict"
        )
        let store = EpisodicStore()
        try await insert(turns: turns, into: store)

        try await store.evict(id: turns[0].id)

        #expect(await store.count == 4)
        let all = await store.allChunks()
        #expect(!all.contains(where: { $0.id == turns[0].id }))
    }

    private func insert(turns: [Turn], into store: EpisodicStore) async throws {
        for turn in turns {
            try await store.insert(turn: turn)
        }
    }

    private func makeTurns(
        embeddings: [[Float]],
        contentPrefix: String
    ) -> [Turn] {
        embeddings.enumerated().map { index, embedding in
            Turn(
                id: uuid(index + 1_000),
                role: .assistant,
                content: "\(contentPrefix)-\(index)",
                timestamp: Date(timeIntervalSince1970: 1_700_000_000 + Double(index)),
                embedding: embedding
            )
        }
    }

    private func makeNearDuplicateDataset(
        uniqueCount: Int,
        pairCount: Int,
        dim: Int
    ) -> (turns: [Turn], expectedPairs: Set<UUIDPair>) {
        var turns = makeTurns(
            embeddings: TestHelpers.randomVectors(n: uniqueCount, dim: dim, seed: 8_201),
            contentPrefix: "unique"
        )

        var expected = Set<UUIDPair>()
        for pairIndex in 0..<pairCount {
            let base = TestHelpers.randomVector(dim: dim, seed: UInt64(8_300 + pairIndex))
            let near = makeNearDuplicate(of: base, targetCosine: 0.96, seed: UInt64(8_400 + pairIndex))

            let leftID = uuid(2_000 + pairIndex * 2)
            let rightID = uuid(2_001 + pairIndex * 2)

            turns.append(
                Turn(
                    id: leftID,
                    role: .assistant,
                    content: "fact-\(pairIndex)",
                    timestamp: Date(timeIntervalSince1970: 1_700_010_000 + Double(pairIndex * 2)),
                    embedding: base
                )
            )
            turns.append(
                Turn(
                    id: rightID,
                    role: .assistant,
                    content: "fact-\(pairIndex) restated with extra context and details",
                    timestamp: Date(timeIntervalSince1970: 1_700_010_001 + Double(pairIndex * 2)),
                    embedding: near
                )
            )
            expected.insert(UUIDPair(leftID, rightID))
        }

        return (turns: turns, expectedPairs: expected)
    }

    private func makeNearDuplicate(of base: [Float], targetCosine: Float, seed: UInt64) -> [Float] {
        var orth = TestHelpers.randomVector(dim: base.count, seed: seed)
        let dot = zip(orth, base).reduce(Float.zero) { $0 + ($1.0 * $1.1) }
        orth = zip(orth, base).map { $0.0 - dot * $0.1 }
        orth = TestHelpers.l2Normalize(orth)

        let sine = (max(0, 1 - (targetCosine * targetCosine))).squareRoot()
        let blended = zip(base, orth).map { (targetCosine * $0.0) + (sine * $0.1) }
        return TestHelpers.l2Normalize(blended)
    }

    private func uuid(_ value: Int) -> UUID {
        UUID(uuidString: String(format: "00000000-0000-0000-0000-%012d", value)) ?? UUID()
    }
}

private struct UUIDPair: Hashable {
    let first: UUID
    let second: UUID

    init(_ lhs: UUID, _ rhs: UUID) {
        if lhs.uuidString < rhs.uuidString {
            first = lhs
            second = rhs
        } else {
            first = rhs
            second = lhs
        }
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
