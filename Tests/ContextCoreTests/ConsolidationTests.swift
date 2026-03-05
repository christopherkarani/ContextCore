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
        let dataset = makeNearDuplicateDataset(uniqueCount: 30, pairCount: 6, dim: 384)
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
            embeddings: TestHelpers.randomVectors(n: 50, dim: 384, seed: 8_005),
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
        let dataset = makeNearDuplicateDataset(uniqueCount: 30, pairCount: 6, dim: 384)
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

    @Test("Consolidate promotes duplicate facts into semantic store")
    func consolidatePromotesFacts() async throws {
        let dataset = makeNearDuplicateDataset(uniqueCount: 30, pairCount: 6, dim: 384)
        let episodicStore = EpisodicStore()
        let semanticStore = SemanticStore()
        try await insert(turns: dataset.turns, into: episodicStore)

        let engine = try ConsolidationEngine(embeddingProvider: StubEmbeddingProvider())
        _ = try await engine.consolidate(
            session: UUID(),
            episodicStore: episodicStore,
            semanticStore: semanticStore,
            threshold: 0.92
        )

        #expect(await semanticStore.count >= 4)
    }

    @Test("Consolidate can shrink episodic store when low-retention chunks exist")
    func consolidateShrinksEpisodicStore() async throws {
        let dataset = makeNearDuplicateDataset(uniqueCount: 30, pairCount: 6, dim: 384)
        let episodicStore = EpisodicStore()
        let semanticStore = SemanticStore()
        try await insert(turns: dataset.turns, into: episodicStore)

        for pair in dataset.expectedPairs {
            try await episodicStore.updateRetentionScore(id: pair.first, delta: -0.4)
            try await episodicStore.updateRetentionScore(id: pair.second, delta: -0.4)
        }

        let initial = await episodicStore.count
        let engine = try ConsolidationEngine(embeddingProvider: StubEmbeddingProvider())
        _ = try await engine.consolidate(
            session: UUID(),
            episodicStore: episodicStore,
            semanticStore: semanticStore,
            threshold: 0.92
        )

        #expect(await episodicStore.count < initial)
    }

    @Test("Consolidate promotes the shorter duplicate chunk")
    func consolidatePromotesShorterChunk() async throws {
        let episodicStore = EpisodicStore()
        let semanticStore = SemanticStore()

        let base = TestHelpers.randomVector(dim: 384, seed: 8_501)
        let near = makeNearDuplicate(of: base, targetCosine: 0.96, seed: 8_502)

        let shortTurn = Turn(
            id: uuid(9_001),
            role: .assistant,
            content: "short fact",
            timestamp: Date(timeIntervalSince1970: 1_700_020_000),
            embedding: base
        )
        let longTurn = Turn(
            id: uuid(9_002),
            role: .assistant,
            content: "short fact with a much longer elaboration that should not be promoted first",
            timestamp: Date(timeIntervalSince1970: 1_700_020_001),
            embedding: near
        )
        try await insert(turns: [shortTurn, longTurn], into: episodicStore)

        let engine = try ConsolidationEngine(embeddingProvider: StubEmbeddingProvider())
        _ = try await engine.consolidate(
            session: UUID(),
            episodicStore: episodicStore,
            semanticStore: semanticStore
        )

        let semanticChunks = await semanticStore.allChunks()
        #expect(semanticChunks.contains(where: { $0.content == shortTurn.content }))
    }

    @Test("Consolidate decrements retention scores for duplicate originals")
    func consolidateRetentionDecrement() async throws {
        let episodicStore = EpisodicStore()
        let semanticStore = SemanticStore()

        let base = TestHelpers.randomVector(dim: 384, seed: 8_601)
        let near = makeNearDuplicate(of: base, targetCosine: 0.96, seed: 8_602)
        let turns = makeTurns(embeddings: [base, near], contentPrefix: "retain")
        try await insert(turns: turns, into: episodicStore)

        let engine = try ConsolidationEngine(embeddingProvider: StubEmbeddingProvider())
        _ = try await engine.consolidate(
            session: UUID(),
            episodicStore: episodicStore,
            semanticStore: semanticStore
        )

        let chunks = await episodicStore.allChunks()
        let byID = Dictionary(uniqueKeysWithValues: chunks.map { ($0.id, $0.retentionScore) })
        #expect(abs((byID[turns[0].id] ?? -1) - 0.3) < 1e-6)
        #expect(abs((byID[turns[1].id] ?? -1) - 0.3) < 1e-6)
    }

    @Test("Consolidate evicts chunks that fall below retention threshold")
    func consolidateEvictionAtLowRetention() async throws {
        let episodicStore = EpisodicStore()
        let semanticStore = SemanticStore()

        let base = TestHelpers.randomVector(dim: 384, seed: 8_701)
        let near = makeNearDuplicate(of: base, targetCosine: 0.96, seed: 8_702)
        let turns = makeTurns(embeddings: [base, near], contentPrefix: "evict-low")
        try await insert(turns: turns, into: episodicStore)

        try await episodicStore.updateRetentionScore(id: turns[0].id, delta: -0.35)
        try await episodicStore.updateRetentionScore(id: turns[1].id, delta: -0.35)

        let engine = try ConsolidationEngine(embeddingProvider: StubEmbeddingProvider())
        _ = try await engine.consolidate(
            session: UUID(),
            episodicStore: episodicStore,
            semanticStore: semanticStore
        )

        let remaining = await episodicStore.allChunks()
        #expect(remaining.isEmpty)
    }

    @Test("Consolidate is idempotent across repeated runs")
    func consolidateIdempotent() async throws {
        let dataset = makeNearDuplicateDataset(uniqueCount: 10, pairCount: 3, dim: 384)
        let episodicStore = EpisodicStore()
        let semanticStore = SemanticStore()
        try await insert(turns: dataset.turns, into: episodicStore)

        let engine = try ConsolidationEngine(embeddingProvider: StubEmbeddingProvider())
        _ = try await engine.consolidate(
            session: UUID(),
            episodicStore: episodicStore,
            semanticStore: semanticStore
        )
        let second = try await engine.consolidate(
            session: UUID(),
            episodicStore: episodicStore,
            semanticStore: semanticStore
        )

        #expect(second.duplicatePairsFound == 0)
    }

    @Test("Consolidate handles empty stores")
    func consolidateEmptyStore() async throws {
        let episodicStore = EpisodicStore()
        let semanticStore = SemanticStore()
        let engine = try ConsolidationEngine(embeddingProvider: StubEmbeddingProvider())

        let result = try await engine.consolidate(
            session: UUID(),
            episodicStore: episodicStore,
            semanticStore: semanticStore
        )

        #expect(result.duplicatePairsFound == 0)
        #expect(result.factsPromoted == 0)
        #expect(result.chunksEvicted == 0)
    }

    @Test("Scheduler triggers on count threshold")
    func schedulerCountThreshold() async throws {
        let episodicStore = EpisodicStore()
        let semanticStore = SemanticStore()
        let scheduler = ConsolidationScheduler(
            engine: try ConsolidationEngine(embeddingProvider: StubEmbeddingProvider()),
            countThreshold: 20,
            insertionThreshold: 500
        )

        let dataset = makeNearDuplicateDataset(uniqueCount: 22, pairCount: 1, dim: 384)
        try await insert(turns: dataset.turns, into: episodicStore)
        await scheduler.notifyInsertion(
            episodicCount: await episodicStore.count,
            session: UUID(),
            episodicStore: episodicStore,
            semanticStore: semanticStore
        )

        try await waitForSchedulerToFinish(scheduler)
        #expect(await scheduler.triggerCount() == 1)
        #expect((await scheduler.lastResult()) != nil)
    }

    @Test("Scheduler triggers on insertion threshold")
    func schedulerInsertionThreshold() async throws {
        let episodicStore = EpisodicStore()
        let semanticStore = SemanticStore()
        let engine = try ConsolidationEngine(embeddingProvider: StubEmbeddingProvider())
        let scheduler = ConsolidationScheduler(
            engine: engine,
            countThreshold: 200,
            insertionThreshold: 50
        )

        let dataset = makeNearDuplicateDataset(uniqueCount: 45, pairCount: 3, dim: 384)
        try await insert(turns: dataset.turns, into: episodicStore)

        for _ in 0..<51 {
            await scheduler.notifyInsertion(
                episodicCount: await episodicStore.count,
                session: UUID(),
                episodicStore: episodicStore,
                semanticStore: semanticStore
            )
        }

        try await waitForSchedulerToFinish(scheduler)
        #expect(await scheduler.triggerCount() == 1)
    }

    @Test("Scheduler does not double trigger while running")
    func schedulerNoDoubleTrigger() async throws {
        let episodicStore = EpisodicStore()
        let semanticStore = SemanticStore()
        let engine = try ConsolidationEngine(embeddingProvider: StubEmbeddingProvider())
        let scheduler = ConsolidationScheduler(
            engine: engine,
            countThreshold: 1,
            insertionThreshold: 1
        )

        let dataset = makeNearDuplicateDataset(uniqueCount: 40, pairCount: 4, dim: 384)
        try await insert(turns: dataset.turns, into: episodicStore)

        async let first: Void = scheduler.notifyInsertion(
            episodicCount: await episodicStore.count,
            session: UUID(),
            episodicStore: episodicStore,
            semanticStore: semanticStore
        )
        async let second: Void = scheduler.notifyInsertion(
            episodicCount: await episodicStore.count,
            session: UUID(),
            episodicStore: episodicStore,
            semanticStore: semanticStore
        )
        _ = await (first, second)

        try await waitForSchedulerToFinish(scheduler)
        #expect(await scheduler.triggerCount() == 1)
    }

    @Test("Scheduler notifyInsertion is non-blocking")
    func schedulerNonBlocking() async throws {
        let episodicStore = EpisodicStore()
        let semanticStore = SemanticStore()
        let engine = try ConsolidationEngine(embeddingProvider: StubEmbeddingProvider())
        let scheduler = ConsolidationScheduler(engine: engine, countThreshold: 1, insertionThreshold: 1)

        let dataset = makeNearDuplicateDataset(uniqueCount: 30, pairCount: 2, dim: 384)
        try await insert(turns: dataset.turns, into: episodicStore)

        let clock = ContinuousClock()
        let elapsed = await clock.measure {
            await scheduler.notifyInsertion(
                episodicCount: await episodicStore.count,
                session: UUID(),
                episodicStore: episodicStore,
                semanticStore: semanticStore
            )
        }

        #expect(elapsed < .milliseconds(10))
    }

    @Test("ConsolidationResult reports expected fields")
    func consolidationResultFields() async throws {
        let episodicStore = EpisodicStore()
        let semanticStore = SemanticStore()
        let base = TestHelpers.randomVector(dim: 384, seed: 8_801)
        let near = makeNearDuplicate(of: base, targetCosine: 0.96, seed: 8_802)
        let turns = makeTurns(embeddings: [base, near], contentPrefix: "result")
        try await insert(turns: turns, into: episodicStore)

        try await episodicStore.updateRetentionScore(id: turns[0].id, delta: -0.45)
        try await episodicStore.updateRetentionScore(id: turns[1].id, delta: -0.45)

        let engine = try ConsolidationEngine(embeddingProvider: StubEmbeddingProvider())
        let result = try await engine.consolidate(
            session: UUID(),
            episodicStore: episodicStore,
            semanticStore: semanticStore
        )

        #expect(result.duplicatePairsFound == 1)
        #expect(result.factsPromoted == 1)
        #expect(result.chunksEvicted == 2)
        #expect(result.durationMs >= 0)
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

    private func waitForSchedulerToFinish(
        _ scheduler: ConsolidationScheduler,
        timeoutSeconds: Double = 5
    ) async throws {
        let deadline = Date().addingTimeInterval(timeoutSeconds)
        while await scheduler.isRunning(), Date() < deadline {
            try await Task.sleep(for: .milliseconds(20))
        }
        #expect(!(await scheduler.isRunning()))
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
