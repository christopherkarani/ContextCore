import ContextCore
import Foundation

struct ConsolidationCaseResult: Sendable {
    let chunks: Int
    let metrics: BenchmarkResult
}

private actor BenchmarkEpisodicStore: ConsolidationEpisodicStore {
    private var chunksByID: [UUID: MemoryChunk]
    private var consolidatedIDs: Set<UUID>

    init(chunks: [MemoryChunk]) {
        self.chunksByID = Dictionary(uniqueKeysWithValues: chunks.map { ($0.id, $0) })
        self.consolidatedIDs = []
    }

    var count: Int {
        chunksByID.count
    }

    func allChunks() async -> [MemoryChunk] {
        chunksByID.values.sorted { $0.createdAt < $1.createdAt }
    }

    func updateRetentionScore(id: UUID, delta: Float) async throws {
        guard var chunk = chunksByID[id] else {
            throw ContextCoreError.chunkNotFound(id: id)
        }
        chunk.retentionScore = max(0, min(1, chunk.retentionScore + delta))
        chunksByID[id] = chunk
    }

    func evict(id: UUID) async throws {
        guard chunksByID.removeValue(forKey: id) != nil else {
            throw ContextCoreError.chunkNotFound(id: id)
        }
        consolidatedIDs.remove(id)
    }

    func markConsolidated(id: UUID) async throws {
        guard chunksByID[id] != nil else {
            throw ContextCoreError.chunkNotFound(id: id)
        }
        consolidatedIDs.insert(id)
    }

    func isConsolidated(id: UUID) async -> Bool {
        consolidatedIDs.contains(id)
    }
}

private actor BenchmarkSemanticStore: ConsolidationSemanticStore {
    private var chunks: [MemoryChunk] = []

    var count: Int {
        chunks.count
    }

    func allChunks() async -> [MemoryChunk] {
        chunks
    }

    func upsert(fact: String, embedding: [Float]) async throws {
        if let index = chunks.firstIndex(where: { $0.content == fact }) {
            var chunk = chunks[index]
            chunk.accessCount += 1
            chunk.lastAccessedAt = .now
            chunks[index] = chunk
            return
        }

        chunks.append(
            MemoryChunk(
                content: fact,
                embedding: embedding,
                type: .semantic,
                createdAt: .now,
                lastAccessedAt: .now,
                accessCount: 1,
                retentionScore: 1.0,
                sourceSessionID: UUID(),
                metadata: ["kind": "fact"]
            )
        )
    }
}

private struct ConsolidationRunFixture {
    let sessionID: UUID
    let episodicStore: BenchmarkEpisodicStore
    let semanticStore: BenchmarkSemanticStore
}

func runConsolidationBenchmarks() async throws -> [ConsolidationCaseResult] {
    var results: [ConsolidationCaseResult] = []
    let chunkCounts = benchmarkProfile == .quick ? [100, 500] : [100, 500, 2000]
    let warmup = benchmarkProfile == .quick ? 1 : 3
    let iterations = benchmarkProfile == .quick ? 3 : 10

    for chunkCount in chunkCounts {
        let requiredRuns = warmup + iterations

        let provider = BenchmarkEmbeddingProvider()
        let engine = try ConsolidationEngine(embeddingProvider: provider)
        let baseChunks = try await makeConsolidationChunks(count: chunkCount, provider: provider)

        var fixtures: [ConsolidationRunFixture] = []
        fixtures.reserveCapacity(requiredRuns)

        for _ in 0..<requiredRuns {
            fixtures.append(
                ConsolidationRunFixture(
                    sessionID: UUID(),
                    episodicStore: BenchmarkEpisodicStore(chunks: baseChunks),
                    semanticStore: BenchmarkSemanticStore()
                )
            )
        }

        var fixtureIndex = 0
        let metrics = try await benchmark(
            name: "consolidate(\(chunkCount))",
            warmup: warmup,
            iterations: iterations
        ) {
            let fixture = fixtures[fixtureIndex]
            fixtureIndex += 1
            _ = try await engine.consolidate(
                session: fixture.sessionID,
                episodicStore: fixture.episodicStore,
                semanticStore: fixture.semanticStore,
                threshold: 0.92
            )
        }

        results.append(
            ConsolidationCaseResult(
                chunks: chunkCount,
                metrics: metrics
            )
        )
    }

    return results
}

private func makeConsolidationChunks(
    count: Int,
    provider: BenchmarkEmbeddingProvider
) async throws -> [MemoryChunk] {
    var chunks: [MemoryChunk] = []
    chunks.reserveCapacity(count)

    for index in 0..<count {
        let content = BenchmarkDataFactory.duplicateProneContent(index: index)
        let embedding = try await provider.embed(content)
        chunks.append(
            MemoryChunk(
                content: content,
                embedding: embedding,
                type: .episodic,
                createdAt: Date(timeIntervalSince1970: Double(index)),
                lastAccessedAt: Date(timeIntervalSince1970: Double(index)),
                accessCount: 1,
                retentionScore: 0.8,
                sourceSessionID: UUID(),
                metadata: [:]
            )
        )
    }

    return chunks
}
