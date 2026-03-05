import Foundation
import Testing
@testable import ContextCore

@Suite("Memory Store Tests")
struct MemoryStoreTests {
    @Test("MemoryChunk JSON roundtrip")
    func memoryChunkJSONRoundtrip() throws {
        let chunk = MemoryChunk(
            id: UUID(uuidString: "33333333-3333-3333-3333-333333333333")!,
            content: "fact",
            embedding: [0.1, 0.2, 0.3],
            type: .semantic,
            createdAt: Date(timeIntervalSince1970: 1_700_000_100),
            lastAccessedAt: Date(timeIntervalSince1970: 1_700_000_200),
            accessCount: 4,
            retentionScore: 1.0,
            sourceSessionID: UUID(uuidString: "44444444-4444-4444-4444-444444444444")!,
            metadata: ["source": "test"]
        )

        let data = try JSONEncoder().encode(chunk)
        let decoded = try JSONDecoder().decode(MemoryChunk.self, from: data)

        #expect(decoded == chunk)
    }

    @Test("EpisodicStore insert and retrieve")
    func episodicInsertAndRetrieve() async throws {
        let store = EpisodicStore()

        for i in 0..<10 {
            var turn = Turn(role: .assistant, content: "turn-\(i)")
            turn.embedding = makeVector(row: i, dim: 8)
            try await store.insert(turn: turn)
        }

        let query = makeVector(row: 3, dim: 8)
        let results = try await store.retrieve(query: query, k: 5)

        #expect(!results.isEmpty)
        #expect(results.count <= 5)
    }

    @Test("EpisodicStore count increments")
    func episodicCountIncrements() async throws {
        let store = EpisodicStore()

        for i in 0..<3 {
            var turn = Turn(role: .user, content: "u-\(i)")
            turn.embedding = makeVector(row: i + 100, dim: 8)
            try await store.insert(turn: turn)
        }

        #expect(await store.count == 3)
    }

    @Test("SemanticStore insert and retrieve")
    func semanticInsertAndRetrieve() async throws {
        let store = SemanticStore()

        for i in 0..<10 {
            try await store.insert(
                content: "fact-\(i)",
                embedding: makeVector(row: i, dim: 8),
                metadata: ["kind": "fact"]
            )
        }

        let query = makeVector(row: 6, dim: 8)
        let results = try await store.retrieve(query: query, k: 4)

        #expect(!results.isEmpty)
    }

    @Test("SemanticStore upsert deduplicates by similarity")
    func semanticUpsertDeduplicates() async throws {
        let store = SemanticStore()
        let vector = makeVector(row: 42, dim: 8)

        try await store.upsert(fact: "Use Swift actors for stores", embedding: vector)
        try await store.upsert(fact: "Use Swift actors for stores", embedding: vector)

        #expect(await store.count == 1)

        let results = try await store.retrieve(query: vector, k: 1)
        #expect(results.count == 1)
        #expect(results[0].accessCount == 2)
    }

    @Test("ProceduralStore exact-key retrieval")
    func proceduralExactKeyRetrieval() async {
        let store = ProceduralStore()
        let calls = sampleToolCalls(prefix: "exact")

        for i in 0..<5 {
            await store.record(taskType: "task.\(i)", tools: [calls[i]])
        }

        let retrieved = await store.retrieve(taskType: "task.3")
        #expect(retrieved == [calls[3]])
    }

    @Test("ProceduralStore prefix retrieval")
    func proceduralPrefixRetrieval() async {
        let store = ProceduralStore()
        let formatCall = ToolCall(name: "format", input: "a.swift", output: "ok", durationMs: 3)
        let lintCall = ToolCall(name: "lint", input: "a.swift", output: "ok", durationMs: 4)

        await store.record(taskType: "code.swift.format", tools: [formatCall])
        await store.record(taskType: "code.swift.lint", tools: [lintCall])

        let retrieved = await store.retrieve(taskType: "code.swift")
        #expect(retrieved.count == 2)
        #expect(Set(retrieved) == Set([formatCall, lintCall]))
    }

    @Test("ProceduralStore evicts oldest entry at capacity")
    func proceduralEvictsOldest() async {
        let store = ProceduralStore()

        for i in 0..<1001 {
            let call = ToolCall(name: "tool-\(i)", input: "in", output: "out", durationMs: Double(i))
            await store.record(taskType: "task.\(i)", tools: [call])
        }

        #expect(await store.count == 1000)
        let oldest = await store.retrieve(taskType: "task.0")
        #expect(oldest.isEmpty)
    }

    @Test("EpisodicStore insert fails when turn embedding is nil")
    func episodicInsertFailsForNilEmbedding() async throws {
        let store = EpisodicStore()
        let turn = Turn(role: .assistant, content: "no embedding")

        do {
            try await store.insert(turn: turn)
            #expect(Bool(false), "Expected embeddingFailed error")
        } catch {
            #expect(Bool(true))
        }
    }

    private func makeVector(row: Int, dim: Int) -> [Float] {
        (0..<dim).map { col in
            let i = Float(row * dim + col)
            return sin(i * 0.173) + cos(i * 0.071)
        }
    }

    private func sampleToolCalls(prefix: String) -> [ToolCall] {
        (0..<5).map { index in
            ToolCall(
                name: "\(prefix)-tool-\(index)",
                input: "input-\(index)",
                output: "output-\(index)",
                durationMs: Double(index)
            )
        }
    }
}
