import Foundation
import Testing
@testable import ContextCore

@Suite("Chunk Orderer Tests")
struct ChunkOrdererTests {
    @Test("typeGrouped places groups in system -> semantic -> episodic -> procedural -> recent order")
    func typeGroupedBasic() {
        let orderer = ChunkOrderer()

        let system = makeChunk(label: "system", source: .semantic, role: .system, timestamp: 1, isSystemPrompt: true)
        let semantic1 = makeChunk(label: "semantic-1", source: .semantic, timestamp: 2)
        let semantic2 = makeChunk(label: "semantic-2", source: .semantic, timestamp: 3)
        let episodic1 = makeChunk(label: "episodic-1", source: .episodic, timestamp: 4)
        let episodic2 = makeChunk(label: "episodic-2", source: .episodic, timestamp: 5)
        let episodic3 = makeChunk(label: "episodic-3", source: .episodic, timestamp: 6)
        let procedural = makeChunk(label: "procedural", source: .procedural, timestamp: 7)
        let recent1 = makeChunk(label: "recent-1", source: .episodic, role: .assistant, timestamp: 8, isGuaranteedRecent: true)
        let recent2 = makeChunk(label: "recent-2", source: .episodic, role: .user, timestamp: 9, isGuaranteedRecent: true)

        let ordered = orderer.order(
            [recent2, episodic2, semantic2, system, procedural, semantic1, episodic1, recent1, episodic3],
            strategy: .typeGrouped
        )

        #expect(ordered.map(\.id) == [
            system.id,
            semantic1.id,
            semantic2.id,
            episodic1.id,
            episodic2.id,
            episodic3.id,
            procedural.id,
            recent1.id,
            recent2.id,
        ])
    }

    @Test("typeGrouped sorts chunks chronologically within each group")
    func typeGroupedChronologicalWithinGroup() {
        let orderer = ChunkOrderer()

        let system = makeChunk(label: "system", source: .semantic, role: .system, timestamp: 0, isSystemPrompt: true)
        let e3 = makeChunk(label: "e3", source: .episodic, timestamp: 30)
        let e1 = makeChunk(label: "e1", source: .episodic, timestamp: 10)
        let e2 = makeChunk(label: "e2", source: .episodic, timestamp: 20)

        let ordered = orderer.order([e3, system, e1, e2], strategy: .typeGrouped)
        let episodic = ordered.filter { $0.source == .episodic && !$0.isGuaranteedRecent }

        #expect(episodic.map(\.id) == [e1.id, e2.id, e3.id])
    }

    @Test("relevanceAscending sorts by score ascending")
    func relevanceAscending() {
        let orderer = ChunkOrderer()

        let chunks = [
            makeChunk(label: "a", source: .episodic, score: 0.9, timestamp: 1),
            makeChunk(label: "b", source: .episodic, score: 0.3, timestamp: 2),
            makeChunk(label: "c", source: .episodic, score: 0.7, timestamp: 3),
            makeChunk(label: "d", source: .episodic, score: 0.1, timestamp: 4),
            makeChunk(label: "e", source: .episodic, score: 0.5, timestamp: 5),
        ]

        let ordered = orderer.order(chunks, strategy: .relevanceAscending)
        #expect(ordered.map(\.score) == [0.1, 0.3, 0.5, 0.7, 0.9])
    }

    @Test("relevanceAscending keeps system prompt pinned at index 0")
    func relevanceAscendingSystemPinned() {
        let orderer = ChunkOrderer()

        let system = makeChunk(label: "system", source: .semantic, role: .system, score: 0.0, timestamp: 9, isSystemPrompt: true)
        let c1 = makeChunk(label: "c1", source: .episodic, score: 0.9, timestamp: 1)
        let c2 = makeChunk(label: "c2", source: .episodic, score: 0.2, timestamp: 2)
        let c3 = makeChunk(label: "c3", source: .episodic, score: 0.6, timestamp: 3)

        let ordered = orderer.order([c1, c2, system, c3], strategy: .relevanceAscending)

        #expect(ordered.first?.id == system.id)
        #expect(ordered.dropFirst().map(\.score) == [0.2, 0.6, 0.9])
    }

    @Test("chronological sorts by timestamp ascending")
    func chronological() {
        let orderer = ChunkOrderer()

        let c1 = makeChunk(label: "1", source: .episodic, timestamp: 30)
        let c2 = makeChunk(label: "2", source: .episodic, timestamp: 10)
        let c3 = makeChunk(label: "3", source: .semantic, timestamp: 20)
        let c4 = makeChunk(label: "4", source: .procedural, timestamp: 40)
        let c5 = makeChunk(label: "5", source: .episodic, timestamp: 50)

        let ordered = orderer.order([c1, c2, c3, c4, c5], strategy: .chronological)
        #expect(ordered.map(\.id) == [c2.id, c3.id, c1.id, c4.id, c5.id])
    }

    @Test("chronological keeps system prompt pinned at index 0")
    func chronologicalSystemPinned() {
        let orderer = ChunkOrderer()

        let system = makeChunk(label: "system", source: .semantic, role: .system, timestamp: 999, isSystemPrompt: true)
        let old = makeChunk(label: "old", source: .episodic, timestamp: 1)
        let mid = makeChunk(label: "mid", source: .episodic, timestamp: 2)

        let ordered = orderer.order([mid, system, old], strategy: .chronological)

        #expect(ordered.first?.id == system.id)
        #expect(ordered.dropFirst().map(\.id) == [old.id, mid.id])
    }

    @Test("single chunk input is unchanged")
    func singleChunk() {
        let orderer = ChunkOrderer()
        let chunk = makeChunk(label: "only", source: .episodic, timestamp: 1)

        #expect(orderer.order([chunk], strategy: .typeGrouped) == [chunk])
        #expect(orderer.order([chunk], strategy: .relevanceAscending) == [chunk])
        #expect(orderer.order([chunk], strategy: .chronological) == [chunk])
    }

    @Test("empty input returns empty output")
    func emptyInput() {
        let orderer = ChunkOrderer()

        #expect(orderer.order([], strategy: .typeGrouped).isEmpty)
        #expect(orderer.order([], strategy: .relevanceAscending).isEmpty)
        #expect(orderer.order([], strategy: .chronological).isEmpty)
    }

    @Test("typeGrouped with same type falls back to chronological order")
    func allSameType() {
        let orderer = ChunkOrderer()

        let c1 = makeChunk(label: "1", source: .episodic, timestamp: 30)
        let c2 = makeChunk(label: "2", source: .episodic, timestamp: 10)
        let c3 = makeChunk(label: "3", source: .episodic, timestamp: 20)
        let c4 = makeChunk(label: "4", source: .episodic, timestamp: 40)
        let c5 = makeChunk(label: "5", source: .episodic, timestamp: 50)

        let ordered = orderer.order([c1, c2, c3, c4, c5], strategy: .typeGrouped)
        #expect(ordered.map(\.id) == [c2.id, c3.id, c1.id, c4.id, c5.id])
    }

    @Test("typeGrouped places retrieved episodic memory before guaranteed recent turns")
    func guaranteedRecentSeparation() {
        let orderer = ChunkOrderer()

        let system = makeChunk(label: "system", source: .semantic, role: .system, timestamp: 0, isSystemPrompt: true)
        let episodic1 = makeChunk(label: "episodic-1", source: .episodic, timestamp: 1)
        let episodic2 = makeChunk(label: "episodic-2", source: .episodic, timestamp: 2)
        let recent1 = makeChunk(label: "recent-1", source: .episodic, role: .assistant, timestamp: 3, isGuaranteedRecent: true)
        let recent2 = makeChunk(label: "recent-2", source: .episodic, role: .user, timestamp: 4, isGuaranteedRecent: true)

        let ordered = orderer.order([recent1, episodic2, system, recent2, episodic1], strategy: .typeGrouped)
        #expect(ordered.map(\.id) == [system.id, episodic1.id, episodic2.id, recent1.id, recent2.id])
    }

    private func makeChunk(
        label: String,
        source: MemoryType,
        role: TurnRole = .assistant,
        score: Float = 0.5,
        timestamp: TimeInterval,
        isGuaranteedRecent: Bool = false,
        isSystemPrompt: Bool = false
    ) -> ContextChunk {
        ContextChunk(
            content: label,
            role: role,
            tokenCount: ApproximateTokenCounter().count(label),
            score: score,
            source: source,
            compressionLevel: .none,
            timestamp: Date(timeIntervalSince1970: timestamp),
            isGuaranteedRecent: isGuaranteedRecent,
            isSystemPrompt: isSystemPrompt
        )
    }
}
