import Foundation
import Testing
@testable import ContextCore

@Suite("Window Packer Tests")
struct WindowPackerTests {
    @Test("ContextWindow totalTokens sums chunk token counts")
    func contextWindowTotalTokens() {
        let chunks = [
            makeContextChunk(content: "A", tokenCount: 100),
            makeContextChunk(content: "B", tokenCount: 200),
            makeContextChunk(content: "C", tokenCount: 300),
        ]

        let window = ContextWindow(chunks: chunks, budget: 1_000)
        #expect(window.totalTokens == 600)
    }

    @Test("ContextWindow budgetUsed is total divided by budget")
    func contextWindowBudgetUsed() {
        let chunks = [
            makeContextChunk(content: "A", tokenCount: 100),
            makeContextChunk(content: "B", tokenCount: 200),
            makeContextChunk(content: "C", tokenCount: 300),
        ]

        let window = ContextWindow(chunks: chunks, budget: 1_000)
        #expect(abs(window.budgetUsed - 0.6) < 1e-6)
    }

    @Test("ContextWindow raw formatting joins chunk content")
    func contextWindowRawFormatting() {
        let chunks = [
            makeContextChunk(content: "Hello", role: .user),
            makeContextChunk(content: "World", role: .assistant),
        ]

        let window = ContextWindow(chunks: chunks, budget: 1_000)
        #expect(window.formatted(style: .raw) == "Hello\n\nWorld")
    }

    @Test("ContextWindow chatML formatting includes role wrappers")
    func contextWindowChatMLFormatting() {
        let chunks = [
            makeContextChunk(content: "You are helpful", role: .system, source: .semantic, isSystemPrompt: true),
            makeContextChunk(content: "Hi", role: .user),
        ]

        let window = ContextWindow(chunks: chunks, budget: 1_000)
        let formatted = window.formatted(style: .chatML)

        #expect(formatted.contains("<|im_start|>system\nYou are helpful<|im_end|>"))
        #expect(formatted.contains("<|im_start|>user\nHi<|im_end|>"))
    }

    @Test("ContextWindow alpaca formatting uses role sections")
    func contextWindowAlpacaFormatting() {
        let chunks = [
            makeContextChunk(content: "You are helpful", role: .system, source: .semantic, isSystemPrompt: true),
            makeContextChunk(content: "Question", role: .user),
            makeContextChunk(content: "Answer", role: .assistant),
        ]

        let window = ContextWindow(chunks: chunks, budget: 1_000)
        let formatted = window.formatted(style: .alpaca)

        let instructionRange = formatted.range(of: "### Instruction:")
        let inputRange = formatted.range(of: "### Input:")
        let responseRange = formatted.range(of: "### Response:")

        #expect(instructionRange != nil)
        #expect(inputRange != nil)
        #expect(responseRange != nil)

        if let instructionRange, let inputRange, let responseRange {
            #expect(instructionRange.lowerBound < inputRange.lowerBound)
            #expect(inputRange.lowerBound < responseRange.lowerBound)
        }
    }

    @Test("ContextWindow custom formatting replaces role and content placeholders")
    func contextWindowCustomFormatting() {
        let chunks = [
            makeContextChunk(content: "Hello", role: .user),
            makeContextChunk(content: "World", role: .assistant),
        ]

        let window = ContextWindow(chunks: chunks, budget: 1_000)
        let formatted = window.formatted(style: .custom(template: "[{role}] {content}"))

        #expect(formatted == "[user] Hello\n[assistant] World")
    }

    @Test("ContextWindow empty chunks have zero tokens and empty raw formatting")
    func contextWindowEmpty() {
        let window = ContextWindow(chunks: [], budget: 1_000)

        #expect(window.totalTokens == 0)
        #expect(window.formatted(style: .raw).isEmpty)
    }

    @Test("ContextChunk Codable roundtrip preserves all fields")
    func contextChunkCodableRoundtrip() throws {
        let timestamp = Date(timeIntervalSince1970: 1_700_111_222)
        let chunk = ContextChunk(
            id: UUID(uuidString: "A1A1A1A1-A1A1-A1A1-A1A1-A1A1A1A1A1A1")!,
            content: "Codable payload",
            role: .tool,
            tokenCount: 123,
            score: 0.88,
            source: .procedural,
            compressionLevel: .light,
            timestamp: timestamp,
            isGuaranteedRecent: true,
            isSystemPrompt: false
        )

        let encoded = try JSONEncoder().encode(chunk)
        let decoded = try JSONDecoder().decode(ContextChunk.self, from: encoded)

        #expect(decoded == chunk)
    }

    @Test("CompressionLevel ordering follows none < light < heavy < dropped")
    func compressionLevelOrdering() {
        #expect(CompressionLevel.none < CompressionLevel.light)
        #expect(CompressionLevel.light < CompressionLevel.heavy)
        #expect(CompressionLevel.heavy < CompressionLevel.dropped)
    }

    @Test("ContextWindow retrievedFromMemory counts episodic and semantic retrieval only")
    func contextWindowRetrievedFromMemoryCount() {
        let chunks = [
            makeContextChunk(content: "episodic-1", source: .episodic),
            makeContextChunk(content: "episodic-2", source: .episodic),
            makeContextChunk(content: "semantic-1", source: .semantic),
            makeContextChunk(content: "system", role: .system, source: .semantic, isSystemPrompt: true),
        ]

        let window = ContextWindow(chunks: chunks, budget: 1_000)
        #expect(window.retrievedFromMemory == 3)
    }

    @Test("WindowPacker respects budget when guaranteed chunks do not exceed it")
    func packerRespectsBudget() async throws {
        let ranker = MockSentenceRanker()
        let packer = WindowPacker(
            sentenceRanker: ranker,
            tokenCounter: ApproximateTokenCounter(),
            minimumChunkSize: 50,
            recentTurnsGuaranteed: 3
        )

        let scoredMemory = (0..<20).map { index in
            makeScoredChunk(tokenCount: 400, score: Float(index) / 20.0, type: .episodic)
        }

        let window = try await packer.pack(
            systemPrompt: nil,
            recentTurns: [],
            scoredMemory: scoredMemory,
            budget: 4_096
        )

        #expect(window.totalTokens <= 4_096)
    }

    @Test("WindowPacker includes higher-scored chunks before lower-scored chunks")
    func packerHighestScoresFirst() async throws {
        let ranker = MockSentenceRanker()
        let packer = WindowPacker(
            sentenceRanker: ranker,
            tokenCounter: ApproximateTokenCounter(),
            minimumChunkSize: 50,
            recentTurnsGuaranteed: 3
        )

        let scoredMemory = (1...20).map { raw in
            makeScoredChunk(tokenCount: 300, score: Float(raw) / 20.0, type: .episodic)
        }

        let window = try await packer.pack(
            systemPrompt: nil,
            recentTurns: [],
            scoredMemory: scoredMemory,
            budget: 1_200
        )

        let includedIDs = Set(
            window.chunks
                .filter { !$0.isSystemPrompt && !$0.isGuaranteedRecent }
                .map(\.id)
        )

        let includedScores = scoredMemory
            .filter { includedIDs.contains($0.chunk.id) }
            .map(\.score)
        let excludedScores = scoredMemory
            .filter { !includedIDs.contains($0.chunk.id) }
            .map(\.score)

        #expect(!includedScores.isEmpty)
        #expect(!excludedScores.isEmpty)
        #expect((includedScores.min() ?? 0) >= (excludedScores.max() ?? 0))
    }

    @Test("WindowPacker guarantees inclusion of last N recent turns")
    func packerRecentTurnsGuaranteed() async throws {
        let ranker = MockSentenceRanker()
        let packer = WindowPacker(
            sentenceRanker: ranker,
            tokenCounter: ApproximateTokenCounter(),
            minimumChunkSize: 50,
            recentTurnsGuaranteed: 3
        )

        let turns = (0..<5).map { index in
            makeTurn(content: makeContent(approximateTokenCount: 100, stem: "turn\(index)"), role: index % 2 == 0 ? .user : .assistant)
        }
        let expectedRecentIDs = turns.suffix(3).map(\.id)

        let window = try await packer.pack(
            systemPrompt: nil,
            recentTurns: turns,
            scoredMemory: [],
            budget: 2_000
        )

        let outputIDs = Set(window.chunks.map(\.id))
        for id in expectedRecentIDs {
            #expect(outputIDs.contains(id))
        }

        let guaranteedIDs = Set(window.chunks.filter(\.isGuaranteedRecent).map(\.id))
        #expect(guaranteedIDs == Set(expectedRecentIDs))
    }

    @Test("WindowPacker always includes system prompt")
    func packerAlwaysIncludesSystemPrompt() async throws {
        let ranker = MockSentenceRanker()
        let packer = WindowPacker(
            sentenceRanker: ranker,
            tokenCounter: ApproximateTokenCounter(),
            minimumChunkSize: 50,
            recentTurnsGuaranteed: 3
        )

        let prompt = makeContent(approximateTokenCount: 200, stem: "system")
        let window = try await packer.pack(
            systemPrompt: prompt,
            recentTurns: [],
            scoredMemory: [],
            budget: 4_096
        )

        let systemChunks = window.chunks.filter(\.isSystemPrompt)
        #expect(systemChunks.count == 1)
        #expect(systemChunks[0].content == prompt)
    }

    @Test("WindowPacker keeps guaranteed chunks even when they exceed budget")
    func packerGuaranteedChunksCanExceedBudget() async throws {
        let ranker = MockSentenceRanker()
        let packer = WindowPacker(
            sentenceRanker: ranker,
            tokenCounter: ApproximateTokenCounter(),
            minimumChunkSize: 50,
            recentTurnsGuaranteed: 3
        )

        let systemPrompt = makeContent(approximateTokenCount: 500, stem: "system")
        let recentTurns = (0..<3).map { index in
            makeTurn(content: makeContent(approximateTokenCount: 2_000, stem: "turn\(index)"), role: .user)
        }

        let scoredMemory = [makeScoredChunk(tokenCount: 100, score: 0.9, type: .semantic)]

        let window = try await packer.pack(
            systemPrompt: systemPrompt,
            recentTurns: recentTurns,
            scoredMemory: scoredMemory,
            budget: 4_096
        )

        #expect(window.chunks.count == 4)
        #expect(window.totalTokens > 4_096)
        #expect(window.retrievedFromMemory == 0)
    }

    @Test("WindowPacker compresses oversized chunk when compression can fit")
    func packerCompressionFallback() async throws {
        let s1 = makeSentence(approximateTokenCount: 220, stem: "s1")
        let s2 = makeSentence(approximateTokenCount: 200, stem: "s2")
        let s3 = makeSentence(approximateTokenCount: 180, stem: "s3")
        let content = [s1, s2, s3].joined(separator: " ")

        let ranker = MockSentenceRanker(canned: [
            content: [
                (sentence: s1, importance: 0.9),
                (sentence: s2, importance: 0.8),
                (sentence: s3, importance: 0.7),
            ]
        ])

        let packer = WindowPacker(
            sentenceRanker: ranker,
            tokenCounter: ApproximateTokenCounter(),
            minimumChunkSize: 50,
            recentTurnsGuaranteed: 3
        )

        let scoredMemory = [
            makeScoredChunk(content: content, tokenCount: 800, score: 0.95, type: .semantic)
        ]

        let window = try await packer.pack(
            systemPrompt: nil,
            recentTurns: [],
            scoredMemory: scoredMemory,
            budget: 400
        )

        #expect(window.chunks.count == 1)
        #expect(window.chunks[0].compressionLevel == .light)
        #expect(window.chunks[0].tokenCount <= 400)
        #expect(window.chunks[0].content != content)
    }

    @Test("WindowPacker drops chunk when top sentence cannot fit remaining budget")
    func packerDropsChunkIfTopSentenceTooLarge() async throws {
        let longSentence = makeSentence(approximateTokenCount: 600, stem: "long")
        let content = longSentence

        let ranker = MockSentenceRanker(canned: [
            content: [
                (sentence: longSentence, importance: 1.0)
            ]
        ])

        let packer = WindowPacker(
            sentenceRanker: ranker,
            tokenCounter: ApproximateTokenCounter(),
            minimumChunkSize: 50,
            recentTurnsGuaranteed: 3
        )

        let scoredMemory = [makeScoredChunk(content: content, tokenCount: 600, score: 0.95, type: .semantic)]

        let window = try await packer.pack(
            systemPrompt: nil,
            recentTurns: [],
            scoredMemory: scoredMemory,
            budget: 400
        )

        #expect(window.chunks.isEmpty)
    }

    @Test("WindowPacker short-circuits memory packing below minimumChunkSize")
    func packerShortCircuitAtMinimumChunkSize() async throws {
        let ranker = MockSentenceRanker()
        let packer = WindowPacker(
            sentenceRanker: ranker,
            tokenCounter: ApproximateTokenCounter(),
            minimumChunkSize: 50,
            recentTurnsGuaranteed: 3
        )

        let scoredMemory = (0..<5).map { index in
            makeScoredChunk(tokenCount: 40, score: Float(5 - index) / 5.0, type: .episodic)
        }

        let window = try await packer.pack(
            systemPrompt: nil,
            recentTurns: [],
            scoredMemory: scoredMemory,
            budget: 30
        )

        #expect(window.chunks.isEmpty)
    }

    @Test("WindowPacker returns empty window for empty inputs")
    func packerEmptyInputs() async throws {
        let ranker = MockSentenceRanker()
        let packer = WindowPacker(
            sentenceRanker: ranker,
            tokenCounter: ApproximateTokenCounter(),
            minimumChunkSize: 50,
            recentTurnsGuaranteed: 3
        )

        let window = try await packer.pack(
            systemPrompt: nil,
            recentTurns: [],
            scoredMemory: [],
            budget: 4_096
        )

        #expect(window.chunks.isEmpty)
        #expect(window.totalTokens == 0)
    }

    @Test("WindowPacker handles zero budget with and without guaranteed items")
    func packerZeroBudget() async throws {
        let ranker = MockSentenceRanker()
        let packer = WindowPacker(
            sentenceRanker: ranker,
            tokenCounter: ApproximateTokenCounter(),
            minimumChunkSize: 50,
            recentTurnsGuaranteed: 3
        )

        let empty = try await packer.pack(
            systemPrompt: nil,
            recentTurns: [],
            scoredMemory: [makeScoredChunk(tokenCount: 100, score: 0.8, type: .semantic)],
            budget: 0
        )

        #expect(empty.chunks.isEmpty)

        let guaranteed = try await packer.pack(
            systemPrompt: makeContent(approximateTokenCount: 100, stem: "system"),
            recentTurns: [makeTurn(content: makeContent(approximateTokenCount: 120, stem: "recent"), role: .user)],
            scoredMemory: [],
            budget: 0
        )

        #expect(guaranteed.chunks.count == 2)
        #expect(guaranteed.totalTokens > 0)
    }

    @Test("WindowPacker includes chunk that exactly matches budget")
    func packerSingleChunkFitsExactly() async throws {
        let counter = ApproximateTokenCounter()
        let ranker = MockSentenceRanker()
        let packer = WindowPacker(
            sentenceRanker: ranker,
            tokenCounter: counter,
            minimumChunkSize: 50,
            recentTurnsGuaranteed: 3
        )

        let content = makeContent(approximateTokenCount: 4_095, stem: "exact")
        let exact = counter.count(content)
        let scoredMemory = [makeScoredChunk(content: content, tokenCount: exact, score: 0.95, type: .semantic)]

        let window = try await packer.pack(
            systemPrompt: nil,
            recentTurns: [],
            scoredMemory: scoredMemory,
            budget: exact
        )

        #expect(window.chunks.count == 1)
        #expect(window.totalTokens == exact)
        #expect(abs(window.budgetUsed - 1.0) < 1e-6)
    }

    @Test("WindowPacker reports retrievedFromMemory for episodic and semantic chunks")
    func packerRetrievedFromMemoryCount() async throws {
        let ranker = MockSentenceRanker()
        let packer = WindowPacker(
            sentenceRanker: ranker,
            tokenCounter: ApproximateTokenCounter(),
            minimumChunkSize: 50,
            recentTurnsGuaranteed: 3
        )

        let scoredMemory: [(chunk: MemoryChunk, score: Float)] = [
            makeScoredChunk(tokenCount: 100, score: 0.95, type: .episodic),
            makeScoredChunk(tokenCount: 100, score: 0.94, type: .episodic),
            makeScoredChunk(tokenCount: 100, score: 0.93, type: .episodic),
            makeScoredChunk(tokenCount: 100, score: 0.92, type: .semantic),
            makeScoredChunk(tokenCount: 100, score: 0.91, type: .semantic),
        ]

        let window = try await packer.pack(
            systemPrompt: nil,
            recentTurns: [],
            scoredMemory: scoredMemory,
            budget: 2_000
        )

        #expect(window.retrievedFromMemory == 5)
    }

    @Test("WindowPacker reports compressedChunks count")
    func packerCompressedChunksCount() async throws {
        let c1s1 = makeSentence(approximateTokenCount: 400, stem: "c1s1")
        let c1s2 = makeSentence(approximateTokenCount: 250, stem: "c1s2")
        let c1s3 = makeSentence(approximateTokenCount: 250, stem: "c1s3")
        let content1 = [c1s1, c1s2, c1s3].joined(separator: " ")

        let c2s1 = makeSentence(approximateTokenCount: 180, stem: "c2s1")
        let c2s2 = makeSentence(approximateTokenCount: 180, stem: "c2s2")
        let c2s3 = makeSentence(approximateTokenCount: 180, stem: "c2s3")
        let content2 = [c2s1, c2s2, c2s3].joined(separator: " ")

        let ranker = MockSentenceRanker(canned: [
            content1: [
                (sentence: c1s1, importance: 0.9),
                (sentence: c1s2, importance: 0.8),
                (sentence: c1s3, importance: 0.7),
            ],
            content2: [
                (sentence: c2s1, importance: 0.9),
                (sentence: c2s2, importance: 0.8),
                (sentence: c2s3, importance: 0.7),
            ],
        ])

        let packer = WindowPacker(
            sentenceRanker: ranker,
            tokenCounter: ApproximateTokenCounter(),
            minimumChunkSize: 50,
            recentTurnsGuaranteed: 3
        )

        let scoredMemory: [(chunk: MemoryChunk, score: Float)] = [
            makeScoredChunk(content: content1, tokenCount: 900, score: 0.99, type: .semantic),
            makeScoredChunk(content: content2, tokenCount: 600, score: 0.98, type: .semantic),
        ]

        let window = try await packer.pack(
            systemPrompt: nil,
            recentTurns: [],
            scoredMemory: scoredMemory,
            budget: 600
        )

        #expect(window.compressedChunks == 2)
    }

    @Test("WindowPacker keeps memory chunks in non-increasing score order")
    func packerScoreOrderingPreserved() async throws {
        let ranker = MockSentenceRanker()
        let packer = WindowPacker(
            sentenceRanker: ranker,
            tokenCounter: ApproximateTokenCounter(),
            minimumChunkSize: 50,
            recentTurnsGuaranteed: 3
        )

        let scoredMemory: [(chunk: MemoryChunk, score: Float)] = (0..<10).map { index in
            let score = Float(10 - index) / 10.0
            return makeScoredChunk(tokenCount: 100, score: score, type: .semantic)
        }

        let window = try await packer.pack(
            systemPrompt: nil,
            recentTurns: [],
            scoredMemory: scoredMemory,
            budget: 2_000
        )

        let memoryScores = window.chunks
            .filter { !$0.isSystemPrompt && !$0.isGuaranteedRecent }
            .map(\.score)

        for index in 0..<(memoryScores.count - 1) {
            #expect(memoryScores[index] >= memoryScores[index + 1])
        }
    }

    private func makeContextChunk(
        content: String,
        role: TurnRole = .assistant,
        tokenCount: Int? = nil,
        score: Float = 0.5,
        source: MemoryType = .episodic,
        compressionLevel: CompressionLevel = .none,
        timestamp: Date = .now,
        isGuaranteedRecent: Bool = false,
        isSystemPrompt: Bool = false
    ) -> ContextChunk {
        ContextChunk(
            content: content,
            role: role,
            tokenCount: tokenCount ?? ApproximateTokenCounter().count(content),
            score: score,
            source: source,
            compressionLevel: compressionLevel,
            timestamp: timestamp,
            isGuaranteedRecent: isGuaranteedRecent,
            isSystemPrompt: isSystemPrompt
        )
    }

    private func makeScoredChunk(
        content: String = "Test content for memory chunk evaluation.",
        tokenCount: Int = 100,
        score: Float = 0.5,
        type: MemoryType = .episodic
    ) -> (chunk: MemoryChunk, score: Float) {
        let resolvedContent: String
        if content == "Test content for memory chunk evaluation." {
            resolvedContent = makeContent(approximateTokenCount: tokenCount, stem: "chunk")
        } else {
            resolvedContent = content
        }

        let chunk = MemoryChunk(
            content: resolvedContent,
            embedding: [0.1, 0.2, 0.3, 0.4],
            type: type,
            retentionScore: 1.0,
            sourceSessionID: UUID()
        )

        return (chunk: chunk, score: score)
    }

    private func makeTurn(content: String, role: TurnRole) -> Turn {
        Turn(
            id: UUID(),
            role: role,
            content: content,
            timestamp: Date(),
            tokenCount: ApproximateTokenCounter().count(content),
            embedding: nil,
            metadata: [:]
        )
    }

    private func makeSentence(approximateTokenCount: Int, stem: String) -> String {
        makeContent(approximateTokenCount: approximateTokenCount, stem: stem) + "."
    }

    private func makeContent(approximateTokenCount: Int, stem: String) -> String {
        let targetWords = max(1, Int(ceil(Double(approximateTokenCount) / 1.3)))
        let words = (0..<targetWords).map { index in "\(stem)\(index)" }
        return words.joined(separator: " ")
    }
}

private actor MockSentenceRanker: SentenceRanker {
    private let canned: [String: [(sentence: String, importance: Float)]]

    init(canned: [String: [(sentence: String, importance: Float)]] = [:]) {
        self.canned = canned
    }

    func rankSentences(
        in chunk: String,
        chunkEmbedding: [Float]
    ) async throws -> [(sentence: String, importance: Float)] {
        if let canned = canned[chunk] {
            return canned
        }

        let sentence = chunk.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !sentence.isEmpty else {
            return []
        }

        return [(sentence: sentence, importance: 1.0)]
    }
}
