import Foundation
import NaturalLanguage
import Testing
@testable import ContextCore

#if !targetEnvironment(simulator)
@Suite("Compression Engine Tests")
struct CompressionEngineTests {
    private let tokenCounter = ApproximateTokenCounter()

    @Test("Extractive compress meets token budget")
    func extractiveCompressBudgetMet() async throws {
        let provider = SemanticMockEmbeddingProvider()
        let engine = try makeEngine(provider: provider)
        let delegate = ExtractiveFallbackDelegate(
            compressionEngine: engine,
            tokenCounter: tokenCounter
        )

        let text = testParagraph()
        let compressed = try await delegate.compress(text, targetTokens: 150)

        #expect(tokenCounter.count(compressed) <= 150)
    }

    @Test("Extractive compress keeps highest-ranked sentence")
    func extractiveCompressKeepsBestSentence() async throws {
        let provider = SemanticMockEmbeddingProvider()
        let engine = try makeEngine(provider: provider)
        let delegate = ExtractiveFallbackDelegate(
            compressionEngine: engine,
            tokenCounter: tokenCounter
        )

        let text = testParagraph()
        let embedding = try await provider.embed(text)
        let ranked = try await engine.rankSentences(in: text, chunkEmbedding: embedding)
        let best = ranked.first?.sentence

        let compressed = try await delegate.compress(text, targetTokens: 150)

        if let best {
            #expect(compressed.contains(best))
        } else {
            Issue.record("Expected at least one ranked sentence")
        }
    }

    @Test("Extractive compress preserves original sentence order")
    func extractiveCompressOrderPreserved() async throws {
        let provider = SemanticMockEmbeddingProvider()
        let engine = try makeEngine(provider: provider)
        let delegate = ExtractiveFallbackDelegate(
            compressionEngine: engine,
            tokenCounter: tokenCounter
        )

        let text = [
            "Swift introduced modern syntax for safer development.",
            "Automatic Reference Counting manages memory automatically.",
            "Protocols and generics enable reusable abstractions.",
            "Compilers can optimize value types effectively.",
            "The weather in Cupertino is typically mild.",
        ].joined(separator: " ")

        let compressed = try await delegate.compress(text, targetTokens: 40)

        let original = splitSentences(text)
        let output = splitSentences(compressed)

        let indices = output.compactMap { sentence in
            original.firstIndex(of: sentence)
        }

        #expect(indices.count == output.count)
        for index in 0..<(max(indices.count - 1, 0)) {
            #expect(indices[index] < indices[index + 1])
        }
    }

    @Test("Extractive compress returns input when already under budget")
    func extractiveCompressAlreadyFits() async throws {
        let provider = SemanticMockEmbeddingProvider()
        let engine = try makeEngine(provider: provider)
        let delegate = ExtractiveFallbackDelegate(
            compressionEngine: engine,
            tokenCounter: tokenCounter
        )

        let text = "Swift is safe and fast."
        let compressed = try await delegate.compress(text, targetTokens: 200)

        #expect(compressed == text)
    }

    @Test("Extractive compress returns one sentence when sentence exceeds target")
    func extractiveCompressSingleSentenceOverTarget() async throws {
        let provider = SemanticMockEmbeddingProvider()
        let engine = try makeEngine(provider: provider)
        let delegate = ExtractiveFallbackDelegate(
            compressionEngine: engine,
            tokenCounter: tokenCounter
        )

        let sentence = makeLongSentence(approximateTokens: 300, stem: "swift")
        let compressed = try await delegate.compress(sentence, targetTokens: 100)

        #expect(compressed == sentence)
    }

    @Test("extractFacts splits into standalone sentences")
    func extractFactsSentenceSplit() async throws {
        let provider = SemanticMockEmbeddingProvider()
        let engine = try makeEngine(provider: provider)
        let delegate = ExtractiveFallbackDelegate(
            compressionEngine: engine,
            tokenCounter: tokenCounter
        )

        let facts = try await delegate.extractFacts(from: "First fact. Second fact. Third fact!")

        #expect(facts == ["First fact.", "Second fact.", "Third fact!"])
    }

    @Test("extractFacts returns empty array for empty input")
    func extractFactsEmpty() async throws {
        let provider = SemanticMockEmbeddingProvider()
        let engine = try makeEngine(provider: provider)
        let delegate = ExtractiveFallbackDelegate(
            compressionEngine: engine,
            tokenCounter: tokenCounter
        )

        let facts = try await delegate.extractFacts(from: "")

        #expect(facts.isEmpty)
    }

    @Test("compress(chunk) reduces content to target budget")
    func compressChunkBasic() async throws {
        let provider = SemanticMockEmbeddingProvider()
        let engine = try makeEngine(provider: provider)

        let content = makeLongParagraph(approximateTokens: 400)
        let chunk = try await makeChunk(content: content, provider: provider)

        let compressed = try await engine.compress(chunk: chunk, targetTokens: 100)

        #expect(tokenCounter.count(compressed.content) <= 100)
    }

    @Test("compress(chunk) keeps highest-ranked sentence")
    func compressChunkKeepsBestSentence() async throws {
        let provider = SemanticMockEmbeddingProvider()
        let engine = try makeEngine(provider: provider)

        let content = testParagraph()
        let chunk = try await makeChunk(content: content, provider: provider)
        let ranked = try await engine.rankSentences(in: chunk.content, chunkEmbedding: chunk.embedding)
        let top = ranked.first?.sentence

        let compressed = try await engine.compress(chunk: chunk, targetTokens: 70)

        if let top {
            #expect(compressed.content.contains(top))
        } else {
            Issue.record("Expected at least one ranked sentence")
        }
    }

    @Test("compress(chunk) returns original when already under target")
    func compressChunkAlreadyFits() async throws {
        let provider = SemanticMockEmbeddingProvider()
        let engine = try makeEngine(provider: provider)

        let content = "Swift is concise and expressive."
        let chunk = try await makeChunk(content: content, provider: provider)

        let compressed = try await engine.compress(chunk: chunk, targetTokens: 200)

        #expect(compressed == chunk)
    }

    @Test("compress(chunk) sets compression metadata")
    func compressChunkSetsMetadata() async throws {
        let provider = SemanticMockEmbeddingProvider()
        let engine = try makeEngine(provider: provider)

        let content = makeLongParagraph(approximateTokens: 400)
        let chunk = try await makeChunk(content: content, provider: provider)

        let compressed = try await engine.compress(chunk: chunk, targetTokens: 100)
        let ratio = compressed.metadata["compressionRatio"].flatMap(Float.init)

        #expect(compressed.metadata["compressionRatio"] != nil)
        #expect(compressed.metadata["originalTokenCount"] == "\(tokenCounter.count(content))")
        #expect(ratio != nil)
        #expect(abs((ratio ?? 0) - 4.0) < 1.5)
    }

    @Test("compress(chunk) refreshes embedding after content change")
    func compressChunkUpdatesEmbedding() async throws {
        let provider = SemanticMockEmbeddingProvider()
        let engine = try makeEngine(provider: provider)

        let content = makeLongParagraph(approximateTokens: 400)
        let chunk = try await makeChunk(content: content, provider: provider)

        let compressed = try await engine.compress(chunk: chunk, targetTokens: 100)

        #expect(compressed.content != chunk.content)
        #expect(compressed.embedding != chunk.embedding)
    }

    @Test("compress(chunk) routes through injected delegate")
    func compressChunkUsesInjectedDelegate() async throws {
        let provider = SemanticMockEmbeddingProvider()
        let mock = MockCompressionDelegate(returnedText: "mock compressed content")
        let engine = try makeEngine(provider: provider, compressionDelegate: mock)

        let content = makeLongParagraph(approximateTokens: 240)
        let chunk = try await makeChunk(content: content, provider: provider)

        let compressed = try await engine.compress(chunk: chunk, targetTokens: 80)

        #expect(mock.compressCalled)
        #expect(mock.lastText == content)
        #expect(mock.lastTargetTokens == 80)
        #expect(compressed.content == "mock compressed content")
    }

    @Test("setCompressionDelegate replaces compression strategy")
    func setCompressionDelegateOverridesExisting() async throws {
        let provider = SemanticMockEmbeddingProvider()
        let engine = try makeEngine(provider: provider)
        let mock = MockCompressionDelegate(returnedText: "override delegate output")

        await engine.setCompressionDelegate(mock)

        let content = makeLongParagraph(approximateTokens: 240)
        let chunk = try await makeChunk(content: content, provider: provider)
        let compressed = try await engine.compress(chunk: chunk, targetTokens: 60)

        #expect(mock.compressCalled)
        #expect(compressed.content == "override delegate output")
    }

    @Test("compressTurn reduces turn to target budget")
    func compressTurnBasic() async throws {
        let provider = SemanticMockEmbeddingProvider()
        let engine = try makeEngine(provider: provider)

        let turn = makeTurn(content: makeLongParagraph(approximateTokens: 400), role: .assistant)
        let compressed = try await engine.compressTurn(turn: turn, targetTokens: 100)

        #expect(tokenCounter.count(compressed.content) <= 100)
    }

    @Test("compressTurn preserves identity fields")
    func compressTurnPreservesIdentity() async throws {
        let provider = SemanticMockEmbeddingProvider()
        let engine = try makeEngine(provider: provider)

        let turn = makeTurn(content: makeLongParagraph(approximateTokens: 320), role: .user)
        let compressed = try await engine.compressTurn(turn: turn, targetTokens: 100)

        #expect(compressed.id == turn.id)
        #expect(compressed.role == turn.role)
        #expect(compressed.timestamp == turn.timestamp)
    }

    @Test("compressTurn updates tokenCount to match compressed content")
    func compressTurnUpdatesTokenCount() async throws {
        let provider = SemanticMockEmbeddingProvider()
        let engine = try makeEngine(provider: provider)

        let turn = makeTurn(content: makeLongParagraph(approximateTokens: 320), role: .assistant)
        let compressed = try await engine.compressTurn(turn: turn, targetTokens: 100)

        #expect(compressed.tokenCount == tokenCounter.count(compressed.content))
    }

    @Test("compressTurn returns original turn when already under target")
    func compressTurnAlreadyFits() async throws {
        let provider = SemanticMockEmbeddingProvider()
        let engine = try makeEngine(provider: provider)

        let turn = makeTurn(content: "Short response.", role: .assistant)
        let compressed = try await engine.compressTurn(turn: turn, targetTokens: 200)

        #expect(compressed == turn)
    }

    private func makeEngine(
        provider: SemanticMockEmbeddingProvider,
        compressionDelegate: (any CompressionDelegate)? = nil
    ) throws -> CompressionEngine {
        try CompressionEngine(
            embeddingProvider: provider,
            tokenCounter: tokenCounter,
            compressionDelegate: compressionDelegate
        )
    }

    private func makeChunk(content: String, provider: SemanticMockEmbeddingProvider) async throws -> MemoryChunk {
        MemoryChunk(
            content: content,
            embedding: try await provider.embed(content),
            type: .episodic,
            retentionScore: 0.5,
            sourceSessionID: UUID(),
            metadata: [:]
        )
    }

    private func makeTurn(content: String, role: TurnRole) -> Turn {
        Turn(
            id: UUID(),
            role: role,
            content: content,
            timestamp: Date(),
            tokenCount: tokenCounter.count(content),
            embedding: nil,
            metadata: [:]
        )
    }

    private func splitSentences(_ text: String) -> [String] {
        let tokenizer = NLTokenizer(unit: .sentence)
        tokenizer.string = text

        var sentences: [String] = []
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let sentence = text[range].trimmingCharacters(in: .whitespacesAndNewlines)
            if !sentence.isEmpty {
                sentences.append(sentence)
            }
            return true
        }

        return sentences
    }

    private func testParagraph() -> String {
        """
        Swift is a powerful programming language developed by Apple. \
        It was first released in 2014 as a replacement for Objective-C. \
        Swift uses LLVM for compilation and achieves performance comparable to C++. \
        The weather in Cupertino is generally mild year-round. \
        Swift's type system prevents common programming errors at compile time. \
        Memory management in Swift is handled automatically through ARC.
        """
    }

    private func makeLongSentence(approximateTokens: Int, stem: String) -> String {
        let targetWords = max(1, Int(ceil(Double(approximateTokens) / 1.3)))
        let words = (0..<targetWords).map { index in "\(stem)\(index)" }
        return words.joined(separator: " ") + "."
    }

    private func makeLongParagraph(approximateTokens: Int) -> String {
        let sentenceA = "Swift concurrency uses async await and actors for safer parallel programming."
        let sentenceB = "Protocols and generics improve composability and maintainability in large codebases."
        let sentenceC = "Automatic Reference Counting handles object memory management in most scenarios."
        let sentenceD = "The weather in Cupertino is usually mild and pleasant throughout the year."
        let seed = [sentenceA, sentenceB, sentenceC, sentenceD].joined(separator: " ")

        var text = seed
        while tokenCounter.count(text) < approximateTokens {
            text += " " + seed
        }

        return text
    }
}

private final class MockCompressionDelegate: CompressionDelegate, @unchecked Sendable {
    var compressCalled = false
    var lastText: String?
    var lastTargetTokens: Int?

    private let returnedText: String

    init(returnedText: String) {
        self.returnedText = returnedText
    }

    func compress(_ text: String, targetTokens: Int) async throws -> String {
        compressCalled = true
        lastText = text
        lastTargetTokens = targetTokens
        return returnedText
    }

    func extractFacts(from text: String) async throws -> [String] {
        [text]
    }
}

private struct SemanticMockEmbeddingProvider: EmbeddingProvider, Sendable {
    let dimension: Int = 384

    func embed(_ text: String) async throws -> [Float] {
        let lower = text.lowercased()

        let swiftBase = TestHelpers.randomVector(dim: dimension, seed: 8101)
        let weatherBase = orthogonal(to: swiftBase, seed: 8102)
        let genericBase = orthogonal(to: weatherBase, seed: 8103)

        let base: [Float]
        if lower.contains("swift") || lower.contains("arc") || lower.contains("reference") || lower.contains("actor") || lower.contains("protocol") || lower.contains("task") || lower.contains("async") || lower.contains("llvm") || lower.contains("compile") {
            base = swiftBase
        } else if lower.contains("weather") || lower.contains("rain") || lower.contains("oven") || lower.contains("eggs") || lower.contains("whisk") {
            base = weatherBase
        } else {
            base = genericBase
        }

        var hasher = Hasher()
        hasher.combine(lower)
        let seed = UInt64(bitPattern: Int64(hasher.finalize()))
        var rng = TestHelpers.SeededGenerator(seed: seed)

        let noisy = base.map { value -> Float in
            let delta = (Float(rng.next() & 0xFFFF) / Float(0xFFFF)) * 2 - 1
            return value + (delta * 0.01)
        }

        return TestHelpers.l2Normalize(noisy)
    }

    func embedBatch(_ texts: [String]) async throws -> [[Float]] {
        try await texts.asyncMap { text in
            try await embed(text)
        }
    }

    private func orthogonal(to base: [Float], seed: UInt64) -> [Float] {
        var candidate = TestHelpers.randomVector(dim: dimension, seed: seed)
        let dot = zip(candidate, base).reduce(Float.zero) { $0 + ($1.0 * $1.1) }
        candidate = zip(candidate, base).map { $0.0 - dot * $0.1 }
        return TestHelpers.l2Normalize(candidate)
    }
}

private extension Array {
    func asyncMap<T>(_ transform: (Element) async throws -> T) async rethrows -> [T] {
        var result: [T] = []
        result.reserveCapacity(count)
        for element in self {
            let mapped = try await transform(element)
            result.append(mapped)
        }
        return result
    }
}
#endif
