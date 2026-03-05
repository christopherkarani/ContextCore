import Foundation
import NaturalLanguage
import Testing
@testable import ContextCore

#if !targetEnvironment(simulator)
@Suite("Compression Engine Tests")
struct CompressionEngineTests {
    @Test("Extractive compress meets token budget")
    func extractiveCompressBudgetMet() async throws {
        let provider = SemanticMockEmbeddingProvider()
        let engine = try CompressionEngine(embeddingProvider: provider)
        let delegate = ExtractiveFallbackDelegate(
            compressionEngine: engine,
            tokenCounter: ApproximateTokenCounter()
        )

        let text = testParagraph()
        let compressed = try await delegate.compress(text, targetTokens: 150)

        #expect(ApproximateTokenCounter().count(compressed) <= 150)
    }

    @Test("Extractive compress keeps highest-ranked sentence")
    func extractiveCompressKeepsBestSentence() async throws {
        let provider = SemanticMockEmbeddingProvider()
        let engine = try CompressionEngine(embeddingProvider: provider)
        let delegate = ExtractiveFallbackDelegate(
            compressionEngine: engine,
            tokenCounter: ApproximateTokenCounter()
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
        let engine = try CompressionEngine(embeddingProvider: provider)
        let delegate = ExtractiveFallbackDelegate(
            compressionEngine: engine,
            tokenCounter: ApproximateTokenCounter()
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
        let engine = try CompressionEngine(embeddingProvider: provider)
        let delegate = ExtractiveFallbackDelegate(
            compressionEngine: engine,
            tokenCounter: ApproximateTokenCounter()
        )

        let text = "Swift is safe and fast."
        let compressed = try await delegate.compress(text, targetTokens: 200)

        #expect(compressed == text)
    }

    @Test("Extractive compress returns one sentence when sentence exceeds target")
    func extractiveCompressSingleSentenceOverTarget() async throws {
        let provider = SemanticMockEmbeddingProvider()
        let engine = try CompressionEngine(embeddingProvider: provider)
        let delegate = ExtractiveFallbackDelegate(
            compressionEngine: engine,
            tokenCounter: ApproximateTokenCounter()
        )

        let sentence = makeLongSentence(approximateTokens: 300, stem: "swift")
        let compressed = try await delegate.compress(sentence, targetTokens: 100)

        #expect(compressed == sentence)
    }

    @Test("extractFacts splits into standalone sentences")
    func extractFactsSentenceSplit() async throws {
        let provider = SemanticMockEmbeddingProvider()
        let engine = try CompressionEngine(embeddingProvider: provider)
        let delegate = ExtractiveFallbackDelegate(
            compressionEngine: engine,
            tokenCounter: ApproximateTokenCounter()
        )

        let facts = try await delegate.extractFacts(from: "First fact. Second fact. Third fact!")

        #expect(facts == ["First fact.", "Second fact.", "Third fact!"])
    }

    @Test("extractFacts returns empty array for empty input")
    func extractFactsEmpty() async throws {
        let provider = SemanticMockEmbeddingProvider()
        let engine = try CompressionEngine(embeddingProvider: provider)
        let delegate = ExtractiveFallbackDelegate(
            compressionEngine: engine,
            tokenCounter: ApproximateTokenCounter()
        )

        let facts = try await delegate.extractFacts(from: "")

        #expect(facts.isEmpty)
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
