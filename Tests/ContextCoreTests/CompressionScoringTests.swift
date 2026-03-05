import Foundation
import NaturalLanguage
import Testing
@testable import ContextCore

#if !targetEnvironment(simulator)
@Suite("Compression Scoring Tests")
struct CompressionScoringTests {
    @Test("Off-topic sentence ranks last")
    func offTopicSentenceRanksLast() async throws {
        let provider = SemanticMockEmbeddingProvider()
        let engine = try CompressionEngine(
            embeddingProvider: provider,
            tokenCounter: ApproximateTokenCounter()
        )

        let chunk = "Swift is a compiled language. It uses LLVM for optimization. It supports generics and protocols. The weather is nice today. Swift was created by Apple."
        let sentences = splitSentences(chunk)
        let embeddings = try await provider.embedBatch(sentences)
        let chunkEmbedding = TestHelpers.l2Normalize(mean(embeddings))

        let ranked = try await engine.rankSentences(in: chunk, chunkEmbedding: chunkEmbedding)
        #expect(ranked.last?.sentence == "The weather is nice today.")
    }

    @Test("Representative sentence ranks first")
    func representativeSentenceRanksFirst() async throws {
        let provider = SemanticMockEmbeddingProvider()
        let engine = try CompressionEngine(
            embeddingProvider: provider,
            tokenCounter: ApproximateTokenCounter()
        )

        let summary = "ARC determines when objects should be deallocated."
        let chunk = [
            "Automatic Reference Counting manages object lifetimes in Swift.",
            "Strong reference cycles can cause memory leaks.",
            "Use weak references to break cycles.",
            summary,
        ].joined(separator: " ")

        let chunkEmbedding = try await provider.embed(summary)
        let ranked = try await engine.rankSentences(in: chunk, chunkEmbedding: chunkEmbedding)

        #expect(ranked.first?.sentence == summary)
    }

    @Test("Equal sentences produce equal importance")
    func equalSentencesEqualImportance() async throws {
        let provider = SemanticMockEmbeddingProvider()
        let engine = try CompressionEngine(
            embeddingProvider: provider,
            tokenCounter: ApproximateTokenCounter()
        )

        let sentence = "Swift uses value semantics in structs."
        let chunk = "\(sentence) \(sentence) \(sentence)"
        let chunkEmbedding = try await provider.embed(sentence)

        let ranked = try await engine.rankSentences(in: chunk, chunkEmbedding: chunkEmbedding)
        let scores = ranked.map(\.importance)

        for index in 0..<(scores.count - 1) {
            #expect(abs(scores[index] - scores[index + 1]) < 1e-4)
        }
    }

    @Test("Sentence splitting detects three sentences")
    func sentenceSplitting() async throws {
        let provider = SemanticMockEmbeddingProvider()
        let engine = try CompressionEngine(
            embeddingProvider: provider,
            tokenCounter: ApproximateTokenCounter()
        )

        let chunk = "First sentence. Second sentence. Third!"
        let embedding = try await provider.embed("First sentence.")

        let ranked = try await engine.rankSentences(in: chunk, chunkEmbedding: embedding)
        #expect(ranked.count == 3)
    }

    @Test("GPU vs CPU parity for sentence importance")
    func gpuVsCpuParity() async throws {
        let provider = SemanticMockEmbeddingProvider()
        let engine = try CompressionEngine(
            embeddingProvider: provider,
            tokenCounter: ApproximateTokenCounter()
        )

        let sentences = [
            "Swift actors isolate mutable state.",
            "ARC uses reference counts for lifecycle.",
            "Use weak references to avoid cycles.",
            "Async await improves readability.",
            "The oven should be preheated.",
            "Rainfall is expected this afternoon.",
            "Protocol extensions add shared behavior.",
            "Apple introduced Swift in 2014.",
            "Whisk the eggs thoroughly.",
            "Task cancellation is cooperative in Swift concurrency.",
        ]

        let chunk = sentences.joined(separator: " ")
        let sentenceEmbeddings = try await provider.embedBatch(sentences)
        let chunkEmbedding = TestHelpers.l2Normalize(mean(sentenceEmbeddings))

        let cpu = CPUReference.sentenceImportance(
            sentenceEmbeddings: sentenceEmbeddings,
            chunkEmbedding: chunkEmbedding
        )
        let cpuBySentence = Dictionary(uniqueKeysWithValues: zip(sentences, cpu))

        let ranked = try await engine.rankSentences(in: chunk, chunkEmbedding: chunkEmbedding)

        let gpuScores = ranked.map(\.importance)
        let expected = ranked.compactMap { cpuBySentence[$0.sentence] }
        let maxError = TestHelpers.maxAbsError(gpuScores, expected)

        #expect(maxError < 1e-4)
    }

    @Test("Single sentence has importance 1.0")
    func singleSentence() async throws {
        let provider = SemanticMockEmbeddingProvider()
        let engine = try CompressionEngine(
            embeddingProvider: provider,
            tokenCounter: ApproximateTokenCounter()
        )

        let sentence = "Swift is designed for safety and speed."
        let chunkEmbedding = try await provider.embed(sentence)

        let ranked = try await engine.rankSentences(in: sentence, chunkEmbedding: chunkEmbedding)

        #expect(ranked.count == 1)
        #expect(ranked[0].sentence == sentence)
        #expect(abs(ranked[0].importance - 1.0) < 1e-4)
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

    private func mean(_ vectors: [[Float]]) -> [Float] {
        guard let first = vectors.first else { return [] }
        var accumulator = [Float](repeating: 0, count: first.count)

        for vector in vectors {
            for index in vector.indices {
                accumulator[index] += vector[index]
            }
        }

        let count = Float(vectors.count)
        return accumulator.map { $0 / count }
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
        if lower.contains("swift") || lower.contains("arc") || lower.contains("reference") || lower.contains("actor") || lower.contains("protocol") || lower.contains("task") || lower.contains("async") {
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
