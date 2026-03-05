import Foundation
import Testing
@testable import ContextCore

#if !targetEnvironment(simulator)
@Suite("Progressive Compression Tests")
struct ProgressiveCompressionTests {
    private let tokenCounter = ApproximateTokenCounter()

    @Test("Deficit is covered with light compression before touching all chunks")
    func deficitCoveredAtLightLevel() async throws {
        let compressor = try makeCompressor()
        let candidates = makeCandidates(count: 10, tokensPerChunk: 200)

        let results = try await compressor.compress(candidates: candidates, tokenDeficit: 100)
        let totalSaved = results.reduce(into: 0) { $0 += $1.tokensSaved }

        #expect(totalSaved >= 100)
        #expect(results.contains(where: { $0.compressionLevel == .light }))
        #expect(results.contains(where: { $0.compressionLevel == .none }))
    }

    @Test("Lowest scored chunks are compressed first")
    func lowestScoredFirst() async throws {
        let compressor = try makeCompressor()
        let scores: [Float] = [0.1, 0.3, 0.5, 0.7, 0.9]
        let candidates = makeCandidates(scores: scores, tokensPerChunk: 200)

        let results = try await compressor.compress(candidates: candidates, tokenDeficit: 200)

        #expect(results[0].compressionLevel != .none)
        #expect(results[4].compressionLevel == .none)
    }

    @Test("Compression escalates to dropped when light and heavy are insufficient")
    func levelEscalationToDropped() async throws {
        let compressor = try makeCompressor()
        let candidates = makeCandidates(count: 1, tokensPerChunk: 200)

        let results = try await compressor.compress(candidates: candidates, tokenDeficit: 180)

        #expect(results.count == 1)
        #expect(results[0].compressionLevel == .dropped)
        #expect(results[0].compressedTokens == 0)
    }

    @Test("Light compression is used when it already satisfies deficit")
    func lightBeforeHeavy() async throws {
        let compressor = try makeCompressor()
        let candidates = makeCandidates(count: 1, tokensPerChunk: 200)

        let results = try await compressor.compress(candidates: candidates, tokenDeficit: 80)

        #expect(results.count == 1)
        #expect(results[0].compressionLevel == .light)
    }

    @Test("Multiple chunks are compressed for medium deficits")
    func multipleChunksCompressed() async throws {
        let compressor = try makeCompressor()
        let candidates = makeCandidates(count: 5, tokensPerChunk: 200)

        let results = try await compressor.compress(candidates: candidates, tokenDeficit: 400)
        let compressedCount = results.filter { $0.compressionLevel != .none }.count

        #expect(compressedCount >= 2)
        #expect(compressedCount < 5)
    }

    @Test("All chunks are dropped when deficit exceeds total recoverable tokens")
    func allChunksDropped() async throws {
        let compressor = try makeCompressor()
        let candidates = makeCandidates(count: 3, tokensPerChunk: 100)

        let results = try await compressor.compress(candidates: candidates, tokenDeficit: 500)

        #expect(results.allSatisfy { $0.compressionLevel == .dropped })
        #expect(results.reduce(into: 0) { $0 += $1.tokensSaved } > 0)
    }

    @Test("Zero deficit leaves all chunks unchanged")
    func zeroDeficitNoOp() async throws {
        let compressor = try makeCompressor()
        let candidates = makeCandidates(count: 5, tokensPerChunk: 150)

        let results = try await compressor.compress(candidates: candidates, tokenDeficit: 0)

        #expect(results.count == candidates.count)
        #expect(results.allSatisfy { $0.compressionLevel == .none })
        for (result, candidate) in zip(results, candidates) {
            #expect(result.compressedContent == candidate.chunk.content)
        }
    }

    @Test("Heavy level is selected when light is insufficient and heavy is sufficient")
    func heavyLevelSelected() async throws {
        let compressor = try makeCompressor()
        let candidates = makeCandidates(count: 1, tokensPerChunk: 200)

        let results = try await compressor.compress(candidates: candidates, tokenDeficit: 130)

        #expect(results.count == 1)
        #expect(results[0].compressionLevel == .heavy)
    }

    @Test("Token savings accounting covers deficit when enough content exists")
    func tokenSavingsAccounting() async throws {
        let compressor = try makeCompressor()
        let candidates = makeCandidates(count: 8, tokensPerChunk: 200)

        let deficit = 450
        let results = try await compressor.compress(candidates: candidates, tokenDeficit: deficit)
        let totalSaved = results.reduce(into: 0) { $0 += $1.tokensSaved }

        #expect(totalSaved >= deficit)
    }

    @Test("toContextChunk returns nil for dropped results")
    func toContextChunkDropped() {
        let chunk = makeChunkWithTokens(120, seed: 11)
        let result = ProgressiveCompressionResult(
            originalChunk: chunk,
            compressedContent: nil,
            compressionLevel: .dropped,
            originalTokens: 120,
            compressedTokens: 0,
            tokensSaved: 120
        )

        let context = result.toContextChunk(score: 0.1, source: .episodic)
        #expect(context == nil)
    }

    @Test("toContextChunk returns compressed chunk for light level")
    func toContextChunkLight() {
        let chunk = makeChunkWithTokens(120, seed: 12)
        let result = ProgressiveCompressionResult(
            originalChunk: chunk,
            compressedContent: "compressed sentence output",
            compressionLevel: .light,
            originalTokens: 120,
            compressedTokens: 40,
            tokensSaved: 80
        )

        let context = result.toContextChunk(score: 0.6, source: .episodic)

        #expect(context != nil)
        #expect(context?.content == "compressed sentence output")
        #expect(context?.compressionLevel == .light)
        #expect(context?.tokenCount == 40)
    }

    @Test("Result ordering matches input candidate ordering")
    func orderUnchanged() async throws {
        let compressor = try makeCompressor()
        let candidates = makeCandidates(count: 5, tokensPerChunk: 200)

        let results = try await compressor.compress(candidates: candidates, tokenDeficit: 300)
        let inputIDs = candidates.map { $0.chunk.id }
        let outputIDs = results.map { $0.originalChunk.id }

        #expect(outputIDs == inputIDs)
    }

    @Test("WindowPacker applies progressive compression to over-budget tail")
    func windowPackerIntegration() async throws {
        let provider = DeterministicEmbeddingProvider()
        let delegate = TargetHonoringCompressionDelegate()
        let engine = try CompressionEngine(
            embeddingProvider: provider,
            tokenCounter: tokenCounter,
            compressionDelegate: delegate
        )
        let packer = WindowPacker(
            compressionEngine: engine,
            tokenCounter: tokenCounter,
            minimumChunkSize: 50,
            recentTurnsGuaranteed: 0
        )

        let high = makeChunkWithTokens(200, seed: 901)
        let mid = makeChunkWithTokens(200, seed: 902)
        let low = makeChunkWithTokens(200, seed: 903)

        let scoredMemory: [(chunk: MemoryChunk, score: Float)] = [
            (chunk: high, score: 0.9),
            (chunk: mid, score: 0.8),
            (chunk: low, score: 0.1),
        ]

        let window = try await packer.pack(
            systemPrompt: nil,
            recentTurns: [],
            scoredMemory: scoredMemory,
            budget: 250
        )

        #expect(window.totalTokens <= 250)

        let highChunk = window.chunks.first(where: { $0.id == high.id })
        #expect(highChunk != nil)
        #expect(highChunk?.compressionLevel == CompressionLevel.none)
        #expect(!window.chunks.contains(where: { $0.id == low.id }))
    }

    private func makeCompressor() throws -> ProgressiveCompressor {
        let provider = DeterministicEmbeddingProvider()
        let delegate = TargetHonoringCompressionDelegate()
        let engine = try CompressionEngine(
            embeddingProvider: provider,
            tokenCounter: tokenCounter,
            compressionDelegate: delegate
        )
        return ProgressiveCompressor(compressionEngine: engine, tokenCounter: tokenCounter)
    }

    private func makeCandidates(count: Int, tokensPerChunk: Int) -> [(chunk: MemoryChunk, evictionScore: Float)] {
        (0..<count).map { index in
            let score = Float(index + 1) / Float(max(count, 1) + 1)
            return (
                chunk: makeChunkWithTokens(tokensPerChunk, seed: index + 1),
                evictionScore: score
            )
        }
        .sorted { $0.evictionScore < $1.evictionScore }
    }

    private func makeCandidates(scores: [Float], tokensPerChunk: Int) -> [(chunk: MemoryChunk, evictionScore: Float)] {
        zip(scores.indices, scores).map { pair in
            (
                chunk: makeChunkWithTokens(tokensPerChunk, seed: pair.0 + 100),
                evictionScore: pair.1
            )
        }
        .sorted { $0.evictionScore < $1.evictionScore }
    }

    private func makeChunkWithTokens(_ targetTokens: Int, seed: Int) -> MemoryChunk {
        let targetWords = max(1, Int(ceil(Double(targetTokens) / 1.3)))
        let words = (0..<targetWords).map { index in "chunk\(seed)word\(index)" }
        let content = words.joined(separator: " ") + "."

        return MemoryChunk(
            content: content,
            embedding: TestHelpers.randomVector(dim: 384, seed: UInt64(seed)),
            type: .episodic,
            retentionScore: 0.5,
            sourceSessionID: UUID(),
            metadata: [:]
        )
    }
}

private struct DeterministicEmbeddingProvider: EmbeddingProvider, Sendable {
    let dimension: Int = 384

    func embed(_ text: String) async throws -> [Float] {
        var hasher = Hasher()
        hasher.combine(text)
        let seed = UInt64(bitPattern: Int64(hasher.finalize()))
        return TestHelpers.randomVector(dim: dimension, seed: seed)
    }

    func embedBatch(_ texts: [String]) async throws -> [[Float]] {
        try await texts.asyncMap { text in
            try await embed(text)
        }
    }
}

private final class TargetHonoringCompressionDelegate: CompressionDelegate, @unchecked Sendable {
    func compress(_ text: String, targetTokens: Int) async throws -> String {
        let words = text.split(separator: " ")
        guard !words.isEmpty else {
            return text
        }

        let safeTarget = max(1, targetTokens)
        let wordBudget = max(1, Int(floor(Double(safeTarget) / 1.3)))
        return words.prefix(wordBudget).joined(separator: " ")
    }

    func extractFacts(from text: String) async throws -> [String] {
        [text]
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
