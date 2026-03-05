import ContextCoreTypes
import Foundation
import NaturalLanguage

public actor ExtractiveFallbackDelegate: CompressionDelegate {
    private let compressionEngine: CompressionEngine
    private let tokenCounter: any TokenCounter

    public init(
        compressionEngine: CompressionEngine,
        tokenCounter: any TokenCounter
    ) {
        self.compressionEngine = compressionEngine
        self.tokenCounter = tokenCounter
    }

    public func compress(_ text: String, targetTokens: Int) async throws -> String {
        let currentTokens = tokenCounter.count(text)
        if currentTokens <= targetTokens {
            return text
        }

        let textEmbedding = try await compressionEngine.embedForCompression(text)
        let ranked = try await compressionEngine.rankSentences(in: text, chunkEmbedding: textEmbedding)

        guard !ranked.isEmpty else {
            return text
        }

        let originalSentences = splitSentences(from: text)
        let indexedBySentence = sentenceIndicesByText(originalSentences)

        var indices = indexedBySentence
        var rankedWithIndices: [(sentence: String, importance: Float, originalIndex: Int)] = []
        rankedWithIndices.reserveCapacity(ranked.count)

        var fallbackIndex = originalSentences.count
        for pair in ranked {
            if var available = indices[pair.sentence], let index = available.first {
                available.removeFirst()
                indices[pair.sentence] = available
                rankedWithIndices.append((sentence: pair.sentence, importance: pair.importance, originalIndex: index))
            } else {
                rankedWithIndices.append((sentence: pair.sentence, importance: pair.importance, originalIndex: fallbackIndex))
                fallbackIndex += 1
            }
        }

        let sortedByImportance = rankedWithIndices.sorted { lhs, rhs in
            if lhs.importance == rhs.importance {
                return lhs.originalIndex < rhs.originalIndex
            }
            return lhs.importance > rhs.importance
        }

        var selected: [(sentence: String, importance: Float, originalIndex: Int)] = []
        var tokensUsed = 0

        for item in sortedByImportance {
            let sentenceTokens = tokenCounter.count(item.sentence)
            if tokensUsed + sentenceTokens > targetTokens {
                continue
            }

            selected.append(item)
            tokensUsed += sentenceTokens
        }

        if selected.isEmpty, let topSentence = sortedByImportance.first {
            return topSentence.sentence
        }

        selected.sort { $0.originalIndex < $1.originalIndex }
        return selected.map(\.sentence).joined(separator: " ")
    }

    public func extractFacts(from text: String) async throws -> [String] {
        let tokenizer = NLTokenizer(unit: .sentence)
        tokenizer.string = text

        var facts: [String] = []
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let sentence = String(text[range]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !sentence.isEmpty {
                facts.append(sentence)
            }
            return true
        }

        return facts
    }

    private func splitSentences(from text: String) -> [String] {
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

    private func sentenceIndicesByText(_ sentences: [String]) -> [String: [Int]] {
        var output: [String: [Int]] = [:]
        output.reserveCapacity(sentences.count)

        for (index, sentence) in sentences.enumerated() {
            output[sentence, default: []].append(index)
        }

        return output
    }
}
