import ContextCoreTypes
import Foundation
import Metal
import NaturalLanguage

public actor CompressionEngine {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let sentenceImportancePipeline: MTLComputePipelineState
    private let embeddingProvider: any EmbeddingProvider

    public init(
        device: MTLDevice? = nil,
        embeddingProvider: any EmbeddingProvider
    ) throws {
        self.device = try device ?? MetalContext.device()
        self.commandQueue = try MetalContext.commandQueue(device: self.device)
        self.embeddingProvider = embeddingProvider

        let library = try MetalContext.library(device: self.device)
        guard let function = library.makeFunction(name: "sentence_importance") else {
            throw ContextCoreError.compressionFailed("Missing sentence_importance function")
        }

        self.sentenceImportancePipeline = try self.device.makeComputePipelineState(function: function)
    }

    public func rankSentences(
        in chunk: String,
        chunkEmbedding: [Float]
    ) async throws -> [(sentence: String, importance: Float)] {
        let sentences = splitSentences(from: chunk)
        guard !sentences.isEmpty else {
            return []
        }

        let sentenceEmbeddings = try await embeddingProvider.embedBatch(sentences)
        let dim = chunkEmbedding.count

        guard sentenceEmbeddings.count == sentences.count else {
            throw ContextCoreError.embeddingFailed("embedBatch returned mismatched result count")
        }

        guard sentenceEmbeddings.allSatisfy({ $0.count == dim }) else {
            throw ContextCoreError.dimensionMismatch(expected: dim, got: sentenceEmbeddings.first(where: { $0.count != dim })?.count ?? 0)
        }

        let flattened = sentenceEmbeddings.flatMap { $0 }
        let m = sentences.count
        var output = [Float](repeating: 0, count: m)

        guard let sentenceBuffer = device.makeBuffer(from: flattened),
              let chunkBuffer = device.makeBuffer(from: chunkEmbedding),
              let outputBuffer = device.makeBuffer(from: output),
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            throw ContextCoreError.compressionFailed("Failed to allocate compression buffers")
        }

        var dim32 = UInt32(dim)
        var m32 = UInt32(m)

        guard let dimBuffer = device.makeBuffer(bytes: &dim32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let mBuffer = device.makeBuffer(bytes: &m32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)
        else {
            throw ContextCoreError.compressionFailed("Failed to create compression constants")
        }

        encoder.setComputePipelineState(sentenceImportancePipeline)
        encoder.setBuffer(sentenceBuffer, offset: 0, index: 0)
        encoder.setBuffer(chunkBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBuffer(dimBuffer, offset: 0, index: 3)
        encoder.setBuffer(mBuffer, offset: 0, index: 4)

        let threads = MetalContext.threadsPerThreadgroup(pipeline: sentenceImportancePipeline, count: m)
        let groups = MetalContext.threadgroups(threadsPerThreadgroup: threads, count: m)
        encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
        encoder.endEncoding()

        try await MetalContext.awaitCompletion(of: commandBuffer)

        let raw = outputBuffer.contents().bindMemory(to: Float.self, capacity: m)
        output = Array(UnsafeBufferPointer(start: raw, count: m))

        return zip(sentences, output)
            .map { (sentence: $0.0, importance: $0.1) }
            .sorted(by: { $0.importance > $1.importance })
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
}
