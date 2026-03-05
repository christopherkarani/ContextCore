import ContextCoreTypes
import Foundation
import Metal
import NaturalLanguage

public actor CompressionEngine {
    public init(
        device: MTLDevice? = nil,
        embeddingProvider: any EmbeddingProvider
    ) throws {
        _ = try device ?? MetalContext.device()
        _ = embeddingProvider
    }

    public func rankSentences(
        in chunk: String,
        chunkEmbedding: [Float]
    ) async throws -> [(sentence: String, importance: Float)] {
        _ = chunk
        _ = chunkEmbedding
        return []
    }
}
