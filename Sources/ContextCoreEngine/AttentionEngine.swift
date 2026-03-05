import ContextCoreTypes
import Foundation
import Metal

public actor AttentionEngine {
    public init(device: MTLDevice? = nil) throws {
        _ = try device ?? MetalContext.device()
    }

    public func computeCentrality(
        embeddings: [[Float]]
    ) async throws -> [Float] {
        _ = embeddings
        return []
    }

    public func scoreWindowForEviction(
        taskQuery: [Float],
        windowChunks: [MemoryChunk],
        relevanceWeight: Float = 0.6,
        centralityWeight: Float = 0.4
    ) async throws -> [(chunk: MemoryChunk, evictionScore: Float)] {
        _ = taskQuery
        _ = windowChunks
        _ = relevanceWeight
        _ = centralityWeight
        return []
    }
}
