import ContextCoreTypes
import Foundation
import Metal

public actor ScoringEngine {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let relevancePipeline: MTLComputePipelineState
    private let topkPipeline: MTLComputePipelineState
    private let recencyPipeline: MTLComputePipelineState

    public init(device: MTLDevice? = nil) throws {
        self.device = try device ?? MetalContext.device()
        self.commandQueue = try MetalContext.commandQueue(device: self.device)

        let library = try MetalContext.library(device: self.device)

        guard let relevanceFunction = library.makeFunction(name: "relevance_score") else {
            throw ContextCoreError.compressionFailed("Missing relevance_score function")
        }

        guard let topkFunction = library.makeFunction(name: "topk_indices") else {
            throw ContextCoreError.compressionFailed("Missing topk_indices function")
        }

        guard let recencyFunction = library.makeFunction(name: "compute_recency_weights") else {
            throw ContextCoreError.compressionFailed("Missing compute_recency_weights function")
        }

        self.relevancePipeline = try self.device.makeComputePipelineState(function: relevanceFunction)
        self.topkPipeline = try self.device.makeComputePipelineState(function: topkFunction)
        self.recencyPipeline = try self.device.makeComputePipelineState(function: recencyFunction)
    }

    public func scoreChunks(
        query: [Float],
        chunks: [MemoryChunk],
        recencyWeights: [Float],
        relevanceWeight: Float = 0.7,
        recencyWeight: Float = 0.3
    ) async throws -> [(chunk: MemoryChunk, score: Float)] {
        guard !chunks.isEmpty else {
            return []
        }
        guard chunks.count == recencyWeights.count else {
            throw ContextCoreError.dimensionMismatch(expected: chunks.count, got: recencyWeights.count)
        }

        let dim = query.count
        guard chunks.allSatisfy({ $0.embedding.count == dim }) else {
            throw ContextCoreError.dimensionMismatch(expected: dim, got: chunks.first(where: { $0.embedding.count != dim })?.embedding.count ?? 0)
        }

        let flattened = chunks.flatMap(\.embedding)
        let scores = try await scoreEmbeddings(
            query: query,
            flattenedEmbeddings: flattened,
            count: chunks.count,
            dimension: dim,
            recencyWeights: recencyWeights,
            relevanceWeight: relevanceWeight,
            recencyWeight: recencyWeight
        )

        return zip(chunks, scores)
            .map { (chunk: $0.0, score: $0.1) }
            .sorted(by: { $0.score > $1.score })
    }

    public func topKIndices(
        scores: [Float],
        k: Int
    ) async throws -> [Int] {
        guard !scores.isEmpty, k > 0 else {
            return []
        }

        let cappedK = min(k, scores.count)

        guard let scoresBuffer = device.makeBuffer(from: scores) else {
            throw ContextCoreError.compressionFailed("Failed to allocate scores buffer")
        }
        var indexStorage = [UInt32](repeating: 0, count: cappedK)
        guard let indicesBuffer = device.makeBuffer(from: indexStorage) else {
            throw ContextCoreError.compressionFailed("Failed to allocate indices buffer")
        }

        var n32 = UInt32(scores.count)
        var k32 = UInt32(cappedK)

        guard let nBuffer = device.makeBuffer(bytes: &n32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let kBuffer = device.makeBuffer(bytes: &k32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            throw ContextCoreError.compressionFailed("Failed to build top-k command")
        }

        encoder.setComputePipelineState(topkPipeline)
        encoder.setBuffer(scoresBuffer, offset: 0, index: 0)
        encoder.setBuffer(indicesBuffer, offset: 0, index: 1)
        encoder.setBuffer(nBuffer, offset: 0, index: 2)
        encoder.setBuffer(kBuffer, offset: 0, index: 3)

        let threads = MTLSize(width: 1, height: 1, depth: 1)
        let groups = MTLSize(width: 1, height: 1, depth: 1)
        encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
        encoder.endEncoding()

        try await MetalContext.awaitCompletion(of: commandBuffer)

        let raw = indicesBuffer.contents().bindMemory(to: UInt32.self, capacity: cappedK)
        indexStorage = Array(UnsafeBufferPointer(start: raw, count: cappedK))

        return indexStorage.map(Int.init)
    }

    public func computeRecencyWeights(
        timestamps: [Date],
        halfLife: TimeInterval
    ) async throws -> [Float] {
        guard !timestamps.isEmpty else {
            return []
        }

        let timestampValues = timestamps.map { Float($0.timeIntervalSince1970) }
        let n = timestampValues.count
        var output = [Float](repeating: 0, count: n)

        guard let timestampBuffer = device.makeBuffer(from: timestampValues),
              let outputBuffer = device.makeBuffer(from: output),
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            throw ContextCoreError.compressionFailed("Failed to allocate recency buffers")
        }

        var currentTime = Float(Date().timeIntervalSince1970)
        var halfLifeF = Float(halfLife)
        var n32 = UInt32(n)

        guard let currentTimeBuffer = device.makeBuffer(bytes: &currentTime, length: MemoryLayout<Float>.stride, options: .storageModeShared),
              let halfLifeBuffer = device.makeBuffer(bytes: &halfLifeF, length: MemoryLayout<Float>.stride, options: .storageModeShared),
              let nBuffer = device.makeBuffer(bytes: &n32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)
        else {
            throw ContextCoreError.compressionFailed("Failed to create recency constants")
        }

        encoder.setComputePipelineState(recencyPipeline)
        encoder.setBuffer(timestampBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.setBuffer(currentTimeBuffer, offset: 0, index: 2)
        encoder.setBuffer(halfLifeBuffer, offset: 0, index: 3)
        encoder.setBuffer(nBuffer, offset: 0, index: 4)

        let threads = MetalContext.threadsPerThreadgroup(pipeline: recencyPipeline, count: n)
        let groups = MetalContext.threadgroups(threadsPerThreadgroup: threads, count: n)
        encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
        encoder.endEncoding()

        try await MetalContext.awaitCompletion(of: commandBuffer)

        let raw = outputBuffer.contents().bindMemory(to: Float.self, capacity: n)
        output = Array(UnsafeBufferPointer(start: raw, count: n))
        return output
    }

    private func scoreEmbeddings(
        query: [Float],
        flattenedEmbeddings: [Float],
        count: Int,
        dimension: Int,
        recencyWeights: [Float],
        relevanceWeight: Float,
        recencyWeight: Float
    ) async throws -> [Float] {
        var output = [Float](repeating: 0, count: count)
        var weights = SIMD2<Float>(relevanceWeight, recencyWeight)

        guard let queryBuffer = device.makeBuffer(from: query),
              let chunksBuffer = device.makeBuffer(from: flattenedEmbeddings),
              let recencyBuffer = device.makeBuffer(from: recencyWeights),
              let weightsBuffer = device.makeBuffer(bytes: &weights, length: MemoryLayout<SIMD2<Float>>.stride, options: .storageModeShared),
              let outputBuffer = device.makeBuffer(from: output),
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            throw ContextCoreError.compressionFailed("Failed to allocate scoring buffers")
        }

        var dim32 = UInt32(dimension)
        var n32 = UInt32(count)

        guard let dimBuffer = device.makeBuffer(bytes: &dim32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let nBuffer = device.makeBuffer(bytes: &n32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)
        else {
            throw ContextCoreError.compressionFailed("Failed to create scoring constants")
        }

        encoder.setComputePipelineState(relevancePipeline)
        encoder.setBuffer(queryBuffer, offset: 0, index: 0)
        encoder.setBuffer(chunksBuffer, offset: 0, index: 1)
        encoder.setBuffer(recencyBuffer, offset: 0, index: 2)
        encoder.setBuffer(weightsBuffer, offset: 0, index: 3)
        encoder.setBuffer(outputBuffer, offset: 0, index: 4)
        encoder.setBuffer(dimBuffer, offset: 0, index: 5)
        encoder.setBuffer(nBuffer, offset: 0, index: 6)

        let threads = MetalContext.threadsPerThreadgroup(pipeline: relevancePipeline, count: count)
        let groups = MetalContext.threadgroups(threadsPerThreadgroup: threads, count: count)
        encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
        encoder.endEncoding()

        try await MetalContext.awaitCompletion(of: commandBuffer)

        let raw = outputBuffer.contents().bindMemory(to: Float.self, capacity: count)
        output = Array(UnsafeBufferPointer(start: raw, count: count))
        return output
    }
}
