import ContextCoreTypes
import Foundation
import Metal

public actor ConsolidationEngine {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pairwisePipeline: MTLComputePipelineState
    private let pairwiseTiledPipeline: MTLComputePipelineState
    private let mergeCandidatePipeline: MTLComputePipelineState
    private let embeddingProvider: any EmbeddingProvider

    public init(
        device: MTLDevice? = nil,
        embeddingProvider: any EmbeddingProvider
    ) throws {
        self.device = try device ?? MetalContext.device()
        self.commandQueue = try MetalContext.commandQueue(device: self.device)
        self.embeddingProvider = embeddingProvider

        let library = try MetalContext.library(device: self.device)

        guard let pairwiseFunction = library.makeFunction(name: "pairwise_similarity") else {
            throw ContextCoreError.compressionFailed("Missing pairwise_similarity function")
        }
        guard let pairwiseTiledFunction = library.makeFunction(name: "pairwise_similarity_tiled") else {
            throw ContextCoreError.compressionFailed("Missing pairwise_similarity_tiled function")
        }
        guard let mergeFunction = library.makeFunction(name: "find_merge_candidates") else {
            throw ContextCoreError.compressionFailed("Missing find_merge_candidates function")
        }

        self.pairwisePipeline = try self.device.makeComputePipelineState(function: pairwiseFunction)
        self.pairwiseTiledPipeline = try self.device.makeComputePipelineState(function: pairwiseTiledFunction)
        self.mergeCandidatePipeline = try self.device.makeComputePipelineState(function: mergeFunction)
    }

    public func findDuplicates(
        in store: any ConsolidationEpisodicStore,
        threshold: Float = 0.92
    ) async throws -> [(UUID, UUID)] {
        let chunks = await store.allChunks()
        guard chunks.count > 1 else {
            return []
        }

        let dim = try validateDimensions(chunks)
        let pairs: [(Int, Int)]
        if chunks.count <= 2048 {
            pairs = try await findCandidatesSimple(chunks: chunks, dim: dim, threshold: threshold)
        } else {
            pairs = try await findCandidatesTiled(chunks: chunks, dim: dim, threshold: threshold, tileSize: 512)
        }

        return pairs.map { (chunks[$0.0].id, chunks[$0.1].id) }
    }

    public func pairwiseSimilarity(
        embeddings: [[Float]]
    ) async throws -> [[Float]] {
        guard !embeddings.isEmpty else {
            return []
        }
        if embeddings.count == 1 {
            return [[1.0]]
        }

        let dim = embeddings[0].count
        guard embeddings.allSatisfy({ $0.count == dim }) else {
            throw ContextCoreError.dimensionMismatch(
                expected: dim,
                got: embeddings.first(where: { $0.count != dim })?.count ?? 0
            )
        }

        let n = embeddings.count
        var matrix = Array(repeating: Array(repeating: Float.zero, count: n), count: n)
        for index in 0..<n {
            matrix[index][index] = 1.0
        }

        if n <= 2048 {
            let flattened = embeddings.flatMap { $0 }
            let upper = try await computePairwiseUpperSimple(
                flattenedEmbeddings: flattened,
                dim: dim,
                n: n
            )

            for i in 0..<n {
                for j in (i + 1)..<n {
                    let value = upper[(i * n) + j]
                    matrix[i][j] = value
                    matrix[j][i] = value
                }
            }
            return matrix
        }

        let flattened = embeddings.flatMap { $0 }
        guard let embeddingsBuffer = device.makeBuffer(from: flattened) else {
            throw ContextCoreError.compressionFailed("Failed to allocate embeddings buffer")
        }

        for rowTile in stride(from: 0, to: n, by: 512) {
            let rowCount = min(512, n - rowTile)
            for colTile in stride(from: rowTile, to: n, by: 512) {
                let colCount = min(512, n - colTile)
                let tile = try await computeTileSimilarities(
                    embeddingsBuffer: embeddingsBuffer,
                    dim: dim,
                    n: n,
                    rowOffset: rowTile,
                    colOffset: colTile,
                    rowCount: rowCount,
                    colCount: colCount
                )

                for localRow in 0..<rowCount {
                    let globalRow = rowTile + localRow
                    for localCol in 0..<colCount {
                        let globalCol = colTile + localCol
                        guard globalCol > globalRow else {
                            continue
                        }

                        let value = tile[(localRow * colCount) + localCol]
                        matrix[globalRow][globalCol] = value
                        matrix[globalCol][globalRow] = value
                    }
                }
            }
        }

        return matrix
    }

    private func validateDimensions(_ chunks: [MemoryChunk]) throws -> Int {
        guard let dim = chunks.first?.embedding.count, dim > 0 else {
            throw ContextCoreError.dimensionMismatch(expected: 1, got: 0)
        }
        guard chunks.allSatisfy({ $0.embedding.count == dim }) else {
            throw ContextCoreError.dimensionMismatch(
                expected: dim,
                got: chunks.first(where: { $0.embedding.count != dim })?.embedding.count ?? 0
            )
        }
        return dim
    }

    private func findCandidatesSimple(
        chunks: [MemoryChunk],
        dim: Int,
        threshold: Float
    ) async throws -> [(Int, Int)] {
        let n = chunks.count
        let flattened = chunks.flatMap(\.embedding)
        var candidateCountStorage: UInt32 = 0
        let maxCandidates = max(1, n * 10)

        let similarity = try await computePairwiseUpperSimple(
            flattenedEmbeddings: flattened,
            dim: dim,
            n: n
        )

        guard let similarityBuffer = device.makeBuffer(from: similarity),
              let candidatesBuffer = device.makeBuffer(
                length: maxCandidates * MemoryLayout<SIMD2<UInt32>>.stride,
                options: .storageModeShared
              ),
              let candidateCountBuffer = device.makeBuffer(
                bytes: &candidateCountStorage,
                length: MemoryLayout<UInt32>.stride,
                options: .storageModeShared
              ),
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let mergeEncoder = commandBuffer.makeComputeCommandEncoder()
        else {
            throw ContextCoreError.compressionFailed("Failed to allocate consolidation buffers")
        }

        var n32 = UInt32(n)
        var thresholdValue = threshold
        var maxCandidates32 = UInt32(maxCandidates)

        guard let nBuffer = device.makeBuffer(bytes: &n32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let thresholdBuffer = device.makeBuffer(bytes: &thresholdValue, length: MemoryLayout<Float>.stride, options: .storageModeShared),
              let maxCandidatesBuffer = device.makeBuffer(bytes: &maxCandidates32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)
        else {
            throw ContextCoreError.compressionFailed("Failed to allocate consolidation constants")
        }

        mergeEncoder.setComputePipelineState(mergeCandidatePipeline)
        mergeEncoder.setBuffer(similarityBuffer, offset: 0, index: 0)
        mergeEncoder.setBuffer(candidatesBuffer, offset: 0, index: 1)
        mergeEncoder.setBuffer(candidateCountBuffer, offset: 0, index: 2)
        mergeEncoder.setBuffer(thresholdBuffer, offset: 0, index: 3)
        mergeEncoder.setBuffer(nBuffer, offset: 0, index: 4)
        mergeEncoder.setBuffer(maxCandidatesBuffer, offset: 0, index: 5)
        dispatch2D(
            encoder: mergeEncoder,
            width: n,
            height: n,
            pipeline: mergeCandidatePipeline
        )
        mergeEncoder.endEncoding()

        try await awaitCompletion(commandBuffer)

        let countPointer = candidateCountBuffer.contents().bindMemory(to: UInt32.self, capacity: 1)
        let rawCount = Int(countPointer[0])
        let boundedCount = min(rawCount, maxCandidates)
        if rawCount > maxCandidates {
            print("ConsolidationEngine warning: merge candidate buffer capped at \(maxCandidates) of \(rawCount)")
        }

        let pairPointer = candidatesBuffer.contents().bindMemory(to: SIMD2<UInt32>.self, capacity: boundedCount)
        return (0..<boundedCount).map { idx in
            let pair = pairPointer[idx]
            return (Int(pair.x), Int(pair.y))
        }
    }

    private func computePairwiseUpperSimple(
        flattenedEmbeddings: [Float],
        dim: Int,
        n: Int
    ) async throws -> [Float] {
        var output = [Float](repeating: 0, count: n * n)

        guard let embeddingsBuffer = device.makeBuffer(from: flattenedEmbeddings),
              let outputBuffer = device.makeBuffer(from: output),
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            throw ContextCoreError.compressionFailed("Failed to allocate pairwise buffers")
        }

        var dim32 = UInt32(dim)
        var n32 = UInt32(n)

        guard let dimBuffer = device.makeBuffer(bytes: &dim32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let nBuffer = device.makeBuffer(bytes: &n32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)
        else {
            throw ContextCoreError.compressionFailed("Failed to allocate pairwise constants")
        }

        encoder.setComputePipelineState(pairwisePipeline)
        encoder.setBuffer(embeddingsBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.setBuffer(dimBuffer, offset: 0, index: 2)
        encoder.setBuffer(nBuffer, offset: 0, index: 3)
        dispatch2D(
            encoder: encoder,
            width: n,
            height: n,
            pipeline: pairwisePipeline
        )
        encoder.endEncoding()

        try await awaitCompletion(commandBuffer)

        let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: output.count)
        output = Array(UnsafeBufferPointer(start: pointer, count: output.count))
        return output
    }

    private func findCandidatesTiled(
        chunks: [MemoryChunk],
        dim: Int,
        threshold: Float,
        tileSize: Int
    ) async throws -> [(Int, Int)] {
        let n = chunks.count
        let flattened = chunks.flatMap(\.embedding)

        guard let embeddingsBuffer = device.makeBuffer(from: flattened) else {
            throw ContextCoreError.compressionFailed("Failed to allocate embeddings buffer for tiled path")
        }

        var candidates: [(Int, Int)] = []

        for rowTile in stride(from: 0, to: n, by: tileSize) {
            let rowCount = min(tileSize, n - rowTile)
            for colTile in stride(from: rowTile, to: n, by: tileSize) {
                let colCount = min(tileSize, n - colTile)
                let tileSimilarities = try await computeTileSimilarities(
                    embeddingsBuffer: embeddingsBuffer,
                    dim: dim,
                    n: n,
                    rowOffset: rowTile,
                    colOffset: colTile,
                    rowCount: rowCount,
                    colCount: colCount
                )

                for localRow in 0..<rowCount {
                    let globalRow = rowTile + localRow
                    for localCol in 0..<colCount {
                        let globalCol = colTile + localCol
                        guard globalCol > globalRow else {
                            continue
                        }
                        let score = tileSimilarities[(localRow * colCount) + localCol]
                        if score > threshold {
                            candidates.append((globalRow, globalCol))
                        }
                    }
                }
            }
        }

        return candidates
    }

    private func computeTileSimilarities(
        embeddingsBuffer: MTLBuffer,
        dim: Int,
        n: Int,
        rowOffset: Int,
        colOffset: Int,
        rowCount: Int,
        colCount: Int
    ) async throws -> [Float] {
        var output = [Float](repeating: 0, count: rowCount * colCount)

        guard let tileBuffer = device.makeBuffer(from: output),
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            throw ContextCoreError.compressionFailed("Failed to allocate tiled consolidation buffers")
        }

        var dim32 = UInt32(dim)
        var n32 = UInt32(n)
        var rowOffset32 = UInt32(rowOffset)
        var colOffset32 = UInt32(colOffset)
        var rowCount32 = UInt32(rowCount)
        var colCount32 = UInt32(colCount)

        guard let dimBuffer = device.makeBuffer(bytes: &dim32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let nBuffer = device.makeBuffer(bytes: &n32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let rowOffsetBuffer = device.makeBuffer(bytes: &rowOffset32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let colOffsetBuffer = device.makeBuffer(bytes: &colOffset32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let rowCountBuffer = device.makeBuffer(bytes: &rowCount32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let colCountBuffer = device.makeBuffer(bytes: &colCount32, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)
        else {
            throw ContextCoreError.compressionFailed("Failed to allocate tiled consolidation constants")
        }

        encoder.setComputePipelineState(pairwiseTiledPipeline)
        encoder.setBuffer(embeddingsBuffer, offset: 0, index: 0)
        encoder.setBuffer(tileBuffer, offset: 0, index: 1)
        encoder.setBuffer(dimBuffer, offset: 0, index: 2)
        encoder.setBuffer(nBuffer, offset: 0, index: 3)
        encoder.setBuffer(rowOffsetBuffer, offset: 0, index: 4)
        encoder.setBuffer(colOffsetBuffer, offset: 0, index: 5)
        encoder.setBuffer(rowCountBuffer, offset: 0, index: 6)
        encoder.setBuffer(colCountBuffer, offset: 0, index: 7)

        dispatch2D(
            encoder: encoder,
            width: rowCount,
            height: colCount,
            pipeline: pairwiseTiledPipeline
        )
        encoder.endEncoding()

        try await awaitCompletion(commandBuffer)

        let pointer = tileBuffer.contents().bindMemory(to: Float.self, capacity: output.count)
        output = Array(UnsafeBufferPointer(start: pointer, count: output.count))
        return output
    }

    private func dispatch2D(
        encoder: MTLComputeCommandEncoder,
        width: Int,
        height: Int,
        pipeline: MTLComputePipelineState
    ) {
        let maxThreads = max(1, pipeline.maxTotalThreadsPerThreadgroup)
        let threadWidth = min(16, maxThreads)
        let threadHeight = max(1, min(16, maxThreads / threadWidth))

        let threadsPerThreadgroup = MTLSize(width: threadWidth, height: threadHeight, depth: 1)
        let groups = MTLSize(
            width: (width + threadWidth - 1) / threadWidth,
            height: (height + threadHeight - 1) / threadHeight,
            depth: 1
        )

        encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threadsPerThreadgroup)
    }

    private func awaitCompletion(_ commandBuffer: MTLCommandBuffer) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            commandBuffer.addCompletedHandler { buffer in
                if let error = buffer.error {
                    continuation.resume(
                        throwing: ContextCoreError.compressionFailed(
                            "Metal command failed: \(error.localizedDescription)"
                        )
                    )
                    return
                }
                continuation.resume(returning: ())
            }
            commandBuffer.commit()
        }
    }
}
