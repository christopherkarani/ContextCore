import ContextCore
import ContextCoreEngine
import Foundation

enum ScoringBenchmarkTrack: String, Sendable {
    case mathOnly = "math-only"
    case endToEnd = "end-to-end"
}

struct ScoringCaseResult: Sendable {
    let track: ScoringBenchmarkTrack
    let n: Int
    let gpu: BenchmarkResult
    let cpu: BenchmarkResult

    var speedup: Double {
        guard gpu.p50Ms > 0 else {
            return 0
        }
        return cpu.p50Ms / gpu.p50Ms
    }
}

private struct ScoringFixture: Sendable {
    let n: Int
    let dimension: Int
    let query: [Float]
    let vectors: [[Float]]
    let flattenedVectors: [Float]
    let recencyWeights: [Float]
    let chunks: [MemoryChunk]
}

func runScoringBenchmarks() async throws -> [ScoringCaseResult] {
    var results: [ScoringCaseResult] = []
    let sizes = benchmarkProfile == .quick ? [100, 500, 2_000] : [100, 500, 2_000, 10_000, 50_000]

    for (index, n) in sizes.enumerated() {
        let fixture = makeScoringFixture(n: n, dimension: 384)
        let settings = scoringSettings(for: n)
        let scoringEngine = try ContextCoreEngine.ScoringEngine()
        let prepared = try await scoringEngine.makePreparedScoringInputs(
            query: fixture.query,
            flattenedEmbeddings: fixture.flattenedVectors,
            count: fixture.n,
            dimension: fixture.dimension,
            recencyWeights: fixture.recencyWeights
        )

        let mathMeasurements = try await benchmarkPair(
            prefix: "scoreMath",
            size: n,
            preferGPUFirst: index.isMultiple(of: 2),
            warmup: settings.warmup,
            iterations: settings.iterations,
            workUnitsPerIteration: Double(n)
        ) {
            try await scoringEngine.scorePreparedEmbeddings(
                prepared,
                relevanceWeight: 0.7,
                recencyWeight: 0.3
            )
        } cpu: {
            CPUReference.relevanceScores(
                query: fixture.query,
                flattenedChunks: fixture.flattenedVectors,
                count: fixture.n,
                dimension: fixture.dimension,
                recencyWeights: fixture.recencyWeights,
                relevanceWeight: 0.7,
                recencyWeight: 0.3
            )
        }

        let pipelineMeasurements = try await benchmarkPair(
            prefix: "scorePipeline",
            size: n,
            preferGPUFirst: !index.isMultiple(of: 2),
            warmup: settings.warmup,
            iterations: settings.iterations,
            workUnitsPerIteration: Double(n)
        ) {
            try await scoringEngine.scoreChunks(
                query: fixture.query,
                chunks: fixture.chunks,
                recencyWeights: fixture.recencyWeights,
                relevanceWeight: 0.7,
                recencyWeight: 0.3
            )
        } cpu: {
            cpuScoreChunksEndToEnd(
                query: fixture.query,
                vectors: fixture.vectors,
                chunks: fixture.chunks,
                recencyWeights: fixture.recencyWeights,
                relevanceWeight: 0.7,
                recencyWeight: 0.3
            )
        }

        results.append(
            ScoringCaseResult(
                track: .mathOnly,
                n: n,
                gpu: mathMeasurements.gpu,
                cpu: mathMeasurements.cpu
            )
        )
        results.append(
            ScoringCaseResult(
                track: .endToEnd,
                n: n,
                gpu: pipelineMeasurements.gpu,
                cpu: pipelineMeasurements.cpu
            )
        )
    }

    return results
}

private func makeScoringFixture(n: Int, dimension: Int) -> ScoringFixture {
    let vectors = BenchmarkDataFactory.randomVectors(n: n, dim: dimension, seed: 42)
    let query = BenchmarkDataFactory.randomVector(dim: dimension, seed: 99)
    let recencyWeights = (0..<n).map { Float($0) / Float(max(1, n - 1)) }

    var flattenedVectors: [Float] = []
    flattenedVectors.reserveCapacity(n * dimension)
    for vector in vectors {
        flattenedVectors.append(contentsOf: vector)
    }

    let chunks: [MemoryChunk] = vectors.enumerated().map { index, vector in
        MemoryChunk(
            content: "chunk \(index)",
            embedding: vector,
            type: .episodic,
            createdAt: Date(timeIntervalSince1970: Double(index)),
            lastAccessedAt: Date(timeIntervalSince1970: Double(index)),
            accessCount: 1,
            retentionScore: 1.0,
            sourceSessionID: UUID(),
            metadata: [:]
        )
    }

    return ScoringFixture(
        n: n,
        dimension: dimension,
        query: query,
        vectors: vectors,
        flattenedVectors: flattenedVectors,
        recencyWeights: recencyWeights,
        chunks: chunks
    )
}

private func scoringSettings(for n: Int) -> (warmup: Int, iterations: Int) {
    if benchmarkProfile == .quick {
        switch n {
        case ..<1_000:
            return (warmup: 2, iterations: 10)
        default:
            return (warmup: 2, iterations: 5)
        }
    }

    switch n {
    case ..<1_000:
        return (warmup: 8, iterations: 40)
    case ..<5_000:
        return (warmup: 6, iterations: 20)
    case ..<20_000:
        return (warmup: 4, iterations: 10)
    default:
        return (warmup: 3, iterations: 5)
    }
}

private func benchmarkPair<GPUResult, CPUResult>(
    prefix: String,
    size: Int,
    preferGPUFirst: Bool,
    warmup: Int,
    iterations: Int,
    workUnitsPerIteration: Double,
    gpu: @escaping () async throws -> GPUResult,
    cpu: @escaping () throws -> CPUResult
) async throws -> (gpu: BenchmarkResult, cpu: BenchmarkResult) {
    if preferGPUFirst {
        let gpuResult = try await benchmark(
            name: "\(prefix)GPU(\(size))",
            warmup: warmup,
            iterations: iterations,
            workUnitsPerIteration: workUnitsPerIteration
        ) {
            _ = try await gpu()
        }
        let cpuResult = try await benchmark(
            name: "\(prefix)CPU(\(size))",
            warmup: warmup,
            iterations: iterations,
            workUnitsPerIteration: workUnitsPerIteration
        ) {
            _ = try cpu()
        }
        return (gpu: gpuResult, cpu: cpuResult)
    }

    let cpuResult = try await benchmark(
        name: "\(prefix)CPU(\(size))",
        warmup: warmup,
        iterations: iterations,
        workUnitsPerIteration: workUnitsPerIteration
    ) {
        _ = try cpu()
    }
    let gpuResult = try await benchmark(
        name: "\(prefix)GPU(\(size))",
        warmup: warmup,
        iterations: iterations,
        workUnitsPerIteration: workUnitsPerIteration
    ) {
        _ = try await gpu()
    }

    return (gpu: gpuResult, cpu: cpuResult)
}

private func cpuScoreChunksEndToEnd(
    query: [Float],
    vectors: [[Float]],
    chunks: [MemoryChunk],
    recencyWeights: [Float],
    relevanceWeight: Float,
    recencyWeight: Float
) -> [(chunk: MemoryChunk, score: Float)] {
    let scores = CPUReference.relevanceScores(
        query: query,
        chunks: vectors,
        recencyWeights: recencyWeights,
        relevanceWeight: relevanceWeight,
        recencyWeight: recencyWeight
    )

    return zip(chunks, scores)
        .map { (chunk: $0.0, score: $0.1) }
        .sorted { lhs, rhs in
            if lhs.score == rhs.score {
                return lhs.chunk.id.uuidString < rhs.chunk.id.uuidString
            }
            return lhs.score > rhs.score
        }
}
