import ContextCore
import Foundation

struct ScoringCaseResult: Sendable {
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

func runScoringBenchmarks() async throws -> [ScoringCaseResult] {
    var results: [ScoringCaseResult] = []
    let sizes = benchmarkProfile == .quick ? [100, 500] : [100, 500, 2000]
    let warmup = benchmarkProfile == .quick ? 3 : 10
    let iterations = benchmarkProfile == .quick ? 25 : 100

    for n in sizes {
        let vectors = BenchmarkDataFactory.randomVectors(n: n, dim: 384, seed: 42)
        let query = BenchmarkDataFactory.randomVector(dim: 384, seed: 99)
        let recencyWeights = (0..<n).map { Float($0) / Float(max(1, n - 1)) }

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

        let scoringEngine = try ScoringEngine()

        let gpu = try await benchmark(
            name: "gpuScore(\(n))",
            warmup: warmup,
            iterations: iterations
        ) {
            _ = try await scoringEngine.scoreChunks(
                query: query,
                chunks: chunks,
                recencyWeights: recencyWeights,
                relevanceWeight: 0.7,
                recencyWeight: 0.3
            )
        }

        let cpu = try await benchmark(
            name: "cpuScore(\(n))",
            warmup: warmup,
            iterations: iterations
        ) {
            _ = CPUReference.relevanceScores(
                query: query,
                chunks: vectors,
                recencyWeights: recencyWeights,
                relevanceWeight: 0.7,
                recencyWeight: 0.3
            )
        }

        results.append(
            ScoringCaseResult(
                n: n,
                gpu: gpu,
                cpu: cpu
            )
        )
    }

    return results
}
