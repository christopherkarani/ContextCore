import Foundation

@main
struct BenchmarkRunner {
    static func main() async throws {
        let environment = currentBenchmarkEnvironment()

        print("ContextCore Benchmark Suite")
        print("==========================")
        print("Profile: \(benchmarkProfile == .quick ? "quick" : "full")")
        print("Device: \(environment.machine)")
        print("OS: \(environment.osVersion)")
        print("Date: \(environment.dateISO8601)\n")

        print("--- buildWindow Latency ---")
        let buildResults = try await runBuildWindowBenchmarks()
        for result in buildResults {
            print("  buildWindow(\(result.turns), \(result.budget)): \(result.metrics.formattedSummary)")
        }

        print("\n--- Consolidation Latency ---")
        let consolidationResults = try await runConsolidationBenchmarks()
        for result in consolidationResults {
            print("  consolidate(\(result.chunks)): \(result.metrics.formattedSummary)")
        }

        print("\n--- Metal vs CPU Scoring ---")
        let scoringResults = try await runScoringBenchmarks()
        for result in scoringResults.filter({ $0.track == .mathOnly }) {
            print(
                "  math n=\(result.n): gpu=\(formatDuration(result.gpu.p50Ms)) (\(result.gpu.throughputP50.map(formatThroughput) ?? "n/a")) cpu=\(formatDuration(result.cpu.p50Ms)) (\(result.cpu.throughputP50.map(formatThroughput) ?? "n/a")) speedup=\(String(format: "%.2fx", result.speedup))"
            )
        }
        for result in scoringResults.filter({ $0.track == .endToEnd }) {
            print(
                "  pipeline n=\(result.n): gpu=\(formatDuration(result.gpu.p50Ms)) (\(result.gpu.throughputP50.map(formatThroughput) ?? "n/a")) cpu=\(formatDuration(result.cpu.p50Ms)) (\(result.cpu.throughputP50.map(formatThroughput) ?? "n/a")) speedup=\(String(format: "%.2fx", result.speedup))"
            )
        }

        print("\n--- Recall Quality ---")
        let recallResults = try await runRecallQualityBenchmark()
        for result in recallResults {
            print(
                "  precision@\(result.k) = \(String(format: "%.3f", result.precision)) (\(result.truePositives)/\(result.k))"
            )
        }

        try generateBenchmarksMarkdown(
            environment: environment,
            buildWindow: buildResults,
            consolidation: consolidationResults,
            scoring: scoringResults,
            recall: recallResults
        )

        print("\nBENCHMARKS.md updated.")
    }
}
