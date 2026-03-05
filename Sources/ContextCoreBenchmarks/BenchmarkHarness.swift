import Foundation

enum BenchmarkProfile {
    case full
    case quick
}

let benchmarkProfile: BenchmarkProfile = {
    let raw = ProcessInfo.processInfo.environment["CONTEXTCORE_BENCH_PROFILE"]?.lowercased()
    return raw == "quick" ? .quick : .full
}()

struct BenchmarkResult: Sendable {
    let name: String
    let warmup: Int
    let iterations: Int
    let minMs: Double
    let p50Ms: Double
    let p95Ms: Double
    let p99Ms: Double
    let maxMs: Double

    var formattedSummary: String {
        "p50=\(formatMs(p50Ms))ms p95=\(formatMs(p95Ms))ms p99=\(formatMs(p99Ms))ms"
    }
}

func benchmark(
    name: String,
    warmup: Int = 5,
    iterations: Int = 50,
    block: () async throws -> Void
) async throws -> BenchmarkResult {
    precondition(iterations > 0, "iterations must be positive")
    precondition(warmup >= 0, "warmup cannot be negative")

    for _ in 0..<warmup {
        try await block()
    }

    var durationsMs: [Double] = []
    durationsMs.reserveCapacity(iterations)
    let clock = ContinuousClock()

    for _ in 0..<iterations {
        let elapsed = try await clock.measure {
            try await block()
        }

        let components = elapsed.components
        let milliseconds = (Double(components.seconds) * 1_000.0)
            + (Double(components.attoseconds) / 1_000_000_000_000_000.0)
        durationsMs.append(milliseconds)
    }

    durationsMs.sort()

    return BenchmarkResult(
        name: name,
        warmup: warmup,
        iterations: iterations,
        minMs: durationsMs.first ?? 0,
        p50Ms: percentile(0.50, from: durationsMs),
        p95Ms: percentile(0.95, from: durationsMs),
        p99Ms: percentile(0.99, from: durationsMs),
        maxMs: durationsMs.last ?? 0
    )
}

private func percentile(_ p: Double, from sortedValues: [Double]) -> Double {
    guard !sortedValues.isEmpty else {
        return 0
    }

    let clamped = min(max(p, 0), 1)
    let lastIndex = sortedValues.count - 1
    let index = min(lastIndex, Int((Double(lastIndex) * clamped).rounded(.toNearestOrEven)))
    return sortedValues[index]
}

func formatMs(_ value: Double) -> String {
    String(format: "%.2f", value)
}
