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
    let workUnitsPerIteration: Double?
    let minMs: Double
    let p50Ms: Double
    let p95Ms: Double
    let p99Ms: Double
    let maxMs: Double

    var formattedSummary: String {
        var summary = "p50=\(formatDuration(p50Ms)) p95=\(formatDuration(p95Ms)) p99=\(formatDuration(p99Ms))"
        if let throughputP50 {
            summary += " throughput=\(formatThroughput(throughputP50))"
        }
        return summary
    }

    var throughputP50: Double? {
        throughput(for: p50Ms)
    }

    var throughputP95: Double? {
        throughput(for: p95Ms)
    }

    var throughputP99: Double? {
        throughput(for: p99Ms)
    }

    private func throughput(for latencyMs: Double) -> Double? {
        guard let workUnitsPerIteration, latencyMs > 0 else {
            return nil
        }
        return workUnitsPerIteration / (latencyMs / 1_000.0)
    }
}

func benchmark(
    name: String,
    warmup: Int = 5,
    iterations: Int = 50,
    workUnitsPerIteration: Double? = nil,
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
        workUnitsPerIteration: workUnitsPerIteration,
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

func formatDuration(_ milliseconds: Double) -> String {
    if milliseconds < 1.0 {
        return String(format: "%.2fus", milliseconds * 1_000.0)
    }

    return String(format: "%.2fms", milliseconds)
}

func formatThroughput(_ unitsPerSecond: Double) -> String {
    if unitsPerSecond >= 1_000_000 {
        return String(format: "%.2fM chunks/s", unitsPerSecond / 1_000_000.0)
    }
    if unitsPerSecond >= 1_000 {
        return String(format: "%.2fK chunks/s", unitsPerSecond / 1_000.0)
    }

    return String(format: "%.2f chunks/s", unitsPerSecond)
}
