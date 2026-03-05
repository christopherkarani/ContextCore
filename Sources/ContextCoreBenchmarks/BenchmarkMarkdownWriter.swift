import Foundation

struct BenchmarkEnvironment {
    let machine: String
    let osVersion: String
    let dateISO8601: String
}

func currentBenchmarkEnvironment() -> BenchmarkEnvironment {
    let process = ProcessInfo.processInfo
    let machine = process.hostName
    let osVersion = process.operatingSystemVersionString
    let date = ISO8601DateFormatter().string(from: Date())
    return BenchmarkEnvironment(machine: machine, osVersion: osVersion, dateISO8601: date)
}

func generateBenchmarksMarkdown(
    environment: BenchmarkEnvironment,
    buildWindow: [BuildWindowCaseResult],
    consolidation: [ConsolidationCaseResult],
    scoring: [ScoringCaseResult],
    recall: [RecallQualityPoint],
    outputURL: URL = URL(fileURLWithPath: "BENCHMARKS.md")
) throws {
    let buildByKey = Dictionary(uniqueKeysWithValues: buildWindow.map { ("\($0.turns)-\($0.budget)", $0.metrics) })
    let consolidationByKey = Dictionary(uniqueKeysWithValues: consolidation.map { ($0.chunks, $0.metrics) })
    let scoringByKey = Dictionary(uniqueKeysWithValues: scoring.map { ("\($0.track.rawValue)-\($0.n)", $0) })
    let sortedRecall = recall.sorted { $0.k < $1.k }
    let scoringSizes = scoring.map(\.n).sorted().reduce(into: [Int]()) { values, size in
        if values.last != size {
            values.append(size)
        }
    }

    var lines: [String] = []
    lines.append("# ContextCore Benchmarks")
    lines.append("")
    lines.append("> Profile: `\(benchmarkProfile == .quick ? "quick" : "full")`.")
    lines.append("")
    if benchmarkProfile == .quick {
        lines.append("> Quick profile runs a reduced matrix for local iteration speed.")
        lines.append("")
    }
    lines.append("Measured on \(environment.machine), \(environment.osVersion), \(environment.dateISO8601).")
    lines.append("")
    lines.append("## buildWindow Latency")
    lines.append("")
    lines.append("| Turns | Budget | p50 (ms) | p95 (ms) | p99 (ms) |")
    lines.append("|---|---|---|---|---|")
    for turns in [10, 50, 200, 500] {
        for budget in [2048, 4096, 8192] {
            if let metrics = buildByKey["\(turns)-\(budget)"] {
                lines.append("| \(turns) | \(budget) | \(formatMs(metrics.p50Ms)) | \(formatMs(metrics.p95Ms)) | \(formatMs(metrics.p99Ms)) |")
            } else {
                lines.append("| \(turns) | \(budget) | N/A | N/A | N/A |")
            }
        }
    }
    lines.append("")
    lines.append("**Target**: p99 < 20ms for 500 turns on M2.")
    lines.append("")
    lines.append("## Consolidation Latency")
    lines.append("")
    lines.append("| Chunks | p50 (ms) | p95 (ms) | p99 (ms) |")
    lines.append("|---|---|---|---|")
    for chunks in [100, 500, 2000] {
        if let metrics = consolidationByKey[chunks] {
            lines.append("| \(chunks) | \(formatMs(metrics.p50Ms)) | \(formatMs(metrics.p95Ms)) | \(formatMs(metrics.p99Ms)) |")
        } else {
            lines.append("| \(chunks) | N/A | N/A | N/A |")
        }
    }
    lines.append("")
    lines.append("**Target**: p99 < 500ms for 2000 chunks on M2.")
    lines.append("")
    lines.append("## Metal vs CPU Scoring")
    lines.append("")
    lines.append("Math-only isolates score computation over pre-flattened inputs. End-to-end includes public API validation, flattening, zip, and final sort.")
    lines.append("")
    lines.append("### Math-only")
    lines.append("")
    lines.append("| n | GPU p50 | GPU p99 | GPU throughput | CPU p50 | CPU p99 | CPU throughput | Speedup |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for n in scoringSizes {
        if let row = scoringByKey["\(ScoringBenchmarkTrack.mathOnly.rawValue)-\(n)"] {
            lines.append(
                "| \(n) | \(formatDuration(row.gpu.p50Ms)) | \(formatDuration(row.gpu.p99Ms)) | \(row.gpu.throughputP50.map(formatThroughput) ?? "N/A") | \(formatDuration(row.cpu.p50Ms)) | \(formatDuration(row.cpu.p99Ms)) | \(row.cpu.throughputP50.map(formatThroughput) ?? "N/A") | \(String(format: "%.2fx", row.speedup)) |"
            )
        } else {
            lines.append("| \(n) | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
        }
    }
    lines.append("")
    lines.append("### End-to-end")
    lines.append("")
    lines.append("| n | GPU p50 | GPU p99 | GPU throughput | CPU p50 | CPU p99 | CPU throughput | Speedup |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for n in scoringSizes {
        if let row = scoringByKey["\(ScoringBenchmarkTrack.endToEnd.rawValue)-\(n)"] {
            lines.append(
                "| \(n) | \(formatDuration(row.gpu.p50Ms)) | \(formatDuration(row.gpu.p99Ms)) | \(row.gpu.throughputP50.map(formatThroughput) ?? "N/A") | \(formatDuration(row.cpu.p50Ms)) | \(formatDuration(row.cpu.p99Ms)) | \(row.cpu.throughputP50.map(formatThroughput) ?? "N/A") | \(String(format: "%.2fx", row.speedup)) |"
            )
        } else {
            lines.append("| \(n) | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
        }
    }
    lines.append("")
    lines.append("## Recall Quality")
    lines.append("")
    lines.append("| k | Precision@k |")
    lines.append("|---|---|")
    for row in sortedRecall {
        lines.append("| \(row.k) | \(String(format: "%.3f", row.precision)) |")
    }
    lines.append("")
    lines.append("## Memory Footprint")
    lines.append("")
    lines.append("500-turn session, dim=384, degree=32:")
    lines.append("- Episodic index: ~0.9 MB")
    lines.append("- Semantic index: ~0.1 MB")
    lines.append("- Scoring buffers: reused shared Metal buffers sized to the largest active scoring workload")
    lines.append("- **Total GPU memory: dominated by active embedding and scoring workload size**")

    try lines.joined(separator: "\n").write(to: outputURL, atomically: true, encoding: .utf8)
}
