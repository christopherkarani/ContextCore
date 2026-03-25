import ContextCore
import Foundation

struct BuildWindowCaseResult: Sendable {
    let turns: Int
    let budget: Int
    let metrics: BenchmarkResult
}

func runBuildWindowBenchmarks() async throws -> [BuildWindowCaseResult] {
    var results: [BuildWindowCaseResult] = []
    let turnCounts = benchmarkProfile == .quick ? [10, 50] : [10, 50, 200, 500]
    let budgets = benchmarkProfile == .quick ? [2048, 4096] : [2048, 4096, 8192]
    let warmup = benchmarkProfile == .quick ? 2 : 5
    let iterations = benchmarkProfile == .quick ? 10 : 50

    for turnCount in turnCounts {
        for budget in budgets {
            let config = BenchmarkDataFactory.makeConfiguration()
            let context = try await makePrimedContext(
                turnCount: turnCount,
                configuration: config,
                systemPrompt: "You are a helpful assistant."
            )

            let metrics = try await benchmark(
                name: "buildWindow(\(turnCount),\(budget))",
                warmup: warmup,
                iterations: iterations
            ) {
                _ = try await context.buildWindow(
                    currentTask: "Help the user debug a Swift concurrency issue and summarize next steps.",
                    maxTokens: budget
                )
            }

            results.append(
                BuildWindowCaseResult(
                    turns: turnCount,
                    budget: budget,
                    metrics: metrics
                )
            )
        }
    }

    return results
}

private func makePrimedContext(
    turnCount: Int,
    configuration: ContextConfiguration,
    systemPrompt: String
) async throws -> AgentContext {
    let context = try AgentContext(configuration: configuration)
    try await context.beginSession(systemPrompt: systemPrompt)
    for index in 0..<turnCount {
        let role = index.isMultiple(of: 2) ? TurnRole.user : TurnRole.assistant
        try await context.append(
            turn: Turn(
                role: role,
                content: BenchmarkDataFactory.realisticTurnContent(index: index),
                timestamp: Date(timeIntervalSince1970: Double(index))
            )
        )
    }
    return context
}
