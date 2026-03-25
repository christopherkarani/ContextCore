import ContextCore
import Foundation

struct RecallQualityPoint: Sendable {
    let k: Int
    let precision: Double
    let truePositives: Int
}

func runRecallQualityBenchmark() async throws -> [RecallQualityPoint] {
    let config = BenchmarkDataFactory.makeConfiguration()
    let context = try AgentContext(configuration: config)
    try await context.beginSession(systemPrompt: nil)

    let relevantIndices: Set<Int> = [3, 7, 12, 18, 22, 31, 45]
    var turnIDs: [Int: UUID] = [:]

    for index in 0..<50 {
        let content: String
        if relevantIndices.contains(index) {
            content = BenchmarkDataFactory.relevantConcurrencyContent(index: index)
        } else {
            content = BenchmarkDataFactory.irrelevantContent(index: index)
        }

        let turn = Turn(role: .user, content: content)
        turnIDs[index] = turn.id
        try await context.append(turn: turn)
    }

    let window = try await context.buildWindow(
        currentTask: "Explain Swift concurrency patterns and actor isolation",
        maxTokens: 4096
    )

    let relevantIDs = Set(relevantIndices.compactMap { turnIDs[$0] })
    let retrieved = window.chunks
        .filter { !$0.isGuaranteedRecent && !$0.isSystemPrompt }
        .sorted { lhs, rhs in
            if lhs.score == rhs.score {
                return lhs.timestamp > rhs.timestamp
            }
            return lhs.score > rhs.score
        }

    var points: [RecallQualityPoint] = []
    for k in [3, 5, 8] {
        let ids = Set(retrieved.prefix(k).map(\.id))
        let truePositives = ids.intersection(relevantIDs).count
        let precision = Double(truePositives) / Double(k)
        points.append(
            RecallQualityPoint(
                k: k,
                precision: precision,
                truePositives: truePositives
            )
        )
    }

    return points
}
