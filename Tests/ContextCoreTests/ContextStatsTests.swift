import Foundation
import Testing
@testable import ContextCore

@Suite("ContextStats Tests")
struct ContextStatsTests {
    @Test("Initial stats are zero")
    func initialStats() throws {
        let context = try AgentContext(configuration: makeAgentConfiguration())
        let stats = context.stats

        #expect(stats.episodicCount == 0)
        #expect(stats.semanticCount == 0)
        #expect(stats.proceduralCount == 0)
        #expect(stats.totalSessions == 0)
        #expect(stats.lastBuildWindowLatencyMs == 0)
        #expect(stats.lastConsolidationLatencyMs == 0)
        #expect(stats.averageRelevanceScore == 0)
        #expect(stats.compressionRatio == 0)
    }

    @Test("episodicCount updates after append")
    func episodicCountAfterAppend() async throws {
        let context = try AgentContext(configuration: makeAgentConfiguration())
        try await context.beginSession(id: UUID(), systemPrompt: nil)

        for index in 0..<5 {
            try await context.append(turn: Turn(role: .user, content: "turn-\(index)"))
        }

        #expect(context.stats.episodicCount == 5)
    }

    @Test("semanticCount updates after remember")
    func semanticCountAfterRemember() async throws {
        let context = try AgentContext(configuration: makeAgentConfiguration())
        try await context.beginSession(id: UUID(), systemPrompt: nil)

        try await context.remember("The preferred theme is dark")
        #expect(context.stats.semanticCount == 1)
    }

    @Test("totalSessions increments across session lifecycle")
    func totalSessions() async throws {
        let context = try AgentContext(configuration: makeAgentConfiguration())

        for _ in 0..<3 {
            try await context.beginSession(id: UUID(), systemPrompt: nil)
            try await context.endSession()
        }

        #expect(context.stats.totalSessions == 3)
    }

    @Test("buildWindow updates latency and relevance stats")
    func buildWindowStats() async throws {
        let context = try AgentContext(configuration: makeAgentConfiguration())
        try await context.beginSession(id: UUID(), systemPrompt: "Be direct")

        for turn in makeDiverseTurns() {
            try await context.append(turn: turn)
        }

        _ = try await context.buildWindow(currentTask: "Help with Swift errors")
        let stats = context.stats

        #expect(stats.lastBuildWindowLatencyMs > 0)
        #expect(stats.averageRelevanceScore >= 0)
        #expect(stats.averageRelevanceScore <= 1)
    }

    @Test("stats can be read nonisolated without await")
    func statsNonisolatedRead() async throws {
        let context = try AgentContext(configuration: makeAgentConfiguration())
        try await context.beginSession(id: UUID(), systemPrompt: nil)
        try await context.append(turn: Turn(role: .user, content: "Hello"))

        let snapshot: ContextStats = context.stats
        #expect(snapshot.episodicCount == 1)
    }

    @Test("stats remain valid under rapid updates")
    func statsRapidUpdates() async throws {
        let context = try AgentContext(configuration: makeAgentConfiguration())
        try await context.beginSession(id: UUID(), systemPrompt: nil)

        await withTaskGroup(of: Void.self) { group in
            for index in 0..<100 {
                group.addTask {
                    try? await context.append(turn: Turn(role: .user, content: "burst-\(index)"))
                    _ = context.stats
                }
            }
        }

        #expect(context.stats.episodicCount >= 1)
        #expect(context.stats.episodicCount <= 100)
    }
}
