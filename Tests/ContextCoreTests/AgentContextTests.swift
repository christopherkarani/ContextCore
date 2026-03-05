import Foundation
import Testing
@testable import ContextCore

@Suite("AgentContext Integration Tests")
struct AgentContextTests {
    @Test("Init succeeds and stats start at zero")
    func initSucceeds() throws {
        let context = try AgentContext(configuration: makeAgentConfiguration())
        let stats = context.stats

        #expect(stats.episodicCount == 0)
        #expect(stats.semanticCount == 0)
        #expect(stats.totalSessions == 0)
    }

    @Test("Session lifecycle increments totalSessions")
    func sessionLifecycle() async throws {
        let context = try AgentContext(configuration: makeAgentConfiguration())
        try await context.beginSession(id: UUID(), systemPrompt: "You are helpful")
        try await context.endSession()

        #expect(context.stats.totalSessions == 1)
    }

    @Test("Append tracks episodic memory count")
    func appendTracksCount() async throws {
        let context = try AgentContext(configuration: makeAgentConfiguration())
        try await context.beginSession(id: UUID(), systemPrompt: nil)

        for turn in makeDiverseTurns().prefix(5) {
            try await context.append(turn: turn)
        }

        #expect(context.stats.episodicCount == 5)
    }

    @Test("Append without session throws sessionNotStarted")
    func appendWithoutSession() async throws {
        let context = try AgentContext(configuration: makeAgentConfiguration())
        await expectContextError(.sessionNotStarted) {
            try await context.append(turn: Turn(role: .user, content: "hello"))
        }
    }

    @Test("buildWindow respects effective budget and includes guaranteed items")
    func buildWindowBudgetAndGuaranteed() async throws {
        var config = makeAgentConfiguration(maxTokens: 2_048, tokenSafetyMargin: 0.10)
        config.recentTurnsGuaranteed = 3

        let context = try AgentContext(configuration: config)
        try await context.beginSession(id: UUID(), systemPrompt: "Be concise and accurate.")

        for turn in makeDiverseTurns() {
            try await context.append(turn: turn)
        }

        let window = try await context.buildWindow(currentTask: "Debug Swift concurrency issues")
        #expect(window.budget == 1_843)
        #expect(window.totalTokens <= window.budget)

        let guaranteedRecent = window.chunks.filter(\.isGuaranteedRecent)
        #expect(guaranteedRecent.count == 3)
        #expect(window.chunks.contains(where: { $0.isSystemPrompt }))
    }

    @Test("buildWindow without session throws sessionNotStarted")
    func buildWindowWithoutSession() async throws {
        let context = try AgentContext(configuration: makeAgentConfiguration())
        await expectContextError(.sessionNotStarted) {
            _ = try await context.buildWindow(currentTask: "task")
        }
    }

    @Test("buildWindow can retrieve semantic memory across sessions")
    func buildWindowUsesSemanticAcrossSessions() async throws {
        let context = try AgentContext(configuration: makeAgentConfiguration())

        try await context.beginSession(id: UUID(), systemPrompt: nil)
        for _ in 0..<12 {
            try await context.append(turn: Turn(role: .assistant, content: "The user's name is Chris."))
        }
        try await context.endSession()

        try await context.beginSession(id: UUID(), systemPrompt: nil)
        try await context.append(turn: Turn(role: .user, content: "Remind me what name I shared"))

        let window = try await context.buildWindow(currentTask: "Recall the user's name")
        let semanticRetrieved = window.chunks.contains {
            $0.source == .semantic && !$0.isSystemPrompt
        }
        #expect(semanticRetrieved)
    }

    @Test("remember inserts semantic fact and recall finds it")
    func rememberAndRecall() async throws {
        let context = try AgentContext(configuration: makeAgentConfiguration())
        try await context.beginSession(id: UUID(), systemPrompt: nil)

        try await context.remember("User prefers dark mode")

        let results = try await context.recall(query: "dark mode", k: 5)
        #expect(context.stats.semanticCount >= 1)
        #expect(results.contains(where: { $0.content.contains("dark mode") }))
    }

    @Test("forget soft-deletes chunk from recall results")
    func forgetSoftDelete() async throws {
        let context = try AgentContext(configuration: makeAgentConfiguration())
        try await context.beginSession(id: UUID(), systemPrompt: nil)

        let turn = Turn(role: .user, content: "Remember this exact sentence for recall testing")
        try await context.append(turn: turn)

        var before = try await context.recall(query: "exact sentence recall", k: 5)
        #expect(before.contains(where: { $0.id == turn.id }))

        try await context.forget(id: turn.id)

        before = try await context.recall(query: "exact sentence recall", k: 5)
        #expect(!before.contains(where: { $0.id == turn.id }))
    }

    @Test("manual consolidate updates counts and latency")
    func manualConsolidate() async throws {
        let context = try AgentContext(configuration: makeAgentConfiguration())
        try await context.beginSession(id: UUID(), systemPrompt: nil)

        for _ in 0..<20 {
            try await context.append(turn: Turn(role: .assistant, content: "Repeated fact for consolidation"))
            try await context.append(turn: Turn(role: .assistant, content: "Repeated fact for consolidation with extra details"))
        }

        try await context.consolidate()

        #expect(context.stats.lastConsolidationLatencyMs >= 0)
        #expect(context.stats.semanticCount > 0)
    }

    @Test("auto-consolidation can run in background after insertion threshold")
    func autoConsolidation() async throws {
        var config = makeAgentConfiguration(consolidationThreshold: 500)
        config.similarityMergeThreshold = 0.92

        let context = try AgentContext(configuration: config)
        try await context.beginSession(id: UUID(), systemPrompt: nil)

        for _ in 0..<55 {
            try await context.append(turn: Turn(role: .assistant, content: "Auto consolidation repeated fact"))
            try await context.append(turn: Turn(role: .assistant, content: "Auto consolidation repeated fact with context"))
        }

        let becameSemantic = try await waitForSemanticRetrieval(context: context, timeoutMs: 4_000)
        #expect(becameSemantic)
    }

    @Test("embedding failures are wrapped as ContextCoreError.embeddingFailed")
    func embeddingFailure() async throws {
        let context = try AgentContext(configuration: makeAgentConfiguration(provider: FailingEmbeddingProvider()))
        try await context.beginSession(id: UUID(), systemPrompt: nil)

        do {
            try await context.append(turn: Turn(role: .user, content: "hello"))
            #expect(Bool(false))
        } catch let error as ContextCoreError {
            switch error {
            case .embeddingFailed:
                #expect(Bool(true))
            default:
                #expect(Bool(false))
            }
        } catch {
            #expect(Bool(false))
        }
    }

    @Test("tokenBudgetTooSmall when guaranteed context exceeds effective budget")
    func tokenBudgetTooSmall() async throws {
        var config = makeAgentConfiguration(maxTokens: 10, tokenSafetyMargin: 0)
        config.recentTurnsGuaranteed = 1

        let context = try AgentContext(configuration: config)
        try await context.beginSession(id: UUID(), systemPrompt: String(repeating: "verylong ", count: 80))

        await expectContextError(.tokenBudgetTooSmall) {
            _ = try await context.buildWindow(currentTask: "task", maxTokens: 10)
        }
    }

    @Test("dimension mismatch is thrown for inconsistent embeddings")
    func dimensionMismatch() async throws {
        let context = try AgentContext(configuration: makeAgentConfiguration())
        try await context.beginSession(id: UUID(), systemPrompt: nil)

        try await context.append(turn: Turn(role: .user, content: "normal turn"))

        let wrong = Turn(
            role: .assistant,
            content: "bad embedding",
            tokenCount: 3,
            embedding: [Float](repeating: 0.1, count: 128)
        )

        do {
            try await context.append(turn: wrong)
            #expect(Bool(false))
        } catch let error as ContextCoreError {
            switch error {
            case .dimensionMismatch(let expected, let got):
                #expect(expected == 384)
                #expect(got == 128)
            default:
                #expect(Bool(false))
            }
        } catch {
            #expect(Bool(false))
        }
    }

    @Test("multiple sequential errors do not poison context")
    func multipleErrorsThenRecovery() async throws {
        let context = try AgentContext(configuration: makeAgentConfiguration())

        await expectContextError(.sessionNotStarted) {
            try await context.append(turn: Turn(role: .user, content: "a"))
        }
        await expectContextError(.sessionNotStarted) {
            _ = try await context.buildWindow(currentTask: "b")
        }
        await expectContextError(.sessionNotStarted) {
            try await context.consolidate()
        }

        try await context.beginSession(id: UUID(), systemPrompt: nil)
        try await context.append(turn: Turn(role: .user, content: "recovered"))
        let window = try await context.buildWindow(currentTask: "recovered")
        #expect(window.chunks.isEmpty == false)
    }

    private func expectContextError(
        _ expected: ContextCoreError,
        operation: () async throws -> Void
    ) async {
        do {
            try await operation()
            #expect(Bool(false))
        } catch let error as ContextCoreError {
            #expect(error == expected)
        } catch {
            #expect(Bool(false))
        }
    }

    private func waitForSemanticRetrieval(
        context: AgentContext,
        timeoutMs: Int
    ) async throws -> Bool {
        let deadline = Date().addingTimeInterval(Double(timeoutMs) / 1000)

        while Date() < deadline {
            let window = try await context.buildWindow(currentTask: "Auto consolidation repeated fact")
            let hasSemantic = window.chunks.contains {
                $0.source == .semantic && !$0.isSystemPrompt
            }
            if hasSemantic {
                return true
            }
            try await Task.sleep(for: .milliseconds(50))
        }

        return false
    }
}
