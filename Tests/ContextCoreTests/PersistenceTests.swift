import Foundation
import Testing
@testable import ContextCore

@Suite("Persistence Tests")
struct PersistenceTests {
    @Test("checkpoint creates file and JSON is decodable")
    func checkpointCreatesFile() async throws {
        let context = try AgentContext(configuration: makeAgentConfiguration())
        try await context.beginSession(id: UUID(), systemPrompt: "Helpful")
        try await context.append(turn: Turn(role: .user, content: "Hello"))

        let tempDir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }
        let url = tempDir.appendingPathComponent("context.checkpoint")

        try await context.checkpoint(to: url)

        #expect(FileManager.default.fileExists(atPath: url.path))

        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        _ = try decoder.decode(ContextCheckpoint.self, from: data)
    }

    @Test("load restores checkpointed context")
    func loadRestoresContext() async throws {
        let context = try AgentContext(configuration: makeAgentConfiguration())
        try await context.beginSession(id: UUID(), systemPrompt: "Helpful")

        for turn in makeDiverseTurns() {
            try await context.append(turn: turn)
        }

        try await context.remember("Chris prefers concise answers")

        let tempDir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }
        let url = tempDir.appendingPathComponent("restore.checkpoint")

        let window1 = try await context.buildWindow(currentTask: "Help with Swift")
        try await context.checkpoint(to: url)

        let restored = try await AgentContext.load(from: url)
        let window2 = try await restored.buildWindow(currentTask: "Help with Swift")

        #expect(window1.chunks.count == window2.chunks.count)
        #expect(window2.totalTokens <= window2.budget)

        let firstIDs = Set(window1.chunks.map(\.id))
        let secondIDs = Set(window2.chunks.map(\.id))
        let overlap = firstIDs.intersection(secondIDs).count
        #expect(overlap >= Int(Double(firstIDs.count) * 0.6))

        let recall = try await restored.recall(query: "concise answers", k: 5)
        #expect(recall.contains(where: { $0.content.contains("concise") }))
    }

    @Test("roundtrip preserves configuration for budget calculation")
    func roundtripPreservesConfiguration() async throws {
        var config = makeAgentConfiguration(maxTokens: 1_000, tokenSafetyMargin: 0.10)
        config.recentTurnsGuaranteed = 2

        let context = try AgentContext(configuration: config)
        try await context.beginSession(id: UUID(), systemPrompt: "S")
        try await context.append(turn: Turn(role: .user, content: "A short prompt"))

        let tempDir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }
        let url = tempDir.appendingPathComponent("config.checkpoint")

        try await context.checkpoint(to: url)
        let restored = try await AgentContext.load(from: url)

        let window = try await restored.buildWindow(currentTask: "A short prompt")
        #expect(window.budget == 900)
    }

    @Test("load corrupt checkpoint throws checkpointCorrupt")
    func corruptCheckpoint() async throws {
        let tempDir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }
        let url = tempDir.appendingPathComponent("corrupt.checkpoint")

        try "not valid json".data(using: .utf8)?.write(to: url)

        do {
            _ = try await AgentContext.load(from: url)
            #expect(Bool(false))
        } catch let error as ContextCoreError {
            #expect(error == .checkpointCorrupt)
        } catch {
            #expect(Bool(false))
        }
    }

    @Test("load nonexistent file throws file error")
    func nonexistentCheckpoint() async throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("missing-\(UUID().uuidString).checkpoint")

        do {
            _ = try await AgentContext.load(from: url)
            #expect(Bool(false))
        } catch {
            #expect(Bool(true))
        }
    }

    @Test("checkpoint overwrite is valid")
    func checkpointOverwrite() async throws {
        let context = try AgentContext(configuration: makeAgentConfiguration())
        try await context.beginSession(id: UUID(), systemPrompt: nil)
        try await context.append(turn: Turn(role: .user, content: "first"))

        let tempDir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }
        let url = tempDir.appendingPathComponent("overwrite.checkpoint")

        try await context.checkpoint(to: url)
        try await context.append(turn: Turn(role: .assistant, content: "second"))
        try await context.checkpoint(to: url)

        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        _ = try decoder.decode(ContextCheckpoint.self, from: data)

        let restored = try await AgentContext.load(from: url)
        let recall = try await restored.recall(query: "second", k: 5)
        #expect(recall.contains(where: { $0.content.contains("second") }))
    }

    @Test("totalSessions is preserved across checkpoint")
    func totalSessionsPreserved() async throws {
        let context = try AgentContext(configuration: makeAgentConfiguration())

        for _ in 0..<3 {
            try await context.beginSession(id: UUID(), systemPrompt: nil)
            try await context.append(turn: Turn(role: .user, content: "session turn"))
            try await context.endSession()
        }

        let tempDir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }
        let url = tempDir.appendingPathComponent("sessions.checkpoint")

        try await context.checkpoint(to: url)
        let restored = try await AgentContext.load(from: url)

        #expect(restored.stats.totalSessions == 3)
    }

    private func makeTempDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("ContextCoreTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }
}
