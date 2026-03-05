import Foundation
import Testing
@testable import ContextCore

@Suite("Turn Model Tests")
struct TurnTests {
    @Test("Turn JSON roundtrip preserves all fields")
    func turnJSONRoundtrip() throws {
        let id = UUID(uuidString: "11111111-1111-1111-1111-111111111111")!
        let timestamp = Date(timeIntervalSince1970: 1_700_000_000)
        let turn = Turn(
            id: id,
            role: .assistant,
            content: "Hello from ContextCore",
            timestamp: timestamp,
            tokenCount: 42,
            embedding: [0.1, 0.2, 0.3],
            metadata: ["k": "v"]
        )

        let encoded = try JSONEncoder().encode(turn)
        let decoded = try JSONDecoder().decode(Turn.self, from: encoded)

        #expect(decoded.id == turn.id)
        #expect(decoded.role == turn.role)
        #expect(decoded.content == turn.content)
        #expect(decoded.timestamp == turn.timestamp)
        #expect(decoded.tokenCount == turn.tokenCount)
        #expect(decoded.embedding == turn.embedding)
        #expect(decoded.metadata == turn.metadata)
    }

    @Test("ToolCall JSON roundtrip preserves all fields")
    func toolCallJSONRoundtrip() throws {
        let toolCall = ToolCall(
            name: "search",
            input: "swift actors",
            output: "results",
            durationMs: 12.5
        )

        let encoded = try JSONEncoder().encode(toolCall)
        let decoded = try JSONDecoder().decode(ToolCall.self, from: encoded)

        #expect(decoded == toolCall)
    }

    @Test("ToolCall serialized in Turn metadata roundtrips")
    func toolCallInMetadataRoundtrip() throws {
        let toolCall = ToolCall(
            name: "open_url",
            input: "https://example.com",
            output: "200 OK",
            durationMs: 8.25
        )

        let toolCallData = try JSONEncoder().encode(toolCall)
        let toolCallJSON = String(decoding: toolCallData, as: UTF8.self)
        let turn = Turn(role: .tool, content: "Tool output", metadata: ["toolCall": toolCallJSON])

        let turnData = try JSONEncoder().encode(turn)
        let decodedTurn = try JSONDecoder().decode(Turn.self, from: turnData)
        let decodedToolCallData = Data(decodedTurn.metadata["toolCall"]!.utf8)
        let decodedToolCall = try JSONDecoder().decode(ToolCall.self, from: decodedToolCallData)

        #expect(decodedToolCall == toolCall)
    }

    @Test("Turns with different UUIDs are not equal")
    func turnsWithDifferentUUIDsNotEqual() {
        let turn1 = Turn(id: UUID(), role: .user, content: "a")
        let turn2 = Turn(id: UUID(), role: .user, content: "a")

        #expect(turn1 != turn2)
    }

    @Test("Turns with same UUID are equal")
    func turnsWithSameUUIDAreEqual() {
        let id = UUID(uuidString: "22222222-2222-2222-2222-222222222222")!
        let turn1 = Turn(id: id, role: .user, content: "first")
        let turn2 = Turn(id: id, role: .assistant, content: "second")

        #expect(turn1 == turn2)
    }
}
