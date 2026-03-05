import Foundation

public enum TurnRole: String, Codable, Sendable, Hashable {
    case user
    case assistant
    case tool
    case system
}

public struct Turn: Identifiable, Codable, Sendable, Hashable {
    public let id: UUID
    public let role: TurnRole
    public let content: String
    public let timestamp: Date
    public var tokenCount: Int
    public var embedding: [Float]?
    public var metadata: [String: String]

    public init(
        id: UUID = UUID(),
        role: TurnRole,
        content: String,
        timestamp: Date = .now,
        tokenCount: Int = 0,
        embedding: [Float]? = nil,
        metadata: [String: String] = [:]
    ) {
        self.id = id
        self.role = role
        self.content = content
        self.timestamp = timestamp
        self.tokenCount = tokenCount
        self.embedding = embedding
        self.metadata = metadata
    }

    public static func == (lhs: Turn, rhs: Turn) -> Bool {
        lhs.id == rhs.id
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
}

public struct ToolCall: Codable, Sendable, Hashable {
    public let name: String
    public let input: String
    public let output: String
    public let durationMs: Double

    public init(name: String, input: String, output: String, durationMs: Double) {
        self.name = name
        self.input = input
        self.output = output
        self.durationMs = durationMs
    }
}
