import Foundation

public enum MemoryType: String, Codable, Sendable, Hashable {
    case episodic
    case semantic
    case procedural
}

public struct MemoryChunk: Identifiable, Codable, Sendable, Hashable {
    public let id: UUID
    public var content: String
    public var embedding: [Float]
    public let type: MemoryType
    public let createdAt: Date
    public var lastAccessedAt: Date
    public var accessCount: Int
    public var retentionScore: Float
    public let sourceSessionID: UUID
    public var metadata: [String: String]

    public init(
        id: UUID = UUID(),
        content: String,
        embedding: [Float],
        type: MemoryType,
        createdAt: Date = .now,
        lastAccessedAt: Date = .now,
        accessCount: Int = 1,
        retentionScore: Float,
        sourceSessionID: UUID,
        metadata: [String: String] = [:]
    ) {
        self.id = id
        self.content = content
        self.embedding = embedding
        self.type = type
        self.createdAt = createdAt
        self.lastAccessedAt = lastAccessedAt
        self.accessCount = accessCount
        self.retentionScore = retentionScore
        self.sourceSessionID = sourceSessionID
        self.metadata = metadata
    }
}
