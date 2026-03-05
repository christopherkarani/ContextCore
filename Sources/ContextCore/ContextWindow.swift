import Foundation

public enum CompressionLevel: Int, Codable, Sendable, Hashable, Comparable {
    case none = 0
    case light = 1
    case heavy = 2
    case dropped = 3

    public static func < (lhs: CompressionLevel, rhs: CompressionLevel) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

public enum FormatStyle: Sendable {
    case raw
    case chatML
    case alpaca
    case custom(template: String)
}

public struct ContextChunk: Identifiable, Codable, Sendable, Hashable {
    public let id: UUID
    public let content: String
    public let role: TurnRole
    public let tokenCount: Int
    public let score: Float
    public let source: MemoryType
    public var compressionLevel: CompressionLevel
    public let timestamp: Date
    public let isGuaranteedRecent: Bool
    public let isSystemPrompt: Bool

    public init(
        id: UUID = UUID(),
        content: String,
        role: TurnRole,
        tokenCount: Int,
        score: Float,
        source: MemoryType,
        compressionLevel: CompressionLevel = .none,
        timestamp: Date = .now,
        isGuaranteedRecent: Bool = false,
        isSystemPrompt: Bool = false
    ) {
        self.id = id
        self.content = content
        self.role = role
        self.tokenCount = tokenCount
        self.score = score
        self.source = source
        self.compressionLevel = compressionLevel
        self.timestamp = timestamp
        self.isGuaranteedRecent = isGuaranteedRecent
        self.isSystemPrompt = isSystemPrompt
    }
}

public struct ContextWindow: Codable, Sendable, Hashable {
    public let chunks: [ContextChunk]
    public let totalTokens: Int
    public let budgetUsed: Float
    public let budget: Int
    public let retrievedFromMemory: Int
    public let compressedChunks: Int

    public init(chunks: [ContextChunk], budget: Int) {
        self.chunks = chunks
        self.budget = budget
        self.totalTokens = chunks.reduce(into: 0) { partial, chunk in
            partial += chunk.tokenCount
        }

        if budget > 0 {
            let ratio = Float(totalTokens) / Float(budget)
            self.budgetUsed = min(max(ratio, 0), 1)
        } else {
            self.budgetUsed = totalTokens == 0 ? 0 : 1
        }

        self.retrievedFromMemory = chunks.reduce(into: 0) { partial, chunk in
            let isRetrievedMemory = (chunk.source == .episodic || chunk.source == .semantic)
                && !chunk.isGuaranteedRecent
                && !chunk.isSystemPrompt
            if isRetrievedMemory {
                partial += 1
            }
        }

        self.compressedChunks = chunks.reduce(into: 0) { partial, chunk in
            if chunk.compressionLevel > .none {
                partial += 1
            }
        }
    }

    public func formatted(style: FormatStyle) -> String {
        switch style {
        case .raw:
            return chunks.map(\ .content).joined(separator: "\n\n")
        case .chatML:
            return chunks
                .map { chunk in
                    "<|im_start|>\(chunk.role.rawValue)\n\(chunk.content)<|im_end|>"
                }
                .joined(separator: "\n")
        case .alpaca:
            return chunks
                .map { chunk in
                    switch chunk.role {
                    case .system:
                        return "### Instruction:\n\(chunk.content)\n\n"
                    case .user:
                        return "### Input:\n\(chunk.content)\n\n"
                    case .assistant:
                        return "### Response:\n\(chunk.content)\n\n"
                    case .tool:
                        return "### Tool Output:\n\(chunk.content)\n\n"
                    }
                }
                .joined()
        case .custom(let template):
            return chunks
                .map { chunk in
                    template
                        .replacingOccurrences(of: "{role}", with: chunk.role.rawValue)
                        .replacingOccurrences(of: "{content}", with: chunk.content)
                }
                .joined(separator: "\n")
        }
    }
}
