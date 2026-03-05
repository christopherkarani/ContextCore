import Foundation

public actor ProceduralStore {
    private let capacity: Int
    private var entries: [String: [ToolCall]] = [:]
    private var lastAccessedAt: [String: Date] = [:]

    public init(capacity: Int = 1_000) {
        self.capacity = capacity
    }

    public var count: Int {
        entries.count
    }

    public func record(taskType: String, tools: [ToolCall]) async {
        let now = Date()
        let isNewKey = entries[taskType] == nil

        if isNewKey && entries.count >= capacity {
            evictLeastRecentlyAccessed()
        }

        entries[taskType] = tools
        lastAccessedAt[taskType] = now
    }

    public func retrieve(taskType: String) async -> [ToolCall] {
        let matchingKeys = entries.keys.filter { $0.hasPrefix(taskType) }.sorted()
        guard !matchingKeys.isEmpty else {
            return []
        }

        let now = Date()
        for key in matchingKeys {
            lastAccessedAt[key] = now
        }

        return matchingKeys.flatMap { entries[$0] ?? [] }
    }

    private func evictLeastRecentlyAccessed() {
        guard let oldestKey = lastAccessedAt.min(by: { $0.value < $1.value })?.key else {
            return
        }
        entries.removeValue(forKey: oldestKey)
        lastAccessedAt.removeValue(forKey: oldestKey)
    }
}
