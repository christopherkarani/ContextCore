import Foundation
import Testing
@testable import ContextCore

@Suite("Window Packer Tests")
struct WindowPackerTests {
    @Test("ContextWindow totalTokens sums chunk token counts")
    func contextWindowTotalTokens() {
        let chunks = [
            makeContextChunk(content: "A", tokenCount: 100),
            makeContextChunk(content: "B", tokenCount: 200),
            makeContextChunk(content: "C", tokenCount: 300),
        ]

        let window = ContextWindow(chunks: chunks, budget: 1_000)
        #expect(window.totalTokens == 600)
    }

    @Test("ContextWindow budgetUsed is total divided by budget")
    func contextWindowBudgetUsed() {
        let chunks = [
            makeContextChunk(content: "A", tokenCount: 100),
            makeContextChunk(content: "B", tokenCount: 200),
            makeContextChunk(content: "C", tokenCount: 300),
        ]

        let window = ContextWindow(chunks: chunks, budget: 1_000)
        #expect(abs(window.budgetUsed - 0.6) < 1e-6)
    }

    @Test("ContextWindow raw formatting joins chunk content")
    func contextWindowRawFormatting() {
        let chunks = [
            makeContextChunk(content: "Hello", role: .user),
            makeContextChunk(content: "World", role: .assistant),
        ]

        let window = ContextWindow(chunks: chunks, budget: 1_000)
        #expect(window.formatted(style: .raw) == "Hello\n\nWorld")
    }

    @Test("ContextWindow chatML formatting includes role wrappers")
    func contextWindowChatMLFormatting() {
        let chunks = [
            makeContextChunk(content: "You are helpful", role: .system, source: .semantic, isSystemPrompt: true),
            makeContextChunk(content: "Hi", role: .user),
        ]

        let window = ContextWindow(chunks: chunks, budget: 1_000)
        let formatted = window.formatted(style: .chatML)

        #expect(formatted.contains("<|im_start|>system\nYou are helpful<|im_end|>"))
        #expect(formatted.contains("<|im_start|>user\nHi<|im_end|>"))
    }

    @Test("ContextWindow alpaca formatting uses role sections")
    func contextWindowAlpacaFormatting() {
        let chunks = [
            makeContextChunk(content: "You are helpful", role: .system, source: .semantic, isSystemPrompt: true),
            makeContextChunk(content: "Question", role: .user),
            makeContextChunk(content: "Answer", role: .assistant),
        ]

        let window = ContextWindow(chunks: chunks, budget: 1_000)
        let formatted = window.formatted(style: .alpaca)

        let instructionRange = formatted.range(of: "### Instruction:")
        let inputRange = formatted.range(of: "### Input:")
        let responseRange = formatted.range(of: "### Response:")

        #expect(instructionRange != nil)
        #expect(inputRange != nil)
        #expect(responseRange != nil)

        if let instructionRange, let inputRange, let responseRange {
            #expect(instructionRange.lowerBound < inputRange.lowerBound)
            #expect(inputRange.lowerBound < responseRange.lowerBound)
        }
    }

    @Test("ContextWindow custom formatting replaces role and content placeholders")
    func contextWindowCustomFormatting() {
        let chunks = [
            makeContextChunk(content: "Hello", role: .user),
            makeContextChunk(content: "World", role: .assistant),
        ]

        let window = ContextWindow(chunks: chunks, budget: 1_000)
        let formatted = window.formatted(style: .custom(template: "[{role}] {content}"))

        #expect(formatted == "[user] Hello\n[assistant] World")
    }

    @Test("ContextWindow empty chunks have zero tokens and empty raw formatting")
    func contextWindowEmpty() {
        let window = ContextWindow(chunks: [], budget: 1_000)

        #expect(window.totalTokens == 0)
        #expect(window.formatted(style: .raw).isEmpty)
    }

    @Test("ContextChunk Codable roundtrip preserves all fields")
    func contextChunkCodableRoundtrip() throws {
        let timestamp = Date(timeIntervalSince1970: 1_700_111_222)
        let chunk = ContextChunk(
            id: UUID(uuidString: "A1A1A1A1-A1A1-A1A1-A1A1-A1A1A1A1A1A1")!,
            content: "Codable payload",
            role: .tool,
            tokenCount: 123,
            score: 0.88,
            source: .procedural,
            compressionLevel: .light,
            timestamp: timestamp,
            isGuaranteedRecent: true,
            isSystemPrompt: false
        )

        let encoded = try JSONEncoder().encode(chunk)
        let decoded = try JSONDecoder().decode(ContextChunk.self, from: encoded)

        #expect(decoded == chunk)
    }

    @Test("CompressionLevel ordering follows none < light < heavy < dropped")
    func compressionLevelOrdering() {
        #expect(CompressionLevel.none < CompressionLevel.light)
        #expect(CompressionLevel.light < CompressionLevel.heavy)
        #expect(CompressionLevel.heavy < CompressionLevel.dropped)
    }

    @Test("ContextWindow retrievedFromMemory counts episodic and semantic retrieval only")
    func contextWindowRetrievedFromMemoryCount() {
        let chunks = [
            makeContextChunk(content: "episodic-1", source: .episodic),
            makeContextChunk(content: "episodic-2", source: .episodic),
            makeContextChunk(content: "semantic-1", source: .semantic),
            makeContextChunk(content: "system", role: .system, source: .semantic, isSystemPrompt: true),
        ]

        let window = ContextWindow(chunks: chunks, budget: 1_000)
        #expect(window.retrievedFromMemory == 3)
    }

    private func makeContextChunk(
        content: String,
        role: TurnRole = .assistant,
        tokenCount: Int? = nil,
        score: Float = 0.5,
        source: MemoryType = .episodic,
        compressionLevel: CompressionLevel = .none,
        timestamp: Date = .now,
        isGuaranteedRecent: Bool = false,
        isSystemPrompt: Bool = false
    ) -> ContextChunk {
        ContextChunk(
            content: content,
            role: role,
            tokenCount: tokenCount ?? ApproximateTokenCounter().count(content),
            score: score,
            source: source,
            compressionLevel: compressionLevel,
            timestamp: timestamp,
            isGuaranteedRecent: isGuaranteedRecent,
            isSystemPrompt: isSystemPrompt
        )
    }
}
