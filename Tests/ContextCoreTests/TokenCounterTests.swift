import Foundation
import Testing
@testable import ContextCore

@Suite("Token Counter Tests")
struct TokenCounterTests {
    @Test("Hello world token estimate within 20%")
    func helloWorldEstimate() {
        let counter = ApproximateTokenCounter()
        let count = counter.count("Hello, world!")
        #expect((3...5).contains(count))
    }

    @Test("Quick brown fox token estimate within 20%")
    func quickBrownFoxEstimate() {
        let counter = ApproximateTokenCounter()
        let count = counter.count("The quick brown fox jumps over the lazy dog.")
        #expect((8...12).contains(count))
    }

    @Test("Swift language sentence estimate within 20%")
    func swiftSentenceEstimate() {
        let counter = ApproximateTokenCounter()
        let count = counter.count("Swift is a powerful and intuitive programming language.")
        #expect((7...11).contains(count))
    }

    @Test("GPU context sentence estimate within 20%")
    func gpuContextEstimate() {
        let counter = ApproximateTokenCounter()
        let count = counter.count("GPU-accelerated context management for on-device AI agents")
        #expect((8...12).contains(count))
    }

    @Test("Empty string returns zero")
    func emptyStringReturnsZero() {
        let counter = ApproximateTokenCounter()
        #expect(counter.count("") == 0)
    }

    @Test("Default maxTokens is 4096")
    func defaultMaxTokens() {
        #expect(ContextConfiguration.default.maxTokens == 4096)
    }

    @Test("Default tokenBudgetSafetyMargin is 0.10")
    func defaultSafetyMargin() {
        #expect(ContextConfiguration.default.tokenBudgetSafetyMargin == 0.10)
    }

    @Test("Default relevanceWeight is 0.7")
    func defaultRelevanceWeight() {
        #expect(ContextConfiguration.default.relevanceWeight == 0.7)
    }

    @Test("Default consolidationThreshold is 200")
    func defaultConsolidationThreshold() {
        #expect(ContextConfiguration.default.consolidationThreshold == 200)
    }

    @Test("ContextConfiguration can cross isolation boundaries")
    func contextConfigurationIsSendable() async {
        let configuration = ContextConfiguration.default
        let movedValue = await Task.detached {
            configuration.maxTokens
        }.value

        #expect(movedValue == 4096)
    }
}
