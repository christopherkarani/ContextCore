import Foundation
import Testing
@testable import ContextCore

@Suite("Recency Tests")
struct RecencyTests {
    @Test("Current timestamp weight is 1.0")
    func currentTimestamp() async throws {
        let current = Date(timeIntervalSince1970: 1_700_000_000)

        let engine = try ScoringEngine()
        let weights = try await engine.computeRecencyWeights(
            timestamps: [current],
            halfLife: 7 * 86_400,
            currentTime: current
        )

        #expect(weights.count == 1)
        #expect(abs(weights[0] - 1.0) < 1e-6)
    }

    @Test("One half-life ago is ~0.5")
    func oneHalfLife() async throws {
        let halfLife = 7.0 * 86_400
        let current = Date(timeIntervalSince1970: 1_700_000_000)
        let timestamp = current.addingTimeInterval(-halfLife)

        let engine = try ScoringEngine()
        let weights = try await engine.computeRecencyWeights(
            timestamps: [timestamp],
            halfLife: halfLife,
            currentTime: current
        )

        #expect(abs(weights[0] - 0.5) < 1e-3)
    }

    @Test("Two half-lives ago is ~0.25")
    func twoHalfLives() async throws {
        let halfLife = 7.0 * 86_400
        let current = Date(timeIntervalSince1970: 1_700_000_000)
        let timestamp = current.addingTimeInterval(-(halfLife * 2))

        let engine = try ScoringEngine()
        let weights = try await engine.computeRecencyWeights(
            timestamps: [timestamp],
            halfLife: halfLife,
            currentTime: current
        )

        #expect(abs(weights[0] - 0.25) < 1e-3)
    }

    @Test("Ten half-lives ago is below 0.01")
    func tenHalfLives() async throws {
        let halfLife = 7.0 * 86_400
        let current = Date(timeIntervalSince1970: 1_700_000_000)
        let timestamp = current.addingTimeInterval(-(halfLife * 10))

        let engine = try ScoringEngine()
        let weights = try await engine.computeRecencyWeights(
            timestamps: [timestamp],
            halfLife: halfLife,
            currentTime: current
        )

        #expect(weights[0] < 0.01)
    }

    @Test("Episodic half-life decays faster than semantic")
    func episodicVsSemantic() async throws {
        let current = Date(timeIntervalSince1970: 1_700_000_000)
        let timestamp = current.addingTimeInterval(-(30.0 * 86_400))

        let engine = try ScoringEngine()
        let episodic = try await engine.computeRecencyWeights(
            timestamps: [timestamp],
            halfLife: 7 * 86_400,
            currentTime: current
        )
        let semantic = try await engine.computeRecencyWeights(
            timestamps: [timestamp],
            halfLife: 90 * 86_400,
            currentTime: current
        )

        #expect(episodic[0] < semantic[0])
    }

    @Test("Monotonicity across ordered timestamps")
    func monotonicity() async throws {
        let current = Date(timeIntervalSince1970: 1_700_000_000)
        let timestamps = (0..<100).map { offset in
            current.addingTimeInterval(-Double(offset) * 3_600)
        }

        let engine = try ScoringEngine()
        let weights = try await engine.computeRecencyWeights(
            timestamps: timestamps,
            halfLife: 7 * 86_400,
            currentTime: current
        )

        for index in 0..<(weights.count - 1) {
            #expect(weights[index] > weights[index + 1])
        }
    }

    @Test("Weights remain in [0, 1]")
    func rangeCheck() async throws {
        let current = Date(timeIntervalSince1970: 1_700_000_000)
        let yearSeconds = 365.0 * 86_400
        var rng = TestHelpers.SeededGenerator(seed: 5001)

        let timestamps = (0..<1_000).map { _ -> Date in
            let ageFraction = Double(rng.next() & 0xFFFF) / Double(0xFFFF)
            let age = ageFraction * yearSeconds
            return current.addingTimeInterval(-age)
        }

        let engine = try ScoringEngine()
        let weights = try await engine.computeRecencyWeights(
            timestamps: timestamps,
            halfLife: 7 * 86_400,
            currentTime: current
        )

        #expect(weights.count == 1_000)
        #expect(weights.allSatisfy { $0 >= 0 && $0 <= 1 })
    }

#if !targetEnvironment(simulator)
    @Test("GPU vs CPU recency parity")
    func gpuVsCpuParity() async throws {
        let current = Date(timeIntervalSince1970: 1_700_000_000)
        let halfLife = 7.0 * 86_400
        var rng = TestHelpers.SeededGenerator(seed: 5002)

        let timestamps = (0..<1_000).map { _ -> Date in
            let ageFraction = Double(rng.next() & 0xFFFF) / Double(0xFFFF)
            let age = ageFraction * 365.0 * 86_400
            return current.addingTimeInterval(-age)
        }

        let cpu = CPUReference.recencyWeights(
            timestamps: timestamps,
            currentTime: current,
            halfLife: halfLife
        )

        let engine = try ScoringEngine()
        let gpu = try await engine.computeRecencyWeights(
            timestamps: timestamps,
            halfLife: halfLife,
            currentTime: current
        )

        #expect(gpu.count == cpu.count)
        let maxError = TestHelpers.maxAbsError(gpu, cpu)
        #expect(maxError < 1e-5)
    }
#endif
}
