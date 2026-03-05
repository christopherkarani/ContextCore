import Foundation

public enum ContextCoreError: Error, Sendable, Equatable {
    case embeddingFailed(String)
    case storeFull
    case tokenBudgetTooSmall
    case sessionNotStarted
    case compressionFailed(String)
    case checkpointCorrupt
    case metalDeviceUnavailable
    case dimensionMismatch(expected: Int, got: Int)
    case chunkNotFound(id: UUID)
}
