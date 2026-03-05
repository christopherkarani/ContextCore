import Foundation

public protocol TokenCounter: Sendable {
    func count(_ text: String) -> Int
}

public struct ApproximateTokenCounter: TokenCounter, Sendable {
    public init() {}

    public func count(_ text: String) -> Int {
        guard !text.isEmpty else {
            return 0
        }

        let words = text.split { character in
            !(character.isLetter || character.isNumber)
        }

        guard !words.isEmpty else {
            return 0
        }

        return Int(ceil(Double(words.count) * 1.3))
    }
}
