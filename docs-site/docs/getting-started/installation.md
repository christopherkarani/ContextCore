---
sidebar_position: 1
---

# Installation

## Requirements

- **Swift 6.2** or later
- **Platforms:** iOS 17+ / macOS 14+ / visionOS 1+
- **Metal-capable device** -- GPU acceleration is required for scoring engines

## Swift Package Manager

Add ContextCore to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/christopherkarani/ContextCore.git", from: "1.0.0")
]
```

Then add the product to your target's dependencies:

```swift
.target(
    name: "YourApp",
    dependencies: [
        .product(name: "ContextCore", package: "ContextCore")
    ]
)
```

## Verify the Installation

```swift
import ContextCore
```

Build your project to confirm everything resolves correctly.

## Simulator Support

ContextCore requires a Metal GPU for its scoring engines. On the iOS Simulator (which lacks a real GPU), the framework will fall back gracefully. For full performance testing, use a physical device or a Mac target.
