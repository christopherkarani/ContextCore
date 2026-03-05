// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "ContextCore",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
        .visionOS(.v1),
    ],
    products: [
        .library(
            name: "ContextCore",
            targets: ["ContextCore"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/christopherkarani/MetalANNS.git", branch: "main"),
    ],
    targets: [
        .target(
            name: "ContextCoreShaders",
            path: "Sources/ContextCoreShaders",
            resources: [.process("Shaders")],
            linkerSettings: [
                .linkedFramework("Metal"),
            ]
        ),
        .target(
            name: "ContextCoreEngine",
            dependencies: [
                "ContextCoreShaders",
                .product(name: "MetalANNS", package: "MetalANNS"),
            ],
            path: "Sources/ContextCoreEngine",
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("CoreML"),
                .linkedFramework("Accelerate"),
            ]
        ),
        .target(
            name: "ContextCore",
            dependencies: [
                "ContextCoreEngine",
                .product(name: "MetalANNS", package: "MetalANNS"),
            ],
            path: "Sources/ContextCore",
            resources: [.process("Resources")],
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("CoreML"),
                .linkedFramework("Accelerate"),
            ]
        ),
        .testTarget(
            name: "ContextCoreTests",
            dependencies: ["ContextCore"],
            path: "Tests/ContextCoreTests"
        ),
    ]
)
