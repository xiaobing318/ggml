// swift-tools-version: 5.5
// 指定 Swift 工具链的版本，这里使用的是 5.5 版本

// 导入 PackageDescription 模块，这是定义 Swift 包所需的模块
import PackageDescription
// 创建一个 Package 实例，这个实例描述了 Swift 包的名称、支持的平台、包含的产品和目标等信息
let package = Package(
    name: "ggml",           // 包的名称
    platforms: [            // 支持的平台
        .macOS(.v12),       // 支持的macOS版本最低是12.0
        .iOS(.v14),         // 支持的iOS版本最低是14.0
        .watchOS(.v4),      // 支持的watchOS版本最低是4.0
        .tvOS(.v14)         // 支持的tvOS版本最低是14.0
    ],
    products: [
        .library(name: "ggml", targets: ["ggml"]),  // 包含的产品，这里是一个库
    ],
    targets: [
        .target(
            name: "ggml",                   // 目标的名称
            path: ".",                      // 目标的路径
            exclude: [],                    // 排除的文件
            sources: [          
                "src/ggml.c",               // 源文件src/ggml.c
                "src/ggml-alloc.c",         // 源文件src/ggml-alloc.c
                "src/ggml-backend.cpp",     // 源文件src/ggml-backend.cpp
                "src/ggml-quants.c",        // 源文件src/ggml-quants.c
                "src/ggml-metal.m",         // 源文件src/ggml-metal.m
            ],
            resources: [
                // 添加 Metal 文件, 用于编译 Metal Shader
                .process("src/ggml-metal.metal")
            ],
            publicHeadersPath: "spm-headers",   // 公共头文件的路径
            cSettings: [
                //  C 语言编译设置
                .unsafeFlags(["-Wno-shorten-64-to-32", "-O3", "-DNDEBUG"]),
                .define("GGML_USE_ACCELERATE"),
                .unsafeFlags(["-fno-objc-arc"]),
                .define("GGML_USE_METAL"),
                // NOTE: NEW_LAPACK will required iOS version 16.4+
                // We should consider add this in the future when we drop support for iOS 14
                // (ref: ref: https://developer.apple.com/documentation/accelerate/1513264-cblas_sgemm?language=objc)
                // .define("ACCELERATE_NEW_LAPACK"),
                // .define("ACCELERATE_LAPACK_ILP64")
            ],
            linkerSettings: [
                .linkedFramework("Accelerate")
            ]
        )
    ],
    cxxLanguageStandard: .cxx11
)
