#[[
1. 配置作用于 ggml 项目上的 CMake 版本最低不能小于 3.14 即 CMake 的版本大于等于 3.14 ，这样可以确保 CMakeLists.txt 和模块文件中使用的语法可以被正确的解析、执行。
2. 配置整体项目名称以及使用的编程语言，这些内容是在仓库的顶层 CMakeLists.txt 文件中设置，而目前的 CMakeLists.txt 是顶层文件。
]]
cmake_minimum_required(VERSION 3.14) # for add_link_options and implicit target directories.
project("ggml" C CXX ASM)

### GGML Version
#[[
1. 配置 ggml 仓库的版本信息，分为 major/minor/patch 三个部分。
2. 通过 major/minor/patch 三个部分拼接完整的字符串类型的版本信息。
]]
set(GGML_VERSION_MAJOR 0)
set(GGML_VERSION_MINOR 9)
set(GGML_VERSION_PATCH 4)
set(GGML_VERSION_BASE "${GGML_VERSION_MAJOR}.${GGML_VERSION_MINOR}.${GGML_VERSION_PATCH}")

#[[
1. 在当前系统的搜索路径里，找名字叫 git 或 git.exe 的可执行文件，把找到的绝对路径存到变量 GIT_EXE 中，搜索时不要被 CMake 的“交叉编译根路径（CMAKE_FIND_ROOT_PATH）”影响，如果找不到，GIT_EXE 会变成 GIT_EXE-NOTFOUND 。
2. 如果在当前系统中找到了 git 或者 git.exe 可执行文件则执行两个 git/git.exe 相关的命令从而获取一些信息，execute_process 是 CMake 内置的一条命令，用来在配置阶段执行外部可执行文件的，例如执行 gcc/python/git 可执行文件，将可执行文件执行后的输出结果保存到指定变量中，以供后续使用。
]]
find_program(GIT_EXE NAMES git git.exe NO_CMAKE_FIND_ROOT_PATH)
if(GIT_EXE)
    # Get current git commit hash
    execute_process(COMMAND ${GIT_EXE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE GGML_BUILD_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    # Check if the working directory is dirty (i.e., has uncommitted changes)
    execute_process(COMMAND ${GIT_EXE} diff-index --quiet HEAD -- .
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        RESULT_VARIABLE GGML_GIT_DIRTY
        ERROR_QUIET
    )
endif()

#[[
1. 在当前目录/作用域中创建一个普通变量用来保存 ggml 的版本信息。
]]
set(GGML_VERSION "${GGML_VERSION_BASE}")

#[[
1. 如果之前 CMake 调用外部可执行文件没有获取得到 ggml 仓库的 commit 信息则将指定变量设置为 unknown 。
]]
if(NOT GGML_BUILD_COMMIT)
    set(GGML_BUILD_COMMIT "unknown")
endif()

# Build the commit string with optional dirty flag
#[[
1. 如果定义了 GGML_GIT_DIRTY 变量并且该变量的值为 1 ，则将字符串变量 GGML_BUILD_COMMIT 赋值为拼接后的值。
]]
if(DEFINED GGML_GIT_DIRTY AND GGML_GIT_DIRTY EQUAL 1)
    set(GGML_BUILD_COMMIT "${GGML_BUILD_COMMIT}-dirty")
endif()

#[[
1. 引入 CMake 自带模块 CheckIncludeFileCXX.cmake。
2. 该模块提供 CHECK_INCLUDE_FILE_CXX() 宏，会尝试编译一个最小的 C++ 测试程序来包含指定头文件，用于检测当前编译器/平台上是否存在该头文件，并把结果记录到缓存变量/宏中，方便后续做条件编译。
3. 目前该模块文件在该项目中没有被使用，可能曾经使用过。
]]
include(CheckIncludeFileCXX)

#[[
1. 配置 CMake 内置的一个功能：在生成构建系统时，额外生成一个 compile_commands.json 文件
2. 导出的形式是 compile_commands.json 文件形式，该文件中记录了对每一个编译单元使用的编译命令是什么。
3. 这个功能只对支持的 generator 生效，比如 Makefiles / Ninja；对 Visual Studio、Xcode 等 generator 会被忽略，因为如果想要针对这些 generator 达到同样的效果付出的成本是很大的。
]]
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#[[
1. 如果构建、编译使用的不是 XCODE/MSVC 编译器工具链并且没有创建 CMAKE_BUILD_TYPE 变量，那么默认指定构建类型是 Release 类型。
2. 这部分设置是根据不同的编译器工具链进行的设置，而不是操作系统平台。
]]
if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

#[[
1. 如果 ggml 不是作为另外项目的依赖被构建、编译，那么将 GGML_STANDALONE 配置为 ON ，反之则配置为 OFF。
2. 其中 CMAKE_BINARY_DIR 是在 CMake-GUI 界面中选择 "Where to build the binaries" 的目录路径。
]]
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(GGML_STANDALONE ON)

    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

    # configure project version
    # TODO
else()
    set(GGML_STANDALONE OFF)
endif()

#[[
1. 根据不同的构建、编译平台进行不同的配置：
  1.1 Emscripten 编译器工具链：如果使用 Emscripten 编译器工具链构建 ggml 项目，那么编译 ggml 项目使用静态库的方式而不是动态库的方式，因为静态库的方式更加的简单可靠，并且在 Emscripten 下，默认把 wasm 内容塞进一个 ggml.js 里，变成“单文件部署”模式，方便在 web 上分发和加载。
  1.2 Mingw 编译器工具链：如果使用 Mingw 编译器工具链构建 ggml 项目，那么编译 ggml 项目使用静态库的方式而不是动态库的方式，因为静态库的方式更加的简单可靠。
  1.3 其它编译器工具链：如果使用其它编译器工具链构建 ggml 项目，那么编译 ggml 项目使用动态库的方式而不是静态库的方式，因为本地运行 ggml 项目的动态库更具有优势。
2. Emscripten 是一个开源的编译器工具链，主要功能是将 C/C++ 代码编译成 WebAssembly (Wasm) 模块和相应的 JavaScript“胶水”代码，使得这些原生代码能够在 Web 浏览器中或任何支持 WebAssembly 运行时的环境中以接近原生的速度运行。
]]
if (EMSCRIPTEN)
    set(BUILD_SHARED_LIBS_DEFAULT OFF)

    option(GGML_WASM_SINGLE_FILE "ggml: embed WASM inside the generated ggml.js" ON)
else()
    if (MINGW)
        set(BUILD_SHARED_LIBS_DEFAULT OFF)
    else()
        set(BUILD_SHARED_LIBS_DEFAULT ON)
    endif()
endif()

# remove the lib prefix on win32 mingw
#[[
1. CMake 中是通过例如 WIN32/UNIX/APPLE 来区分不同的操作系统，因此这部分的变量赋值则是针对 Windows 平台的。
2. CMAKE_STATIC_LIBRARY_PREFIX、CMAKE_SHARED_LIBRARY_PREFIX、CMAKE_SHARED_MODULE_PREFIX 都是 CMake 的内置变量/缓存键，控制生成静态库、共享库、模块库的文件名前缀。不同平台/工具链有默认值（Unix 上通常是 lib，Windows/MSVC 通常是空）。通过设置它们为空，可以覆盖默认的 lib 前缀。
3. 如果在 Windows 操作系统平台上使用 MinGW/Unix 工具链，那么 MinGW/Unix 工具链根据产物命名规则将会往产物默认添加lib字符串，例如在 Windows + MSVC 组合下编译产物命名为 ggml.dll，而在 Windows + MinGW 组合下编译产物命名为 libggml.dll/libggml.a/libggml.dll.a 之类的。
]]
if (WIN32)
    set(CMAKE_STATIC_LIBRARY_PREFIX "")
    set(CMAKE_SHARED_LIBRARY_PREFIX "")
    set(CMAKE_SHARED_MODULE_PREFIX  "")
endif()

#[[
1. GGML_BACKEND_DL 意味着 CPU 后端要做成可加载的动态插件，预期可以被不同机器/进程加载。GGML_NATIVE 会给这个插件加上 -march=native 等本机特性，生成的二进制只能在构建机的 ISA 上运行，一旦插件被拿到别的 CPU 上就可能崩/非法指令，所以被判定为不兼容。设计意图是：做成可分发的动态后端时必须保证可移植性，不能捆绑本机专属指令集。替代方案是 GGML_CPU_ALL_VARIANTS：构建多套 ISA 变体（baseline/AVX2/AVX512 等），运行时选择合适的变体，从而既能做动态加载，又能利用高级指令。若你只在单机使用，也可关掉 GGML_BACKEND_DL 或关掉 GGML_NATIVE（默认是打开的状态）。
2. 动态链接库：可执行文件里只留下“我需要某个库里哪些符号”的信息，运行时由操作系统的动态链接器去把库文件映射进来。
]]
option(BUILD_SHARED_LIBS           "ggml: build shared libraries" ${BUILD_SHARED_LIBS_DEFAULT})
option(GGML_BACKEND_DL             "ggml: build backends as dynamic libraries (requires BUILD_SHARED_LIBS)" OFF)
set(GGML_BACKEND_DIR "" CACHE PATH "ggml: directory to load dynamic backends from (requires GGML_BACKEND_DL")

#
# option list
#

# TODO: mark all options as advanced when not GGML_STANDALONE

#[[
1. 根据不同的操作系统平台进行变量配置：
  1.1 如果操作系统是 macOS/IOS 那么开启和该平台相关联的配置并且将 BLAS 实现设置为 Apple 。
  1.2 如果操作系统是 Windows/Linux 那么关闭和该平台相关联的配置并且将 BLAS 实现设置为 Generic ，这里的 Generic 意味着不要特指，用找到的任意实现/参考实现。
2. BLAS 是 Basic Linear Algebra Subprograms 的缩写，一组统一接口的基础线性代数例程规范，主要包含向量/矩阵的加减、标量乘、矩阵乘（GEMM）、求解三角/对称系统等。各平台/供应商会提供各自的高性能实现（如 Intel MKL、OpenBLAS、Apple Accelerate/vecLib、Netlib BLAS 等），CMake 中的 GGML_BLAS_* 选项就是决定是否启用 BLAS 后端以及选择哪家实现，以便把部分算子交给这些高度优化的库来加速。
3. 可以参考 ggml/docs/selfLerning-BLAS.md 文档中的内容。
]]
if (APPLE)
    set(GGML_METAL_DEFAULT ON)
    set(GGML_BLAS_DEFAULT ON)
    set(GGML_BLAS_VENDOR_DEFAULT "Apple")
else()
    set(GGML_METAL_DEFAULT OFF)
    set(GGML_BLAS_DEFAULT OFF)
    set(GGML_BLAS_VENDOR_DEFAULT "Generic")
endif()

#[[
1. 在交叉编译或启用可复现构建(SOURCE_DATE_EPOCH)时，默认关闭本机 CPU 特性优化(GGML_NATIVE_DEFAULT=OFF)，其他情况下默认开启，以避免交叉环境/可复现构建中使用 -march=native 等不稳定配置。
2. 如果 GGML_NATIVE_DEFAULT=ON 那么意味着根据当前机器上所具有的一些特性进行编译。例如当前机器支持 AVX512 指令，那么开启 GGML_NATIVE_DEFAULT 意味着可以获得更好的性能。对于使用同一 ISA 的 CPU 来说，开启 GGML_NATIVE_DEFAULT 并不是一个好的选择，因为如果源CPU 上拥有的特性更丰富，而目标 CPU 上拥有的特性不如源 CPU 上的特性丰富，将源 CPU 上编译的 ggml 拷贝到 目标 CPU 上运行可能会出现问题。在发布版本的时候可以关闭 GGML_NATIVE_DEFAULT 从而使得编译产物更加通用（可以打开 GGML_CPU_ALL_VARIANTS ）。
]]
if (CMAKE_CROSSCOMPILING OR DEFINED ENV{SOURCE_DATE_EPOCH})
    message(STATUS "Setting GGML_NATIVE_DEFAULT to OFF")
    set(GGML_NATIVE_DEFAULT OFF)
else()
    set(GGML_NATIVE_DEFAULT ON)
endif()

# defaults
#[[
1. 如果 GGML_LLAMAFILE_DEFAULT 没有被创建则创建该变量并且赋值为 OFF ,该变量将会作为 GGML_LLAMAFILE 的赋值对象，默认情况下该变量都是没有被创建的。
]]
if (NOT GGML_LLAMAFILE_DEFAULT)
    set(GGML_LLAMAFILE_DEFAULT OFF)
endif()

#[[
1. 如果 GGML_CUDA_GRAPHS_DEFAULT 没有被创建则创建该变量并且赋值为 OFF ，该变量将会作为 GGML_CUDA_GRAPHS 的赋值对象，默认情况下该变量都是没有被创建的。
2. 在推理（尤其 llama.cpp 的重复图）时对一整段 CUDA kernel 进行 Graph capture + 复用，减少每步的 kernel launch 和同步开销，提升吞吐。默认 OFF ，因为 CUDA Graph 需要较新驱动/硬件且对图形结构稳定性敏感；部分节点/动态 shape 会触发禁用或退回普通路径，某些 GPU 架构或频繁更新图会导致捕获失败或收益不明显。
]]
if (NOT GGML_CUDA_GRAPHS_DEFAULT)
    set(GGML_CUDA_GRAPHS_DEFAULT OFF)
endif()

# general
option(GGML_STATIC "ggml: static link libraries"                     OFF)
option(GGML_NATIVE "ggml: optimize the build for the current system" ${GGML_NATIVE_DEFAULT})
option(GGML_LTO    "ggml: enable link time optimization"             OFF)
option(GGML_CCACHE "ggml: use ccache if available"                   ON)

# debug
option(GGML_ALL_WARNINGS           "ggml: enable all compiler warnings"                   ON)
option(GGML_ALL_WARNINGS_3RD_PARTY "ggml: enable all compiler warnings in 3rd party libs" OFF)
option(GGML_GPROF                  "ggml: enable gprof"                                   OFF)

# build
option(GGML_FATAL_WARNINGS    "ggml: enable -Werror flag"    OFF)

# sanitizers
option(GGML_SANITIZE_THREAD    "ggml: enable thread sanitizer"    OFF)
option(GGML_SANITIZE_ADDRESS   "ggml: enable address sanitizer"   OFF)
option(GGML_SANITIZE_UNDEFINED "ggml: enable undefined sanitizer" OFF)

# instruction set specific
#[[
1. “默认关掉 SSE/AVX 选项”并不等于编译产物不用这些指令；因为此时 GGML_NATIVE 已经是 ON，编译器会通过 -march=native（或 MSVC/探测逻辑）自动打开本机支持的 SSE/AVX/FMA 等特性，不需要再逐个选项手动勾。
2. 典型场景（非交叉/非可复现）：GGML_NATIVE_DEFAULT=ON -> GGML_NATIVE=ON -> 走 -march=native，INS_ENB=OFF 只是让手动的 GGML_SSE42/AVX/… 选项默认不再重复勾；指令仍然会用，只是由编译器根据本机能力决定。如果你关掉 GGML_NATIVE（为了做通用/可分发二进制），这时需要手动挑指令集，所以让 INS_ENB=ON，默认把常见的 SSE4.2/AVX/AVX2/BMI2/FMA/F16C 选项打开，方便“较新机型通用”的构建；需要更保守时可手动关闭。若是交叉编译或可复现构建（GGML_NATIVE_DEFAULT=OFF），两者条件触发，INS_ENB 也变 OFF，避免误用目标机不一定有的指令。
3. 这段逻辑设计动机无法理解。
]]
if (GGML_NATIVE OR NOT GGML_NATIVE_DEFAULT)
    set(INS_ENB OFF)
else()
    set(INS_ENB ON)
endif()

#[[
1. message(DEBUG …) 只有在日志级别 >= DEBUG 时才会打印；CMake 的默认级别是 NOTICE，所以这三行被静默。用 CMake-GUI 点击 Configure 时没有把日志级别调高，因此看不到输出。要看它们，可以改成 message(STATUS …)，或在配置时把 CMAKE_MESSAGE_LOG_LEVEL 设为 DEBUG（在 GUI 里添加该缓存变量，或命令行 -DCMAKE_MESSAGE_LOG_LEVEL=DEBUG/cmake --log-level=DEBUG）。
]]
message(DEBUG "GGML_NATIVE         : ${GGML_NATIVE}")
message(DEBUG "GGML_NATIVE_DEFAULT : ${GGML_NATIVE_DEFAULT}")
message(DEBUG "INS_ENB             : ${INS_ENB}")

option(GGML_CPU_HBM          "ggml: use memkind for CPU HBM" OFF)
option(GGML_CPU_REPACK       "ggml: use runtime weight conversion of Q4_0 to Q4_X_X" ON)
option(GGML_CPU_KLEIDIAI     "ggml: use KleidiAI optimized kernels if applicable" OFF)
option(GGML_SSE42            "ggml: enable SSE 4.2"          ${INS_ENB})
option(GGML_AVX              "ggml: enable AVX"              ${INS_ENB})
option(GGML_AVX_VNNI         "ggml: enable AVX-VNNI"         OFF)
option(GGML_AVX2             "ggml: enable AVX2"             ${INS_ENB})
option(GGML_BMI2             "ggml: enable BMI2"             ${INS_ENB})
option(GGML_AVX512           "ggml: enable AVX512F"          OFF)
option(GGML_AVX512_VBMI      "ggml: enable AVX512-VBMI"      OFF)
option(GGML_AVX512_VNNI      "ggml: enable AVX512-VNNI"      OFF)
option(GGML_AVX512_BF16      "ggml: enable AVX512-BF16"      OFF)
if (NOT MSVC)
    # in MSVC F16C and FMA is implied with AVX2/AVX512
    option(GGML_FMA          "ggml: enable FMA"              ${INS_ENB})
    option(GGML_F16C         "ggml: enable F16C"             ${INS_ENB})
    # MSVC does not seem to support AMX
    option(GGML_AMX_TILE     "ggml: enable AMX-TILE"         OFF)
    option(GGML_AMX_INT8     "ggml: enable AMX-INT8"         OFF)
    option(GGML_AMX_BF16     "ggml: enable AMX-BF16"         OFF)
endif()
option(GGML_LASX             "ggml: enable lasx"             ON)
option(GGML_LSX              "ggml: enable lsx"              ON)
option(GGML_RVV              "ggml: enable rvv"              ON)
option(GGML_RV_ZFH           "ggml: enable riscv zfh"        ON)
option(GGML_RV_ZVFH          "ggml: enable riscv zvfh"       ON)
option(GGML_RV_ZICBOP        "ggml: enable riscv zicbop"     ON)
option(GGML_XTHEADVECTOR     "ggml: enable xtheadvector"     OFF)
option(GGML_VXE              "ggml: enable vxe"              ${GGML_NATIVE})

option(GGML_CPU_ALL_VARIANTS "ggml: build all variants of the CPU backend (requires GGML_BACKEND_DL)" OFF)
set(GGML_CPU_ARM_ARCH        "" CACHE STRING "ggml: CPU architecture for ARM")
set(GGML_CPU_POWERPC_CPUTYPE "" CACHE STRING "ggml: CPU type for PowerPC")


if (MINGW)
    set(GGML_WIN_VER "0xA00" CACHE STRING   "ggml: Windows version")
endif()

# ggml core
set(GGML_SCHED_MAX_COPIES  "4" CACHE STRING "ggml: max input copies for pipeline parallelism")
option(GGML_CPU                             "ggml: enable CPU backend"                        ON)

# 3rd party libs / backends
option(GGML_ACCELERATE                      "ggml: enable Accelerate framework"               ON)
option(GGML_BLAS                            "ggml: use BLAS"                                  ${GGML_BLAS_DEFAULT})
set(GGML_BLAS_VENDOR ${GGML_BLAS_VENDOR_DEFAULT} CACHE STRING
                                            "ggml: BLAS library vendor")
option(GGML_LLAMAFILE                       "ggml: use LLAMAFILE"                             ${GGML_LLAMAFILE_DEFAULT})

option(GGML_CUDA                            "ggml: use CUDA"                                  OFF)
option(GGML_MUSA                            "ggml: use MUSA"                                  OFF)
option(GGML_CUDA_FORCE_MMQ                  "ggml: use mmq kernels instead of cuBLAS"         OFF)
option(GGML_CUDA_FORCE_CUBLAS               "ggml: always use cuBLAS instead of mmq kernels"  OFF)
set   (GGML_CUDA_PEER_MAX_BATCH_SIZE "128" CACHE STRING
                                            "ggml: max. batch size for using peer access")
option(GGML_CUDA_NO_PEER_COPY               "ggml: do not use peer to peer copies"            OFF)
option(GGML_CUDA_NO_VMM                     "ggml: do not try to use CUDA VMM"                OFF)
option(GGML_CUDA_FA                         "ggml: compile ggml FlashAttention CUDA kernels"  ON)
option(GGML_CUDA_FA_ALL_QUANTS              "ggml: compile all quants for FlashAttention"     OFF)
option(GGML_CUDA_GRAPHS                     "ggml: use CUDA graphs (llama.cpp only)"          ${GGML_CUDA_GRAPHS_DEFAULT})
set   (GGML_CUDA_COMPRESSION_MODE "size" CACHE STRING
                                            "ggml: cuda link binary compression mode; requires cuda 12.8+")
set_property(CACHE GGML_CUDA_COMPRESSION_MODE PROPERTY STRINGS "none;speed;balance;size")

option(GGML_HIP                             "ggml: use HIP"                                   OFF)
option(GGML_HIP_GRAPHS                      "ggml: use HIP graph, experimental, slow"         OFF)
option(GGML_HIP_NO_VMM                      "ggml: do not try to use HIP VMM"                 ON)
option(GGML_HIP_ROCWMMA_FATTN               "ggml: enable rocWMMA for FlashAttention"         OFF)
option(GGML_HIP_MMQ_MFMA                    "ggml: enable MFMA MMA for CDNA in MMQ"           ON)
option(GGML_HIP_EXPORT_METRICS              "ggml: enable kernel perf metrics output"         OFF)
option(GGML_MUSA_GRAPHS                     "ggml: use MUSA graph, experimental, unstable"    OFF)
option(GGML_MUSA_MUDNN_COPY                 "ggml: enable muDNN for accelerated copy"         OFF)
option(GGML_VULKAN                          "ggml: use Vulkan"                                OFF)
option(GGML_VULKAN_CHECK_RESULTS            "ggml: run Vulkan op checks"                      OFF)
option(GGML_VULKAN_DEBUG                    "ggml: enable Vulkan debug output"                OFF)
option(GGML_VULKAN_MEMORY_DEBUG             "ggml: enable Vulkan memory debug output"         OFF)
option(GGML_VULKAN_SHADER_DEBUG_INFO        "ggml: enable Vulkan shader debug info"           OFF)
option(GGML_VULKAN_VALIDATE                 "ggml: enable Vulkan validation"                  OFF)
option(GGML_VULKAN_RUN_TESTS                "ggml: run Vulkan tests"                          OFF)
option(GGML_WEBGPU                          "ggml: use WebGPU"                                OFF)
option(GGML_WEBGPU_DEBUG                    "ggml: enable WebGPU debug output"                OFF)
option(GGML_WEBGPU_CPU_PROFILE              "ggml: enable WebGPU profiling (CPU)"             OFF)
option(GGML_WEBGPU_GPU_PROFILE              "ggml: enable WebGPU profiling (GPU)"             OFF)

option(GGML_ZDNN                            "ggml: use zDNN"                                  OFF)
option(GGML_METAL                           "ggml: use Metal"                                 ${GGML_METAL_DEFAULT})
option(GGML_METAL_NDEBUG                    "ggml: disable Metal debugging"                   OFF)
option(GGML_METAL_SHADER_DEBUG              "ggml: compile Metal with -fno-fast-math"         OFF)
option(GGML_METAL_EMBED_LIBRARY             "ggml: embed Metal library"                       ${GGML_METAL})
set   (GGML_METAL_MACOSX_VERSION_MIN "" CACHE STRING
                                            "ggml: metal minimum macOS version")
set   (GGML_METAL_STD "" CACHE STRING       "ggml: metal standard version (-std flag)")
option(GGML_OPENMP                          "ggml: use OpenMP"                                ON)
option(GGML_RPC                             "ggml: use RPC"                                   OFF)
option(GGML_SYCL                            "ggml: use SYCL"                                  OFF)
option(GGML_SYCL_F16                        "ggml: use 16 bit floats for sycl calculations"   OFF)
option(GGML_SYCL_GRAPH                      "ggml: enable graphs in the SYCL backend"         ON)
option(GGML_SYCL_DNN                        "ggml: enable oneDNN in the SYCL backend"         ON)
set   (GGML_SYCL_TARGET "INTEL" CACHE STRING
                                            "ggml: sycl target device")
set   (GGML_SYCL_DEVICE_ARCH "" CACHE STRING
                                            "ggml: sycl device architecture")

option(GGML_OPENCL                          "ggml: use OpenCL"                                OFF)
option(GGML_OPENCL_PROFILING                "ggml: use OpenCL profiling (increases overhead)" OFF)
option(GGML_OPENCL_EMBED_KERNELS            "ggml: embed kernels"                             ON)
option(GGML_OPENCL_USE_ADRENO_KERNELS       "ggml: use optimized kernels for Adreno"          ON)
set   (GGML_OPENCL_TARGET_VERSION "300" CACHE STRING
                                            "gmml: OpenCL API version to target")

#[[
1. Hexagon 指的是 Qualcomm SoC 里的 DSP 架构/核（Qualcomm Hexagon），常用于安卓手机/平板等移动设备的信号处理和 NN 加速。这个开关是为了给 Hexagon DSP 做专用 backend（需配合 Qualcomm QNN/SDK 等）。因此它面向手机/平板等含 Qualcomm SoC 的设备，不是面向常规 PC/Server CPU。
]]
option(GGML_HEXAGON                         "ggml: enable Hexagon backend"                    OFF)

# toolchain for vulkan-shaders-gen
set   (GGML_VULKAN_SHADERS_GEN_TOOLCHAIN "" CACHE FILEPATH "ggml: toolchain file for vulkan-shaders-gen")

# extra artifacts
option(GGML_BUILD_TESTS    "ggml: build tests"    ${GGML_STANDALONE})
option(GGML_BUILD_EXAMPLES "ggml: build examples" ${GGML_STANDALONE})

#
# dependencies
#

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

set(THREADS_PREFER_PTHREAD_FLAG ON)

#[[
1. 查找并启用 CMake 自带的 Threads 模块以获取线程库/编译选项，REQUIRED 表示若未找到线程支持则中止配置。
]]
find_package(Threads REQUIRED)

#[[
1. 引入 CMake 自带的 GNUInstallDirs 模块，定义标准安装目录变量(如 CMAKE_INSTALL_BINDIR/CMAKE_INSTALL_LIBDIR 等)供后续 install() 使用。
]]
include(GNUInstallDirs)

#
# build the library
#

add_subdirectory(src)

#
# tests and examples
#

if (GGML_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()

if (GGML_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif ()

#
# install
#

#[[
1. 引入 CMakePackageConfigHelpers 模块，用于生成/安装 GGML 的 package 配置文件(如 ggml-config.cmake、ggml-version.cmake)，以便其他项目通过 find_package 使用 GGML。
]]
include(CMakePackageConfigHelpers)

# all public headers
#[[
1. 创建字符串列表变量用来保存作为公开头文件列表，以供其它项目依赖 ggml 所需要的头文件，这些头文件在 install 阶段将会被拷贝到指定安装目录中。
]]
set(GGML_PUBLIC_HEADERS
    include/ggml.h
    include/ggml-cpu.h
    include/ggml-alloc.h
    include/ggml-backend.h
    include/ggml-blas.h
    include/ggml-cann.h
    include/ggml-cpp.h
    include/ggml-cuda.h
    include/ggml-opt.h
    include/ggml-metal.h
    include/ggml-rpc.h
    include/ggml-sycl.h
    include/ggml-vulkan.h
    include/ggml-webgpu.h
    include/gguf.h)

set_target_properties(ggml PROPERTIES PUBLIC_HEADER "${GGML_PUBLIC_HEADERS}")
#if (GGML_METAL)
#    set_target_properties(ggml PROPERTIES RESOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/ggml-metal.metal")
#endif()
#[[
1. 安装 ggml 目标：把生成的共享库安装到库目录，并同时安装该目标声明的 PUBLIC_HEADER 头文件。
2. 安装 ggml-base 目标：只安装其生成的共享库到库目录（不额外安装头文件）。
3. 提示：
  3.1 TARGETS ggml / ggml-base：指的是前面用 add_library(ggml ...) / add_library(ggml-base ...) 定义的库目标。
  3.2 LIBRARY：表示安装“库文件”（在 Linux 是 .so，macOS 是 .dylib，Windows 通常参与到 .dll/.lib 的安装策略里）。
  3.3 PUBLIC_HEADER：会安装这个目标在 PUBLIC_HEADER 属性中列出的头文件，常用来对外暴露 API。
4. 顶层的 install(TARGETS ggml …)/install(TARGETS ggml-base …) 是给在子目录里定义的库装的。CMakeLists.txt 里有 add_subdirectory(src)，而 src/CMakeLists.txt 里确实有 add_library(ggml-base …)、add_library(ggml …)（以及 add_library(ggml::ggml ALIAS ggml)）。CMake 处理完 add_subdirectory(src) 后，这些目标会向上可见，所以在顶层安装它们没有问题。只要保持 add_subdirectory(src) 出现在安装命令之前即可当前顺序已经满足。
]]
install(TARGETS ggml LIBRARY PUBLIC_HEADER)
install(TARGETS ggml-base LIBRARY)

#[[
1. 当 ggml 以“独立库”（standalone）方式构建时：
  1.1 用模板文件 ggml.pc.in 生成 ggml.pc（pkg-config 描述文件）。
  1.2 在安装阶段把这个 ggml.pc 安装到 share/pkgconfig，方便别的项目通过 pkg-config 找到 ggml。
2. CMake 中的安装逻辑目前理解不深。
]]
if (GGML_STANDALONE)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ggml.pc.in
        ${CMAKE_CURRENT_BINARY_DIR}/ggml.pc
        @ONLY)

    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ggml.pc
        DESTINATION share/pkgconfig)
endif()

#
# Create CMake package
#



# Capture variables prefixed with GGML_.

set(variable_set_statements
"
####### Expanded from @GGML_VARIABLES_EXPANED@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run        #######

")

set(GGML_SHARED_LIB ${BUILD_SHARED_LIBS})

#[[
1. 获取当前目录/作用域中定义变量的列表，即当前目录/作用域中所有定义的变量，将其保存到 all_varibles ，然后对这些变量逐个进行处理。现在把所有 GGML_ 开头的配置变量都记录下来，生成一段能重新设置它们的 CMake 脚本，打包进 ggml-config.cmake 之类的文件里，以后别人 find_package 时就能恢复我们当时的配置环境。
2. ${variable_name} 保存的是变量的名称，而 ${${variable_name}} 保存的则是变量名称对应的值。
]]
get_cmake_property(all_variables VARIABLES)
foreach(variable_name IN LISTS all_variables)
    if(variable_name MATCHES "^GGML_")
        string(REPLACE ";" "\\;"
               variable_value "${${variable_name}}")

        set(variable_set_statements
            "${variable_set_statements}set(${variable_name} \"${variable_value}\")\n")
    endif()
endforeach()

set(GGML_VARIABLES_EXPANDED ${variable_set_statements})

# Create the CMake package and set install location.

#[[
1. 记录要安装的 ggml 版本号，并定义 3 个可配置的安装路径变量：
   - GGML_INCLUDE_INSTALL_DIR：头文件安装目录
   - GGML_LIB_INSTALL_DIR    ：库文件安装目录
   - GGML_BIN_INSTALL_DIR    ：可执行文件安装目录
2. 这三个目录变量默认取自 CMAKE_INSTALL_INCLUDEDIR / CMAKE_INSTALL_LIBDIR / CMAKE_INSTALL_BINDIR（这些通常由 GNUInstallDirs 模块或 CMake 默认值提供），在 CMake GUI 或命令行中可以被用户修改。
]]
set(GGML_INSTALL_VERSION ${GGML_VERSION})
set(GGML_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH "Location of header  files")
set(GGML_LIB_INSTALL_DIR     ${CMAKE_INSTALL_LIBDIR}     CACHE PATH "Location of library files")
set(GGML_BIN_INSTALL_DIR     ${CMAKE_INSTALL_BINDIR}     CACHE PATH "Location of binary  files")

#[[
1. 这两行使用 CMakePackageConfigHelpers 模块提供的工具函数生成 ggml 的 package 配置文件，以便其他项目可以通过 find_package(ggml) 来找到并使用已安装的 ggml：
   - configure_package_config_file() 基于模板 ggml-config.cmake.in 生成 ggml-config.cmake，写入头文件/库/可执行文件的安装路径等信息；
   - write_basic_package_version_file() 生成 ggml-version.cmake，提供版本号和版本兼容性信息。
2. INSTALL_DESTINATION 指定未来安装时这两个配置文件将被放到的目录前缀；PATH_VARS 中的 GGML_INCLUDE_INSTALL_DIR / GGML_LIB_INSTALL_DIR / GGML_BIN_INSTALL_DIR会在生成 ggml-config.cmake 时被自动处理为与安装前缀兼容的路径；COMPATIBILITY SameMajorVersion 表示只要主版本号一致，就认为版本是兼容的。
]]
configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ggml-config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/ggml-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ggml
    PATH_VARS GGML_INCLUDE_INSTALL_DIR
              GGML_LIB_INSTALL_DIR
              GGML_BIN_INSTALL_DIR)

write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/ggml-version.cmake
    VERSION ${GGML_INSTALL_VERSION}
    COMPATIBILITY SameMajorVersion)

#[[
1. 给编译系统添加两个预处理宏，这两个宏在具体的代码中会被使用到而不是用在动态库属性中，例如在代码中以字符串等形式出现，在运行时能知道自己是哪个版本、哪个 git 提交，以及在日志 / about 信息里显示，用的就是这俩宏。可以通过给编译器添加宏从而控制编译代码的行为，类似直接通过 -DXXX 的方式传递宏定义。
]]
target_compile_definitions(ggml-base PRIVATE
    GGML_VERSION="${GGML_INSTALL_VERSION}"
    GGML_COMMIT="${GGML_BUILD_COMMIT}"
)
message(STATUS "ggml version: ${GGML_INSTALL_VERSION}")
message(STATUS "ggml commit:  ${GGML_BUILD_COMMIT}")

#[[
1. 使用 CMake 的 install(FILES ...) 命令为安装步骤添加一条规则：在执行 “cmake --install” 或 “ninja install” 时，将生成的 ggml-config.cmake 和 ggml-version.cmake 这两个文件复制到 ${CMAKE_INSTALL_LIBDIR}/cmake/ggml 目录，供下游项目通过 find_package(ggml) 查找使用。
2. 对 install 理解的不够。
]]
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ggml-config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/ggml-version.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ggml)

#[[
1. 这里的 MSVC 指的是使用 MSVC 编译器工具链，不是“Windows 操作系统”。Windows 上可以不用 MSVC（比如用 MinGW、Clang），这时 MSVC 为假，因此 MSVC/MinGW/Clang 等等都是编译器工具链而不是平台。
2. 对比分析：
  2.1 WIN32：表示操作系统平台是 Windows（包括 32/64 位），无论你用的是 MSVC、MinGW、Clang，只要目标平台是 Windows，WIN32 就为真。
  2.2 MSVC：表示用的是 MSVC 编译器（或者与 MSVC 兼容的，比如 clang-cl）。
3. 平台变量：WIN32、UNIX、APPLE、...... 用于区分在哪个操作系统上？
4. 编译器变量：MSVC、MINGW、CMAKE_CXX_COMPILER_ID、...... 用于区分在哪个编译器？
]]
if (MSVC)
    #[[
    1. 给这些目标加上一堆 MSVC 编译选项 /wdXXXX，目的是 关闭这些 warning。
    2. 4005/4244/4267 这些是警告编号而不是错误类型。
    ]]
    set(MSVC_WARNING_FLAGS
        /wd4005  # Macro redefinition
        /wd4244  # Conversion from one type to another type, possible loss of data
        /wd4267  # Conversion from 'size_t' to a smaller type, possible loss of data
        /wd4305  # Conversion from 'type1' to 'type2', possible loss of data
        /wd4566  # Conversion from 'char' to 'wchar_t', possible loss of data
        /wd4996  # Disable POSIX deprecation warnings
        /wd4702  # Unreachable code warnings
    )
    function(disable_msvc_warnings target_name)
        if(TARGET ${target_name})
            target_compile_options(${target_name} PRIVATE ${MSVC_WARNING_FLAGS})
        endif()
    endfunction()

    disable_msvc_warnings(ggml-base)
    disable_msvc_warnings(ggml)
    disable_msvc_warnings(ggml-cpu)
    disable_msvc_warnings(ggml-cpu-x64)
    disable_msvc_warnings(ggml-cpu-sse42)
    disable_msvc_warnings(ggml-cpu-sandybridge)
    disable_msvc_warnings(ggml-cpu-haswell)
    disable_msvc_warnings(ggml-cpu-skylakex)
    disable_msvc_warnings(ggml-cpu-icelake)
    disable_msvc_warnings(ggml-cpu-alderlake)

    if (GGML_BUILD_EXAMPLES)
        disable_msvc_warnings(common-ggml)
        disable_msvc_warnings(common)

        disable_msvc_warnings(mnist-common)
        disable_msvc_warnings(mnist-eval)
        disable_msvc_warnings(mnist-train)

        disable_msvc_warnings(gpt-2-ctx)
        disable_msvc_warnings(gpt-2-alloc)
        disable_msvc_warnings(gpt-2-backend)
        disable_msvc_warnings(gpt-2-sched)
        disable_msvc_warnings(gpt-2-quantize)
        disable_msvc_warnings(gpt-2-batched)

        disable_msvc_warnings(gpt-j)
        disable_msvc_warnings(gpt-j-quantize)

        disable_msvc_warnings(magika)
        disable_msvc_warnings(yolov3-tiny)
        disable_msvc_warnings(sam)

        disable_msvc_warnings(simple-ctx)
        disable_msvc_warnings(simple-backend)
    endif()

    if (GGML_BUILD_TESTS)
        disable_msvc_warnings(test-mul-mat)
        disable_msvc_warnings(test-arange)
        disable_msvc_warnings(test-backend-ops)
        disable_msvc_warnings(test-cont)
        disable_msvc_warnings(test-conv-transpose)
        disable_msvc_warnings(test-conv-transpose-1d)
        disable_msvc_warnings(test-conv1d)
        disable_msvc_warnings(test-conv2d)
        disable_msvc_warnings(test-conv2d-dw)
        disable_msvc_warnings(test-customop)
        disable_msvc_warnings(test-dup)
        disable_msvc_warnings(test-opt)
        disable_msvc_warnings(test-pool)
    endif ()
endif()
