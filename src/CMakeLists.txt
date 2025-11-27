#[[
1. include 这个 CMake 命令在当前目录/作用域中包含名为 CheckCXXCompilerFlag.cmake 模块文件，然后再执行这个模块文件，即总体分为包含、执行这两个执行步骤。该模块文件中定义的一些变量/宏/函数在当前作用域中是可用的。这种模块化的组织方式是为了功能划分、调试、使用等更加方便。
2. CheckCXXCompilerFlag.cmake 模块通过编译一个 C++ 编译单元从而查看当前编译系统是否支持某些编译参数。
3. 自定义 CMake 模块文件解释：
  3.1 ggml_get_flags(CCID, CCVER)：根据当前编译器类型和版本（传入的 CCID、CCVER，通常是 CMAKE_CXX_COMPILER_ID/CMAKE_CXX_COMPILER_VERSION），生成一组额外的警告/诊断编译选项，分别写回父作用域的 GF_C_FLAGS 和 GF_CXX_FLAGS 供后续追加到编译选项里，用于增强警告覆盖但兼顾版本兼容性。
  3.2 ggml_get_system_arch()：根据 CMake 的架构相关变量（CMAKE_OSX_ARCHITECTURES、CMAKE_GENERATOR_PLATFORM_LWR、CMAKE_SYSTEM_PROCESSOR）判断当前目标 CPU 架构，设置父作用域的 GGML_SYSTEM_ARCH 为 "ARM"/"x86"/PowerPC/loongarch64/riscv64/s390x 或 "UNKNOWN"，供后续按架构做条件配置。
]]
include(CheckCXXCompilerFlag)
include("../cmake/common.cmake")

#[[
1. 给编译系统中的编译器添加宏定义，在 GCC 编译器工具链中相当于通过命令行 -DXXX 的方式给编译器添加宏定义，从而使得编译器可以对代码进行条件编译，在 CMake 中通过该命令添加宏定义的好处就是可以使得开发者更轻松，而不是针对不同平台使用不同的编译命令。
]]
add_compile_definitions(GGML_SCHED_MAX_COPIES=${GGML_SCHED_MAX_COPIES})

# enable libstdc++ assertions for debug builds
#[[
1. 根据不同的操作系统平台设置不同的配置：
  1.1 Linux：只有 Debug 配置下启用 libstdc++ assertions 特性。
  1.2 Others：任何情况下都不启用 libstdc++ assertions 特性。
]]
if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    add_compile_definitions($<$<CONFIG:Debug>:_GLIBCXX_ASSERTIONS>)
endif()

#[[
1. 根据不同的编译器工具链设置不同的配置：
  1.1 MSVC 编译器工具链：不启用 ThreadSanitizer、AddressSanitizer（附带保留栈桢）、UndefinedSanitizer 这些编译器/链接器参数，因为 MSVC 不支持这些 -fsanitize=* 语法。MSVC/clang-cl 要启用类似功能需要不同开关/运行库，支持的不是特别完善（了解不多）。
  1.2 MinGW/XCODE/GCC/Clang 等编译器工具链：根据开启的选项加入编译器/链接器参数以启用 ThreadSanitizer、AddressSanitizer（附带保留栈桢）、UndefinedSanitizer。
2. 作用说明：
  2.1 ThreadSanitizer (-fsanitize=thread): 检测数据竞争、锁顺序等线程并发错误，运行时插桩并报出可疑访问。
  2.2 AddressSanitizer (-fsanitize=address -fno-omit-frame-pointer): 检测越界访问、Use‑After‑Free/Use‑After‑Return、双重释放等内存错误；保留帧指针是为了更好地生成栈回溯。
  2.3 UndefinedSanitizer (-fsanitize=undefined): 检测未定义行为（整数溢出、越界位移、空/野指针算术、未对齐访问、无效的动态类型转换等）。
]]
if (NOT MSVC)
    if (GGML_SANITIZE_THREAD)
        add_compile_options(-fsanitize=thread)
        link_libraries     (-fsanitize=thread)
    endif()

    if (GGML_SANITIZE_ADDRESS)
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        link_libraries     (-fsanitize=address)
    endif()

    if (GGML_SANITIZE_UNDEFINED)
        add_compile_options(-fsanitize=undefined)
        link_libraries     (-fsanitize=undefined)
    endif()
endif()

#[[
1. 如果启用 GGML_FATAL_WARNINGS 即启用 -Werror 编译/链接参数，那么根据不同的编译器工具链配置选项：
  1.1 GNU/Clang 编译器工具链：如果编译器工具链是 GNU 系列或者是 Clang 系列，那么向 C_FLAGS/CXX_FLAGS 追加 -Werror 参数标志。
  1.2 MSVC 编译器工具链：如果编译器工具链是 MSVC ，那么添加 /WX 编译器宏。
2. CMAKE_CXX_COMPILER_ID 值列表：
  2.01 GNU（gcc、MinGW GCC）
  2.02 Clang（包括 Emscripten/LLVM Clang 等）
  2.04 AppleClang（Xcode 自带 Clang）
  2.05 MSVC（包括 clang-cl 以 MSVC 前端模式时通常仍是 MSVC）
  2.06 IntelLLVM（oneAPI/ICC 2021+ 新版）
  2.07 Intel（ICC 旧版经典编译器）
  2.08 NVHPC（NVIDIA HPC/PGI）
  2.09 SunPro（Oracle/Solaris Studio）
  2.10 ARMClang / IAR / QCC 等特定平台也会据实际前端给出对应 ID。
]]
if (GGML_FATAL_WARNINGS)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        list(APPEND C_FLAGS   -Werror)
        list(APPEND CXX_FLAGS -Werror)
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        add_compile_options(/WX)
    endif()
endif()

#[[
1. 如果启用 GGML_ALL_WARNINGS 即启用所有编译器警告，那么根据不同的编译器工具链配置选项：
  1.1 GNU/AppleClang 等编译器工具链：如果编译器工具链不是 GNU/AppleClang 等，那么向 C_FLAGS/CXX_FLAGS/GF_C_FLAGS/GF_CXX_FLAGS 参数标志追加指定参数。
  1.2 MSVC 编译器工具链：如果编译器工具链是 MSVC ，还没有设置对应的编译参数。
2. 这一块是在启用 GGML_ALL_WARNINGS 时追加一组“加严警告”参数。非 MSVC 下直接用 GCC/Clang 习惯的开关（-Wall -Wextra -Wpedantic … 以及 C/C++ 分别的附加项），因为这些参数在 MSVC 不存在对应名称。MSVC 没有 1:1 的等价组合，需要用另一套选项（如 /W4、/permissive- 等）才能达到类似效果，但这里的脚本还没补充 MSVC 侧的实现，所以在 MSVC 分支留空。
]]
if (GGML_ALL_WARNINGS)
    if (NOT MSVC)
        list(APPEND WARNING_FLAGS -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function)
        list(APPEND C_FLAGS       -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes
                                  -Werror=implicit-int -Werror=implicit-function-declaration)
        list(APPEND CXX_FLAGS     -Wmissing-declarations -Wmissing-noreturn)

        list(APPEND C_FLAGS   ${WARNING_FLAGS})
        list(APPEND CXX_FLAGS ${WARNING_FLAGS})

        #[[
        1. 根据当前编译器类型和版本（传入的 CCID、CCVER，通常是 CMAKE_CXX_COMPILER_ID/CMAKE_CXX_COMPILER_VERSION），生成一组额外的警告/诊断编译选项，分别写回父作用域的 GF_C_FLAGS 和 GF_CXX_FLAGS 供后续追加到编译选项里，用于增强警告覆盖但兼顾版本兼容性。
        2. 通过 ggml_get_flags 自定义宏获取得到一组额外的警告/诊断编译选项，写回到当前作用域中的 GF_C_FLAGS 和 GF_CXX_FLAGS 供 add_compile_options 命令添加编译参数。
        ]]
        ggml_get_flags(${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})

        add_compile_options("$<$<COMPILE_LANGUAGE:C>:${C_FLAGS};${GF_C_FLAGS}>"
                            "$<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS};${GF_CXX_FLAGS}>")
    else()
        # todo : msvc
        set(C_FLAGS   "")
        set(CXX_FLAGS "")
    endif()
endif()

#[[
1. 如果启用 GGML_LTO 即启用 link time optimization ，通过使用 CMake 官方提供模块文件中的 check_ipo_supported 函数检测当前系统中是否支持 link time optimization ，如果支持 link time optimization 那么开启 CMake 的优化开关，否则输出提示信息。
]]
if (GGML_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT result OUTPUT output)
    if (result)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(WARNING "IPO is not supported: ${output}")
    endif()
endif()

#[[
1. 这部分内容是用来配置 ccache/sccache 工具的，目前不需要深入理解。
]]
if (GGML_CCACHE AND NOT CMAKE_C_COMPILER_LAUNCHER AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
    find_program(GGML_CCACHE_FOUND ccache)
    find_program(GGML_SCCACHE_FOUND sccache)

    if (GGML_CCACHE_FOUND OR GGML_SCCACHE_FOUND)
        if(GGML_CCACHE_FOUND)
            set(GGML_CCACHE_VARIANT ccache)
        else()
            set(GGML_CCACHE_VARIANT sccache)
        endif()
        # TODO: should not be set globally
        if (GGML_SYCL AND GGML_CCACHE_FOUND AND WIN32)
            set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "ccache compiler_type=icl")
        else ()
            set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${GGML_CCACHE_VARIANT}")
        endif ()
        set(ENV{CCACHE_SLOPPINESS} time_macros)
        message(STATUS "${GGML_CCACHE_VARIANT} found, compilation results will be cached. Disable with GGML_CCACHE=OFF.")
    else()
        message(STATUS "Warning: ccache not found - consider installing it for faster compilation or disable this warning with GGML_CCACHE=OFF")
    endif ()
endif()

# this version of Apple ld64 is buggy
#[[
1. 这段用当前 C 编译器执行 -Wl,-v 获取链接器版本，检测是否是已知有问题的 Apple ld64（后面代码匹配到 dyld-1015.7 时会定义 HAVE_BUGGY_APPLE_LINKER）。目的是在配置阶段识别特定版本的苹果链接器缺陷，便于后续编译条件分支或规避措施（如禁用某些特性或启用修复）。
2. 能够找出这个问题也是🐂。
]]
execute_process(
    COMMAND ${CMAKE_C_COMPILER} ${CMAKE_EXE_LINKER_FLAGS} -Wl,-v
    ERROR_VARIABLE output
    OUTPUT_QUIET
)

if (output MATCHES "dyld-1015\.7")
    add_compile_definitions(HAVE_BUGGY_APPLE_LINKER)
endif()

# architecture specific
# TODO: probably these flags need to be tweaked on some architectures
#       feel free to update the Makefile for your architecture and send a pull request or issue
#[[
1. 注释解释：某些架构可能需要调整这些标志位，请随时更新适用于您架构的 Makefile 文件，并提交 pull request 或 issue 。
2. 打印当前探测到的处理器架构 CMAKE_SYSTEM_PROCESSOR，便于配置日志查看平台信息。如果使用 MSVC，取生成器平台（如 x64/Win32/ARM64 等）转成小写保存到 CMAKE_GENERATOR_PLATFORM_LWR，并打印原始值。非 MSVC 则把 CMAKE_GENERATOR_PLATFORM_LWR 置空。后面用于判定系统架构（ggml_get_system_arch() 会用到）。
3. string(TOLOWER "原字符串" 目标变量) 的效果是把给定字符串转为小写并存入目标变量；此处把生成器平台名统一成小写，方便后续用正则匹配判断平台。
4. CMAKE_SYSTEM_PROCESSOR
  4.1 x86/AMD64 系列：x86_64 / AMD64 / amd64 / i386 / i686 / x86
  4.2 ARM 系列：arm / armv7 / armv7l / aarch64 / arm64
  4.3 其他常见：ppc64 / ppc64le / powerpc / riscv64 / loongarch64 / s390x / mips / mips64 等
5. CMAKE_GENERATOR_PLATFORM
  5.1 Visual Studio 常见：Win32 / x64 / ARM / ARM64
  5.2 某些交叉场景或自定义：可能是 toolset 定义的自定义平台名（如 x86、x86_amd64、ARM64EC 等），也可以为空（单配置生成器如 Ninja/Unix Makefiles 通常留空）。
]]
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if (MSVC)
    string(TOLOWER "${CMAKE_GENERATOR_PLATFORM}" CMAKE_GENERATOR_PLATFORM_LWR)
    message(STATUS "CMAKE_GENERATOR_PLATFORM: ${CMAKE_GENERATOR_PLATFORM}")
else ()
    set(CMAKE_GENERATOR_PLATFORM_LWR "")
endif ()
ggml_get_system_arch()
#[[
1. GGML_SYSTEM_ARCH 变量的值是通过 ggml_get_system_arch 函数赋予的。
]]
message(STATUS "GGML_SYSTEM_ARCH: ${GGML_SYSTEM_ARCH}")

#[[
1. 根据不同的编译器工具链进行不同的配置：
  1.1 非 MSVC 编译器工具链：若开启 GGML_STATIC，为构建静态链接做准备：在 UNIX 非 macOS 上把库查找后缀限定为 .a;.so，再加上 -static，MinGW 时再加 -static-libgcc -static-libstdc++ 以静态链接 GCC 运行时/标准库。若开启 GGML_GPROF，添加 -pg 启用 gprof 性能分析插桩。
  1.2 MSVC 编译器工具链：不做处理。
]]
if (NOT MSVC)
    if (GGML_STATIC)
        if (UNIX AND NOT APPLE)
            set(CMAKE_FIND_LIBRARY_SUFFIXES ".a;.so")
        endif()
        add_link_options(-static)
        if (MINGW)
            add_link_options(-static-libgcc -static-libstdc++)
        endif()
    endif()
    if (GGML_GPROF)
        add_compile_options(-pg)
    endif()
endif()

#[[
1. 如果编译器工具链是 MinGW 那么向编译器添加指定宏定义。
]]
if (MINGW)
    add_compile_definitions(_WIN32_WINNT=${GGML_WIN_VER})
endif()

#
# POSIX conformance
#

# clock_gettime came in POSIX.1b (1993)
# CLOCK_MONOTONIC came in POSIX.1-2001 / SUSv3 as optional
# posix_memalign came in POSIX.1-2001 / SUSv3
# M_PI is an XSI extension since POSIX.1-2001 / SUSv3, came in XPG1 (1985)

# Somehow in OpenBSD whenever POSIX conformance is specified
# some string functions rely on locale_t availability,
# which was introduced in POSIX.1-2008, forcing us to go higher
if (CMAKE_SYSTEM_NAME MATCHES "OpenBSD")
    add_compile_definitions(_XOPEN_SOURCE=700)
elseif (CMAKE_SYSTEM_NAME MATCHES "AIX")
    # Don't define _XOPEN_SOURCE.  We need _ALL_SOURCE, which is the default,
    # in order to define _SC_PHYS_PAGES.
else()
    add_compile_definitions(_XOPEN_SOURCE=600)
endif()

# Data types, macros and functions related to controlling CPU affinity and
# some memory allocation are available on Linux through GNU extensions in libc
if (CMAKE_SYSTEM_NAME MATCHES "Linux" OR CMAKE_SYSTEM_NAME MATCHES "Android")
    add_compile_definitions(_GNU_SOURCE)
endif()

# RLIMIT_MEMLOCK came in BSD, is not specified in POSIX.1,
# and on macOS its availability depends on enabling Darwin extensions
# similarly on DragonFly, enabling BSD extensions is necessary
if (
    CMAKE_SYSTEM_NAME MATCHES "Darwin" OR
    CMAKE_SYSTEM_NAME MATCHES "iOS"    OR
    CMAKE_SYSTEM_NAME MATCHES "tvOS"   OR
    CMAKE_SYSTEM_NAME MATCHES "DragonFly"
)
    add_compile_definitions(_DARWIN_C_SOURCE)
endif()

# alloca is a non-standard interface that is not visible on BSDs when
# POSIX conformance is specified, but not all of them provide a clean way
# to enable it in such cases
if (CMAKE_SYSTEM_NAME MATCHES "FreeBSD")
    add_compile_definitions(__BSD_VISIBLE)
endif()
if (CMAKE_SYSTEM_NAME MATCHES "NetBSD")
    add_compile_definitions(_NETBSD_SOURCE)
endif()
if (CMAKE_SYSTEM_NAME MATCHES "OpenBSD")
    add_compile_definitions(_BSD_SOURCE)
endif()

if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

# ggml

if (GGML_BACKEND_DL AND NOT BUILD_SHARED_LIBS)
    message(FATAL_ERROR "GGML_BACKEND_DL requires BUILD_SHARED_LIBS")
endif()

#[[
1. 这段通过 add_library(ggml-base …) 定义 ggml-base 目标，列出其全部源文件和公开头文件，生成 ggml 的基础库（后续其他目标会在此之上叠加特定后端/功能），即 ggml-base 库只依赖系统库并且作为其它目标的依赖对象。
]]
add_library(ggml-base
            ../include/ggml.h
            ../include/ggml-alloc.h
            ../include/ggml-backend.h
            ../include/ggml-cpp.h
            ../include/ggml-opt.h
            ../include/gguf.h
            ggml.c
            ggml.cpp
            ggml-alloc.c
            ggml-backend.cpp
            ggml-opt.cpp
            ggml-threading.cpp
            ggml-threading.h
            ggml-quants.c
            ggml-quants.h
            gguf.cpp)

#[[
1. 设置 ggml-base 的 VERSION/SOVERSION 属性，目的是给生成的共享库带上完整版本号和主版本号（影响生成文件名和 soname/DLL 元数据），方便动态链接时依赖方按版本解析、区分兼容性。并非直接“让其它目标能链接”，而是让被链接的共享库带有清晰的版本标识。
2. 在 ELF 系统（如 Ubuntu）下，VERSION/SOVERSION 会让 CMake/链接器生成带 soname 的共享库，并创建符号链接：通常真实文件名形如 libggml-base.so.<VERSION>（例 libggml-base.so.0.9.4），SOVERSION 决定 soname 里的主版本并生成 libggml-base.so.<SOVERSION>（你看到的 libggml-base.so.0），再配一个无版本的方便开发时链接的 libggml-base.so。Windows 下的 DLL 不使用 ELF 的 soname 机制，VERSION/SOVERSION 不会改 DLL 文件名，生成的就是 ggml-base.dll 及对应的 import lib ggml-base.lib，版本信息若需要需通过资源文件或自定义命名规则另行处理。
]]
set_target_properties(ggml-base PROPERTIES
    VERSION ${GGML_VERSION}
    SOVERSION ${GGML_VERSION_MAJOR}
)

#[[
1. 通过 target_include_directories 命令向 ggml-base 目标添加需要包含的头文件路径，从而使得编译系统在编译目标的时候能够找到随需要的头文件，在 GNU/Linux 操作系统中则对应 -IPath2Dir 编译器参数。
2. 如果开启 GGML_BACKEND_DL 意味着将 ggml 后端构建成动态库形式，在这种情况下需要向 ggml-base 添加编译器选项（宏定义）
]]
target_include_directories(ggml-base PRIVATE .)
if (GGML_BACKEND_DL)
    target_compile_definitions(ggml-base PUBLIC GGML_BACKEND_DL)
endif()

#[[
1. 这段通过 add_library(ggml …) 定义 ggml 目标，列出其全部源文件，生成 ggml 库，该库依赖 ggml-base 目标。
2. 在 ELF 系统（如 Ubuntu）下，VERSION/SOVERSION 会让 CMake/链接器生成带 soname 的共享库，并创建符号链接：通常真实文件名形如 libggml.so.<VERSION>（例 libggml.so.0.9.4），SOVERSION 决定 soname 里的主版本并生成 libggml.so.<SOVERSION>（你看到的 libggml.so.0），再配一个无版本的方便开发时链接的 libggml.so。Windows 下的 DLL 不使用 ELF 的 soname 机制，VERSION/SOVERSION 不会改 DLL 文件名，生成的就是 ggml.dll 及对应的 import lib ggml.lib，版本信息若需要需通过资源文件或自定义命名规则另行处理。
]]
add_library(ggml
            ggml-backend-reg.cpp)
add_library(ggml::ggml ALIAS ggml)

set_target_properties(ggml PROPERTIES
    VERSION ${GGML_VERSION}
    SOVERSION ${GGML_VERSION_MAJOR}
)

if (GGML_BACKEND_DIR)
    if (NOT GGML_BACKEND_DL)
        message(FATAL_ERROR "GGML_BACKEND_DIR requires GGML_BACKEND_DL")
    endif()
    target_compile_definitions(ggml PUBLIC GGML_BACKEND_DIR="${GGML_BACKEND_DIR}")
endif()

target_link_libraries(ggml PUBLIC ggml-base)

if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_link_libraries(ggml PRIVATE dl)
endif()

#[[
1. ggml_add_backend_library 是封装的后端库构建/安装函数：
  1.1 若开启 GGML_BACKEND_DL：按插件模式构建（add_library(... MODULE ...)），输出目录指向运行目录，定义 GGML_BACKEND_DL 宏，并让 ggml 依赖该插件；安装到指定后端目录或可执行目录。
  1.2 若没开 GGML_BACKEND_DL：将后端直接编译为普通库并链接到主库 ggml，同时安装。
2. ggml_add_backend_library 这个函数没有在当前文件中被使用，而是在每个后端中被使用了，这样做的好处就是避免在每一个后端中都包含一份相同的逻辑代码。
]]
function(ggml_add_backend_library backend)
    #[[
    1. 这个函数的作用是统一规范“后端库（backend）”是怎么被构建、怎么安装、怎么和核心 ggml 关联起来，并顺便把所有已注册的后端名字收集到一个列表里。更细一点说：它根据 GGML_BACKEND_DL 的开关，决定把后端编译成“可动态加载的模块插件”（add_library(MODULE …)，运行时由 ggml 动态加载）还是正常链接到 ggml 里的普通库；
    ]]
    if (GGML_BACKEND_DL)
        add_library(${backend} MODULE ${ARGN})
        # write the shared library to the output directory
        set_target_properties(${backend} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
        target_compile_definitions(${backend} PRIVATE GGML_BACKEND_DL)
        add_dependencies(ggml ${backend})
        if (GGML_BACKEND_DIR)
            install(TARGETS ${backend} LIBRARY DESTINATION ${GGML_BACKEND_DIR})
        else()
            install(TARGETS ${backend} LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR})
        endif()
    else()
        add_library(${backend} ${ARGN})
        target_link_libraries(ggml PUBLIC ${backend})
        install(TARGETS ${backend} LIBRARY)
    endif()

    #[[
    1. 无论哪种方式，都会让该后端链接 ggml-base、包含上层目录头文件、在构建为共享库时定义统一的宏（GGML_BACKEND_BUILD/GGML_BACKEND_SHARED），并为库设置版本号（VERSION/SOVERSION）；最后，还把当前这个 backend 名字添加到内部缓存变量 GGML_AVAILABLE_BACKENDS 中，方便后面打包或生成配置文件时知道“当前构建里有哪些后端”。
    ]]
    target_link_libraries(${backend} PRIVATE ggml-base)
    target_include_directories(${backend} PRIVATE ..)

    if (${BUILD_SHARED_LIBS})
        target_compile_definitions(${backend} PRIVATE GGML_BACKEND_BUILD)
        target_compile_definitions(${backend} PUBLIC  GGML_BACKEND_SHARED)
    endif()

    # Set versioning properties for all backend libraries
    set_target_properties(${backend} PROPERTIES
        VERSION ${GGML_VERSION}
        SOVERSION ${GGML_VERSION_MAJOR}
    )

    if(NOT GGML_AVAILABLE_BACKENDS)
        set(GGML_AVAILABLE_BACKENDS "${backend}"
            CACHE INTERNAL "List of backends for cmake package")
    else()
        list(FIND GGML_AVAILABLE_BACKENDS "${backend}" has_backend)
        if(has_backend EQUAL -1)
            set(GGML_AVAILABLE_BACKENDS "${GGML_AVAILABLE_BACKENDS};${backend}"
                CACHE INTERNAL "List of backends for cmake package")
        endif()
    endif()
endfunction()

#[[
1. ggml_add_backend 是当前文件中定义和实现的一个函数，有一个用来表示后端名称的参数，首先将拼接的字符串使用字符串操作转化成大写的字符串，这么做是为了检查指定的 CMake 变量是否存在且有效。如果指定的 CMake 变量（后端）存在且有效，那么需要跳转到对应的子文件夹中递归执行配置。非动态后端模式下再给主库 ggml 添加 GGML_USE_<后端名> 宏，用于编译期启用该后端。
2. 通过大小写转换生成统一的变量名/目标名，兼容例如 CPU/CUDA/METAL 等不同后端。
]]
function(ggml_add_backend backend)
    string(TOUPPER "GGML_${backend}" backend_id)
    if (${backend_id})
        string(TOLOWER "ggml-${backend}" backend_target)
        add_subdirectory(${backend_target})
        message(STATUS "Including ${backend} backend")
        if (NOT GGML_BACKEND_DL)
            string(TOUPPER "GGML_USE_${backend}" backend_use)
            target_compile_definitions(ggml PUBLIC ${backend_use})
        endif()
    endif()
endfunction()

#[[
1. 为 CPU 后端生成具体的架构/指令集变体：先清零相关特性，再按参数开启（x86 走 GGML_<feat> 标志，其他架构走 GGML_INTERNAL_<feat>）。
2. 将变体标签记录到 GGML_CPU_TAG_NAME，并调用 ggml_add_cpu_backend_variant_impl 生成对应的后端库， ggml_add_cpu_backend_variant_impl 函数定义在 ./ggml-cpu/CMakeLists.txt 文件中，只要使用 add_subdirectory(ggml-cpu) 在前面那么上层作用域中也是可以看到并使用该函数。
3. 用于批量构建 x86/ARM/PowerPC/s390x 不同 ISA 组合的 CPU 插件或静态后端。
]]
function(ggml_add_cpu_backend_variant tag_name)
    set(GGML_CPU_TAG_NAME ${tag_name})
    # other: OPENMP LLAMAFILE CPU_HBM
    if (GGML_SYSTEM_ARCH STREQUAL "x86")
        foreach (feat NATIVE
                      SSE42
                      AVX AVX2 BMI2 AVX_VNNI FMA F16C
                      AVX512 AVX512_VBMI AVX512_VNNI AVX512_BF16
                      AMX_TILE AMX_INT8 AMX_BF16)
            set(GGML_${feat} OFF)
        endforeach()

        foreach (feat ${ARGN})
            set(GGML_${feat} ON)
        endforeach()
    elseif (GGML_SYSTEM_ARCH STREQUAL "ARM")
        foreach (feat ${ARGN})
            set(GGML_INTERNAL_${feat} ON)
        endforeach()
    elseif (GGML_SYSTEM_ARCH STREQUAL "PowerPC")
        foreach (feat ${ARGN})
            set(GGML_INTERNAL_${feat} ON)
        endforeach()
    elseif (GGML_SYSTEM_ARCH STREQUAL "s390x")
        foreach (feat VXE2 NNPA)
            set(GGML_INTERNAL_${feat} OFF)
        endforeach()

        foreach (feat ${ARGN})
            set(GGML_INTERNAL_${feat} ON)
        endforeach()
    endif()

    ggml_add_cpu_backend_variant_impl(${tag_name})
endfunction()

ggml_add_backend(CPU)

if (GGML_CPU_ALL_VARIANTS)
    if (NOT GGML_BACKEND_DL)
        message(FATAL_ERROR "GGML_CPU_ALL_VARIANTS requires GGML_BACKEND_DL")
    elseif (GGML_CPU_ARM_ARCH)
        message(FATAL_ERROR "Cannot use both GGML_CPU_ARM_ARCH and GGML_CPU_ALL_VARIANTS")
    endif()
    if (GGML_SYSTEM_ARCH STREQUAL "x86")
        ggml_add_cpu_backend_variant(x64)
        ggml_add_cpu_backend_variant(sse42        SSE42)
        ggml_add_cpu_backend_variant(sandybridge  SSE42 AVX)
        ggml_add_cpu_backend_variant(haswell      SSE42 AVX F16C AVX2 BMI2 FMA)
        ggml_add_cpu_backend_variant(skylakex     SSE42 AVX F16C AVX2 BMI2 FMA AVX512)
        ggml_add_cpu_backend_variant(icelake      SSE42 AVX F16C AVX2 BMI2 FMA AVX512 AVX512_VBMI AVX512_VNNI)
        ggml_add_cpu_backend_variant(alderlake    SSE42 AVX F16C AVX2 BMI2 FMA AVX_VNNI)
        if (NOT MSVC)
            # MSVC doesn't support AMX
            ggml_add_cpu_backend_variant(sapphirerapids SSE42 AVX F16C AVX2 BMI2 FMA AVX512 AVX512_VBMI AVX512_VNNI AVX512_BF16 AMX_TILE AMX_INT8)
        endif()
    elseif(GGML_SYSTEM_ARCH STREQUAL "ARM")
        if (CMAKE_SYSTEM_NAME MATCHES "Linux")
            # Many of these features are optional so we build versions with popular
            # combinations and name the backends based on the version they were
            # first released with
            ggml_add_cpu_backend_variant(armv8.0_1)
            ggml_add_cpu_backend_variant(armv8.2_1    DOTPROD)
            ggml_add_cpu_backend_variant(armv8.2_2    DOTPROD FP16_VECTOR_ARITHMETIC)
            ggml_add_cpu_backend_variant(armv8.2_3    DOTPROD FP16_VECTOR_ARITHMETIC SVE)
            ggml_add_cpu_backend_variant(armv8.6_1    DOTPROD FP16_VECTOR_ARITHMETIC SVE MATMUL_INT8)
            ggml_add_cpu_backend_variant(armv8.6_2    DOTPROD FP16_VECTOR_ARITHMETIC SVE MATMUL_INT8 SVE2)
            ggml_add_cpu_backend_variant(armv9.2_1    DOTPROD FP16_VECTOR_ARITHMETIC SVE MATMUL_INT8 SME)
            ggml_add_cpu_backend_variant(armv9.2_2    DOTPROD FP16_VECTOR_ARITHMETIC SVE MATMUL_INT8 SVE2 SME)
        elseif (CMAKE_SYSTEM_NAME MATCHES "Android")
            # Android-specific backends with SoC-compatible feature sets
            ggml_add_cpu_backend_variant(android_armv8.0_1)
            ggml_add_cpu_backend_variant(android_armv8.2_1    DOTPROD)
            ggml_add_cpu_backend_variant(android_armv8.2_2    DOTPROD FP16_VECTOR_ARITHMETIC)
            ggml_add_cpu_backend_variant(android_armv8.6_1    DOTPROD FP16_VECTOR_ARITHMETIC MATMUL_INT8)
        elseif (APPLE)
            ggml_add_cpu_backend_variant(apple_m1             DOTPROD)
            ggml_add_cpu_backend_variant(apple_m2_m3          DOTPROD MATMUL_INT8)
            ggml_add_cpu_backend_variant(apple_m4             DOTPROD MATMUL_INT8 NOSVE SME)
        else()
            message(FATAL_ERROR "Unsupported ARM target OS: ${CMAKE_SYSTEM_NAME}")
        endif()
    elseif (GGML_SYSTEM_ARCH STREQUAL "PowerPC")
        if (CMAKE_SYSTEM_NAME MATCHES "Linux")
            ggml_add_cpu_backend_variant(power0)
            ggml_add_cpu_backend_variant(power7_1       POWER7)
            ggml_add_cpu_backend_variant(power7_2       POWER7  VSX)
            ggml_add_cpu_backend_variant(power8_1       POWER8)
            ggml_add_cpu_backend_variant(power8_2       POWER8  VSX)
            ggml_add_cpu_backend_variant(power9         POWER9  VSX)
            ggml_add_cpu_backend_variant(power10        POWER10 VSX)
            ggml_add_cpu_backend_variant(power11        POWER11 VSX)
        else()
            message(FATAL_ERROR "Unsupported PowerPC target OS: ${CMAKE_SYSTEM_NAME}")
        endif()
    elseif (GGML_SYSTEM_ARCH STREQUAL "s390x")
        if (CMAKE_SYSTEM_NAME MATCHES "Linux")
            ggml_add_cpu_backend_variant(z15    Z15 VXE2)
            ggml_add_cpu_backend_variant(z16    Z16 VXE2 NNPA)
        else()
            message(FATAL_ERROR "Unsupported s390x target OS: ${CMAKE_SYSTEM_NAME}")
        endif()
    else()
        message(FATAL_ERROR "GGML_CPU_ALL_VARIANTS not yet supported with ${GGML_SYSTEM_ARCH} on ${CMAKE_SYSTEM_NAME}")
    endif()
elseif (GGML_CPU)
    ggml_add_cpu_backend_variant_impl("")
endif()

ggml_add_backend(BLAS)
ggml_add_backend(CANN)
ggml_add_backend(CUDA)
ggml_add_backend(HIP)
ggml_add_backend(METAL)
ggml_add_backend(MUSA)
ggml_add_backend(RPC)
ggml_add_backend(SYCL)
ggml_add_backend(Vulkan)
ggml_add_backend(WebGPU)
ggml_add_backend(zDNN)
ggml_add_backend(OpenCL)
ggml_add_backend(Hexagon)

foreach (target ggml-base ggml)
    target_include_directories(${target} PUBLIC    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include> $<INSTALL_INTERFACE:include>)
    target_compile_features   (${target} PRIVATE c_std_11 cxx_std_17) # don't bump
endforeach()

target_link_libraries(ggml-base PRIVATE Threads::Threads)

find_library(MATH_LIBRARY m)
if (MATH_LIBRARY)
    if (NOT WIN32 OR NOT DEFINED ENV{ONEAPI_ROOT})
        target_link_libraries(ggml-base PRIVATE m)
    endif()
endif()

if (CMAKE_SYSTEM_NAME MATCHES "Android")
    target_link_libraries(ggml-base PRIVATE dl)
endif()

if(CMAKE_SYSTEM_NAME MATCHES "visionOS")
    target_compile_definitions(ggml-base PUBLIC _DARWIN_C_SOURCE)
endif()

if (BUILD_SHARED_LIBS)
    foreach (target ggml-base ggml)
        set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
        target_compile_definitions(${target} PRIVATE GGML_BUILD)
        target_compile_definitions(${target} PUBLIC  GGML_SHARED)
    endforeach()
endif()
