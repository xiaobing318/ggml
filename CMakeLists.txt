#[[
### 1. cmake_minimum_required(VERSION 3.14)
- **作用**：指定项目所需的最低 CMake 版本为 3.14。
- **解释**：这一行确保 CMake 的版本至少是 3.14，以便使用该版本或更高版本中引入的特定功能。在这个示例中，提到了`add_link_options`和隐式目标目录，
这些功能需要至少 3.14 版本的 CMake。

### 2. project("ggml" C CXX)
- **作用**：定义了一个名为 "ggml" 的项目，并指定项目使用的编程语言是 C 和 C++。
- **解释**：`project()` 命令用于设置项目名称和支持的语言。这一行配置告诉 CMake，这是一个名为 "ggml" 的项目，同时项目中将会使用 C 和 C++ 语言。
这影响 CMake 如何查找编译器以及设置相关的默认编译标志。

### 3. include(CheckIncludeFileCXX)
- **作用**：包含 CMake 模块 `CheckIncludeFileCXX`。
- **解释**：`include()` 命令用于加载并运行 CMake 提供的或用户定义的脚本。`CheckIncludeFileCXX` 是一个 CMake 模块，用于检查 C++ 头文件是否存在。
这对于确定系统是否支持某些特定的头文件非常有用，可以基于此做条件编译。

### 4. set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
- **作用**：设置 `CMAKE_EXPORT_COMPILE_COMMANDS` 变量为 `ON`。
- **解释**：这条命令启用了编译命令的导出功能，将生成一个名为 `compile_commands.json` 的文件，其中包含了项目编译中所有文件的详细编译命令。这对于
开发者使用一些工具（如代码分析工具、编辑器插件等）进行静态分析或者代码导航非常有帮助。

### 总结
    这些 CMake 命令为项目 "ggml" 设置了基本的构建配置，包括项目名称、支持的语言、如何检查系统头文件的存在以及如何导出编译命令，使得开发和调试过程
更加高效。这些设置是构建具有依赖性和特定系统要求的 C/C++ 项目的基础。
]]
cmake_minimum_required(VERSION 3.14) # for add_link_options and implicit target directories.
project("ggml" C CXX)
include(CheckIncludeFileCXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)









#[[
### 1. 第一个 if 条件块
```cmake
if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
```
- **作用与解释**：
  - **if 条件**：检查是否不是在 Xcode 或 MSVC 环境中编译，且没有指定 `CMAKE_BUILD_TYPE`。如果这三个条件都满足，说明没有指定构建类型，
并且不在这两个特定的IDE环境中。
  - **set 命令**：在满足条件的情况下，将 `CMAKE_BUILD_TYPE` 设置为 `Release`，使用 `CACHE` 选项确保这个变量在缓存中可用，而 `FORCE` 选项确保
即使变量已经存在也要覆盖它。
  - **set_property**：设置 `CMAKE_BUILD_TYPE` 可以接受的值为 "Debug", "Release", "MinSizeRel", "RelWithDebInfo"，这样在 CMake GUI 或 
ccmake 中可以选择这些预定义的构建类型。

### 2. 第二个 if 条件块
```cmake
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(GGML_STANDALONE ON)

    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

    # configure project version
    # TODO
else()
    set(GGML_STANDALONE OFF)
endif()
```
- **作用与解释**：
  - **if 条件**：检查项目是否作为顶层项目被配置（即没有被其他项目包含）。`CMAKE_SOURCE_DIR` 是项目的顶层目录，`CMAKE_CURRENT_SOURCE_DIR` 
是当前处理的 CMakeLists.txt 文件的目录。如果它们相等，说明没有其他上级项目。
  - **set(GGML_STANDALONE ON/OFF)**：根据项目是否为独立项目设置变量 `GGML_STANDALONE`。这个变量可用于在后续的配置中根据项目是独立还是作为
子项目被包含来调整设置。
  - **set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)**：当项目是独立项目时，设置执行文件的输出目录为构建目录下的 `bin` 子目录。
  - **# TODO**：这是一个待完成的标记，提示后续还需配置项目版本信息等，具体实现尚未完成。

这些设置为项目提供了灵活的构建配置，使得项目可以适应不同的编译环境和构建需求，同时提供了作为独立项目或子项目被包含时不同的处理逻辑。
]]
if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(GGML_STANDALONE ON)

    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

    # configure project version
    # TODO
else()
    set(GGML_STANDALONE OFF)
endif()




# 杨小兵-2024-08-08：Set MSVC specific flags
if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /utf-8")
endif()





#[[
这段 CMake 脚本是用于根据不同的编译环境设置项目的一些构建选项。下面是对各部分的逐行解释：

### 1. 第一个 if 条件块
```cmake
if (EMSCRIPTEN)
    set(BUILD_SHARED_LIBS_DEFAULT OFF)

    option(GGML_WASM_SINGLE_FILE "ggml: embed WASM inside the generated ggml.js" ON)
else()
```
- **作用与解释**：
  - **if (EMSCRIPTEN)**：检查是否使用 Emscripten 编译环境。Emscripten 是一个可以将 C/C++ 代码编译为 WebAssembly 的工具链，用于在 web 浏览器中
运行 C/C++ 程序。
  - **set(BUILD_SHARED_LIBS_DEFAULT OFF)**：在 Emscripten 环境下，默认不构建共享库（动态链接库）。这通常是因为在 WebAssembly 环境中，动态链接
较为复杂且常常不被支持。
  - **option(GGML_WASM_SINGLE_FILE "ggml: embed WASM inside the generated ggml.js" ON)**：定义一个选项 `GGML_WASM_SINGLE_FILE`，默认值为
 `ON`。此选项用于控制是否将 WebAssembly 代码嵌入到生成的 JavaScript 文件中，即生成单一文件包含所有代码，便于在 Web 环境中部署和加载。

### 2. else 条件块
```cmake
    if (MINGW)
        set(BUILD_SHARED_LIBS_DEFAULT OFF)
    else()
        set(BUILD_SHARED_LIBS_DEFAULT ON)
    endif()
endif()
```
- **作用与解释**：
  - **else()**：对于非 Emscripten 环境的处理。
  - **if (MINGW)**：检查是否在使用 MinGW 编译环境。MinGW 是 "Minimalist GNU for Windows" 的缩写，是一个将 GCC 编译器和 GNU 工具链移植到
 Windows 的项目。
  - **set(BUILD_SHARED_LIBS_DEFAULT OFF)**：在使用 MinGW 的情况下，默认不构建共享库。这可能是由于在 MinGW 中构建共享库时存在特定的限制或问题。
  - **else()**：对于既不是 Emscripten 也不是 MinGW 的其他环境。
  - **set(BUILD_SHARED_LIBS_DEFAULT ON)**：在其他环境下，默认构建共享库。这适用于大多数 Linux 和 macOS 系统，其中共享库的支持通常更加成熟和普遍。

这行 CMake 命令定义了一个用户可配置的选项 `BUILD_SHARED_LIBS`，这是一个常见的 CMake 变量，用于控制是否构建共享库（动态链接库）还是静态库。这个选项允许在配置项目时通过 `-D` 标志来明确指定是否启用共享库的构建。下面是对这行命令的详细解释：


```cmake
option(BUILD_SHARED_LIBS "ggml: build shared libraries" ${BUILD_SHARED_LIBS_DEFAULT})
```

- **option**：CMake 的 `option` 命令用于定义一个布尔值选项，该选项可以在命令行或者通过图形界面（如 ccmake 或 CMake GUI）进行设置。

- **BUILD_SHARED_LIBS**：这是选项的名称。`BUILD_SHARED_LIBS` 是 CMake 中预定义的一个变量，它会被 CMake 和很多使用 CMake 构建的库识别，
用于决定是构建动态库还是静态库。如果设置为 `ON`，项目会构建共享库；如果是 `OFF`，项目会构建静态库。

- **"ggml: build shared libraries"**：这是对选项的描述，通常在图形配置界面中显示，告知用户这个选项的作用。

- **${BUILD_SHARED_LIBS_DEFAULT}**：这是选项的默认值，它是一个变量，其值在前面的 CMake 脚本中根据不同的编译环境（如 Emscripten、MinGW）进行设置。
这样做的好处是可以根据特定环境的最佳实践自动设置推荐的默认值，同时也允许用户在需要时覆盖这些设置。

### 作用和意义

- **灵活性**：这个设置提供了灵活性，允许在项目配置阶段确定库的链接方式，这对于库的用户来说是一个重要的配置点，因为它影响到库的使用方式和依赖管理。
- **环境适应性**：通过使用一个变量 (`BUILD_SHARED_LIBS_DEFAULT`) 来提供默认值，项目可以更好地适应不同的编译环境和用户的需求。
- **易用性**：通过 `option` 命令定义的选项很容易通过 CMake 的命令行接口或者图形用户界面进行配置，提高了项目的易用性和可配置性。

    这种方式特别适合跨平台的项目，因为不同的操作系统和编译环境对动态库和静态库的支持不同，所以提供一个易于配置的选项是实现最佳兼容性的有效方法。这些
条件编译设置确保项目可以在多种编译环境中灵活地调整其构建配置，优化各个平台上的性能和兼容性。
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

option(BUILD_SHARED_LIBS "ggml: build shared libraries" ${BUILD_SHARED_LIBS_DEFAULT})













#
# option list
#

# TODO: mark all options as advanced when not GGML_STANDALONE
#[[
这段 CMake 脚本通过一系列条件设置来配置项目的默认编译选项，特别是针对不同操作系统平台和编译环境的优化。以下是逐行解释：

### TODO Comment
- **# TODO: mark all options as advanced when not GGML_STANDALONE**
  - 这是一个待办事项注释，提示需要在项目不是作为独立项目（`GGML_STANDALONE` 不为 `ON` 时）运行时，将所有配置选项标记为高级（不直接显示给一般用户）。
这通常在 CMake GUI 中操作，以简化配置界面，避免普通用户混淆。

### 第一个 if 条件块（针对苹果操作系统）
```cmake
if (APPLE)
    set(GGML_METAL_DEFAULT ON)
    set(GGML_BLAS_DEFAULT ON)
    set(GGML_BLAS_VENDOR_DEFAULT "Apple")
else()
```
- **if (APPLE)**: 检查是否在苹果操作系统（如 macOS）上编译。这通常包括所有基于 Darwin 的系统。
  - **set(GGML_METAL_DEFAULT ON)**: 在苹果系统上，默认启用 Metal。Metal 是 Apple 开发的图形和计算应用编程接口，优化了性能和效率。
  - **set(GGML_BLAS_DEFAULT ON)**: 同样在苹果系统上，默认启用 BLAS（基础线性代数子程序）支持。
  - **set(GGML_BLAS_VENDOR_DEFAULT "Apple")**: 设置默认的 BLAS 供应商为 "Apple"，意味着使用苹果提供的优化过的数学库。

### else 条件块（针对非苹果操作系统）
```cmake
    set(GGML_METAL_DEFAULT OFF)
    set(GGML_BLAS_DEFAULT OFF)
    set(GGML_BLAS_VENDOR_DEFAULT "Generic")
endif()
```
- **else()**: 对于非苹果操作系统的设置。
  - **set(GGML_METAL_DEFAULT OFF)**: 在非苹果系统上，Metal 不可用，因此默认关闭。
  - **set(GGML_BLAS_DEFAULT OFF)**: 在非苹果系统上，BLAS 支持不默认启用。
  - **set(GGML_BLAS_VENDOR_DEFAULT "Generic")**: 默认 BLAS 供应商设置为 "Generic"，表示使用通用或不特定于供应商的 BLAS 实现。

### 第二个 if 条件块（针对交叉编译情况）
```cmake
if (CMAKE_CROSSCOMPILING)
    set(GGML_NATIVE_DEFAULT OFF)
else()
```
- **if (CMAKE_CROSSCOMPILING)**: 检查是否在进行交叉编译，这是在一个系统上编译另一个系统使用的软件的过程。
  - **set(GGML_NATIVE_DEFAULT OFF)**: 如果是交叉编译，那么本机（native）优化默认关闭，因为交叉编译的目标平台可能与编译平台极其不同。

### else 条件块（针对非交叉编译情况）
```cmake
    set(GGML_NATIVE_DEFAULT ON)
endif()
```
- **else()**: 如果不是交叉编译，则默认启用本机优化。
  - **set(GGML_NATIVE_DEFAULT ON)**: 在正常编译（非交叉编译）时，本机优化默认启用，以获得最佳性能。

### 总结
    这些配置确保了在不同的系统和编译环境下，项目可以根据各自的优势启用或禁用特定的功能，以此获得最佳的性能和兼容性。这样的条件编译设置对于维护
跨平台项目至关重要，确保项目在各种环境下均能正常运行。
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

if (CMAKE_CROSSCOMPILING)
    set(GGML_NATIVE_DEFAULT OFF)
else()
    set(GGML_NATIVE_DEFAULT ON)
endif()
















# general
#[[
    这部分 CMake 脚本定义了几个用户可配置的选项，允许在项目构建时自定义特定的构建行为。每个选项都有一个描述和默认值，方便在项目配置时根据需要进行
调整。以下是对这些选项的逐行解释：

### 1. GGML_STATIC
```cmake
option(GGML_STATIC "ggml: static link libraries" OFF)
```
- **作用**：定义一个名为 `GGML_STATIC` 的配置选项。
- **描述**：此选项用于控制是否将库静态链接到项目中。
- **默认值**：`OFF`，意味着默认情况下不使用静态链接。如果设置为 `ON`，则库将被静态链接到最终的可执行文件或其他库中，这有助于减少依赖但可能增加最终
文件的大小。

### 2. GGML_NATIVE
```cmake
option(GGML_NATIVE "ggml: enable -march=native flag" ${GGML_NATIVE_DEFAULT})
```
- **作用**：定义一个名为 `GGML_NATIVE` 的配置选项。
- **描述**：此选项用于控制是否启用 `-march=native` 编译器标志。
- **默认值**：取决于之前定义的 `GGML_NATIVE_DEFAULT` 变量的值。此编译器标志指示编译器优化生成的代码，以适配正在编译代码的特定硬件（CPU类型），
通常可以提高程序性能。

### 3. GGML_LTO
```cmake
option(GGML_LTO "ggml: enable link time optimization" OFF)
```
- **作用**：定义一个名为 `GGML_LTO` 的配置选项。
- **描述**：此选项用于控制是否启用链接时优化（Link Time Optimization, LTO）。
- **默认值**：`OFF`，表示默认不启用 LTO。LTO 是一种优化技术，可以在链接程序时进行，优化整个程序的性能，而不仅仅是单个编译单元。

### 4. GGML_CCACHE
```cmake
option(GGML_CCACHE "ggml: use ccache if available" ON)
```
- **作用**：定义一个名为 `GGML_CCACHE` 的配置选项。
- **描述**：此选项用于控制是否使用 ccache 来加速再次编译。
- **默认值**：`ON`，意味着如果 ccache 可用，则默认使用它。ccache 是一个编译缓存工具，它可以重用之前的编译结果来加速后续的编译过程，特别是在
频繁修改源代码的开发环境中非常有用。

### 总结
    这些配置选项提供了对项目构建过程中一些关键行为的控制，使得开发者可以根据具体的需求和环境选择最佳的构建策略，如是否优化代码以适应特定硬件，是否
启用高级优化技术，以及是否使用辅助工具来加速编译过程。这样的灵活性对于维护大型或复杂项目至关重要。
]]
option(GGML_STATIC "ggml: static link libraries"         OFF)
option(GGML_NATIVE "ggml: enable -march=native flag"     ${GGML_NATIVE_DEFAULT})
option(GGML_LTO    "ggml: enable link time optimization" OFF)
option(GGML_CCACHE "ggml: use ccache if available"       ON)














# debug
#[[
    这部分 CMake 脚本中的命令定义了与调试和代码质量保障相关的几个可配置选项。这些选项允许开发者在构建项目时启用特定的编译器警告和性能分析工具。
下面是对这些选项的逐行解释：

### 1. GGML_ALL_WARNINGS
```cmake
option(GGML_ALL_WARNINGS "ggml: enable all compiler warnings" ON)
```
- **作用**：定义一个名为 `GGML_ALL_WARNINGS` 的配置选项。
- **描述**：此选项用于控制是否启用编译器的所有警告。
- **默认值**：`ON`，意味着默认情况下会启用所有编译器警告。这有助于在开发过程中识别潜在的代码问题和不规范的编程实践，从而提高代码质量。

### 2. GGML_ALL_WARNINGS_3RD_PARTY
```cmake
option(GGML_ALL_WARNINGS_3RD_PARTY "ggml: enable all compiler warnings in 3rd party libs" OFF)
```
- **作用**：定义一个名为 `GGML_ALL_WARNINGS_3RD_PARTY` 的配置选项。
- **描述**：此选项用于控制是否在第三方库的编译过程中也启用所有编译器警告。
- **默认值**：`OFF`，意味着默认情况下不对第三方库启用所有编译器警告。这通常是因为第三方库可能不完全遵循当前项目的编码标准，或者其警告可能过多且
不相关，开启警告可能会产生大量不必要的干扰信息。

### 3. GGML_GPROF
```cmake
option(GGML_GPROF "ggml: enable gprof" OFF)
```
- **作用**：定义一个名为 `GGML_GPROF` 的配置选项。
- **描述**：此选项用于控制是否启用 gprof 性能分析工具。
- **默认值**：`OFF`，表示默认不启用 gprof。gprof 是一个性能分析程序，可以帮助开发者分析程序的运行时间和函数调用频率等，用于优化程序性能。不默认
启用是因为性能分析可能会影响程序的运行速度，且通常只在特定的开发阶段中需要。

### 总结
    这些配置选项提供了对编译过程中警告和性能分析工具的控制，使得开发者可以根据当前的开发需求灵活选择是否启用这些工具。启用所有警告有助于提早发现和
修复潜在的代码问题，而性能分析工具则有助于识别和优化性能瓶颈。这些选项是项目质量保证和性能调优的重要工具。
]]
option(GGML_ALL_WARNINGS           "ggml: enable all compiler warnings"                   ON)
option(GGML_ALL_WARNINGS_3RD_PARTY "ggml: enable all compiler warnings in 3rd party libs" OFF)
option(GGML_GPROF                  "ggml: enable gprof"                                   OFF)
















# build
#[[

这行 CMake 脚本定义了一个名为 `GGML_FATAL_WARNINGS` 的配置选项，用来控制编译器行为，特别是关于警告的处理方式。下面是对这条命令的详细解释：

```cmake
option(GGML_FATAL_WARNINGS "ggml: enable -Werror flag" OFF)
```

- **option**: 这是 CMake 中用来定义一个可由用户在构建时配置的布尔选项的命令。

- **GGML_FATAL_WARNINGS**: 这是选项的名称，用于在 CMake 配置过程中引用。

- **"ggml: enable -Werror flag"**: 这是选项的描述，它解释了这个选项的用途。描述清楚地指出，此选项与 `-Werror` 编译器标志相关，该标志将所有警告
转换为错误。

- **OFF**: 这是此配置选项的默认值。默认情况下，`GGML_FATAL_WARNINGS` 设置为 `OFF`，意味着 `-Werror` 标志默认不启用。这样，编译器在遇到警告时不
会将其视为错误，编译过程可以继续进行，即使存在警告也不会导致构建失败。

### 作用和意义

- **灵活性**: 这个选项提供了一种灵活的方式来处理编译中的警告。开发者可以选择在调试或发布前的清理阶段将警告视为错误，以确保代码质量。

- **代码质量保证**: 通过将警告视为错误，团队可以强制解决所有潜在的代码问题，从而提高代码的稳定性和质量。

- **可配置性**: 由于这是一个可配置的选项，它允许不同的开发者或团队根据他们的具体需求来启用或禁用此行为，而无需更改代码库中的实际代码。

### 总结

    `GGML_FATAL_WARNINGS` 选项使开发团队能够根据他们对代码质量的严格程度选择是否启用 `-Werror`。这种做法在某些开发环境中是非常有用的，尤其是在
需要保证代码在发布前无任何警告的情况下。此外，这还有助于新开发人员写出更规范的代码，因为他们需要解决编译过程中遇到的所有警告。
]]
option(GGML_FATAL_WARNINGS    "ggml: enable -Werror flag"    OFF)














# sanitizers
#[[

    这三行 CMake 脚本定义了关于使用不同类型的sanitizers（内存和线程错误检测工具）的配置选项。Sanitizers 是用于在运行时检测各种编程错误的工具，
如数据竞争、内存泄漏和未定义行为等。下面是对这些选项的逐行解释：

### 1. GGML_SANITIZE_THREAD
```cmake
option(GGML_SANITIZE_THREAD "ggml: enable thread sanitizer" OFF)
```
- **option**: 这是 CMake 中用来定义一个可由用户在构建时配置的布尔选项的命令。
- **GGML_SANITIZE_THREAD**: 这是选项的名称，用于在 CMake 配置过程中引用。
- **"ggml: enable thread sanitizer"**: 这是选项的描述，它解释了这个选项的用途。描述指出此选项与启用 thread sanitizer（线程分析器）相关。
- **OFF**: 这是此配置选项的默认值。默认情况下，`GGML_SANITIZE_THREAD` 设置为 `OFF`，意味着线程分析器默认不启用。线程分析器用于检测多线程程序
中的数据竞争和其他线程相关的错误。

### 2. GGML_SANITIZE_ADDRESS
```cmake
option(GGML_SANITIZE_ADDRESS "ggml: enable address sanitizer" OFF)
```
- **option**: 同上，定义一个可配置的布尔值选项。
- **GGML_SANITIZE_ADDRESS**: 选项的名称。
- **"ggml: enable address sanitizer"**: 描述此选项用于启用 address sanitizer（地址分析器）。
- **OFF**: 默认值，表示地址分析器默认不启用。地址分析器用于检测内存访问错误，如堆栈溢出、堆栈访问错误和对已释放内存的操作等。

### 3. GGML_SANITIZE_UNDEFINED
```cmake
option(GGML_SANITIZE_UNDEFINED "ggml: enable undefined sanitizer" OFF)
```
- **option**: 同上，定义一个可配置的布尔值选项。
- **GGML_SANITIZE_UNDEFINED**: 选项的名称。
- **"ggml: enable undefined sanitizer"**: 描述此选项用于启用 undefined behavior sanitizer（未定义行为分析器）。
- **OFF**: 默认值，表示未定义行为分析器默认不启用。未定义行为分析器用于检测C++程序中的未定义行为，如整数溢出、类型不匹配等。

### 总结

    这些sanitizer选项提供了强大的工具，用于在开发阶段捕获复杂的编程错误，帮助开发者提前修复可能在生产环境中引发失败的错误。虽然sanitizers在运行时
会增加程序的性能开销，因此通常只在调试或测试阶段启用。这种灵活性允许开发者在需要时开启详细的错误检查，而在性能敏感的生产环境中关闭它们。
]]
option(GGML_SANITIZE_THREAD    "ggml: enable thread sanitizer"    OFF)
option(GGML_SANITIZE_ADDRESS   "ggml: enable address sanitizer"   OFF)
option(GGML_SANITIZE_UNDEFINED "ggml: enable undefined sanitizer" OFF)











# instruction set specific
#[[
这段 CMake 脚本涵盖了多个编译选项，包括指令集支持和操作系统特定配置。下面逐行解释这些 CMake 命令：

### Instruction Set Conditional
```cmake
if (GGML_NATIVE OR NOT GGML_NATIVE_DEFAULT)
    set(INS_ENB OFF)
else()
    set(INS_ENB ON)
endif()
```
- **if 条件**：检查 `GGML_NATIVE` 是否被设置为 `ON` 或者 `GGML_NATIVE_DEFAULT` 是否不是默认值。此条件用于决定是否启用特定的优化。
- **set(INS_ENB OFF)**：如果上述条件为真，则不启用指令集特定优化。
- **set(INS_ENB ON)**：如果上述条件为假，即在默认或非 native 模式下，启用指令集特定优化。

### Memory Options
```cmake
option(GGML_CPU_HBM "ggml: use memkind for CPU HBM" OFF)
```
- **option(GGML_CPU_HBM OFF)**：定义一个配置选项，用于决定是否使用 memkind 库来利用 CPU 的高带宽内存 (HBM)。

### AVX Instruction Sets
```cmake
option(GGML_AVX         "ggml: enable AVX"              ${INS_ENB})
option(GGML_AVX2        "ggml: enable AVX2"             ${INS_ENB})
option(GGML_AVX512      "ggml: enable AVX512"           OFF)
option(GGML_AVX512_VBMI "ggml: enable AVX512-VBMI"      OFF)
option(GGML_AVX512_VNNI "ggml: enable AVX512-VNNI"      OFF)
option(GGML_AVX512_BF16 "ggml: enable AVX512-BF16"      OFF)
```
- **option(GGML_AVX...)**：这些选项用于启用特定的 AVX 指令集，包括 AVX, AVX2, 以及多种 AVX-512 扩展。`INS_ENB` 变量决定了 AVX 和 AVX2 是否
默认启用，而其他 AVX-512 相关的选项默认关闭。

### FMA and F16C Instructions
```cmake
option(GGML_FMA         "ggml: enable FMA"              ${INS_ENB})
if (NOT MSVC)
    option(GGML_F16C    "ggml: enable F16C"             ${INS_ENB})
endif()
```
- **option(GGML_FMA...)**：启用 FMA 指令集，根据 `INS_ENB` 设置。
- **option(GGML_F16C...)**：在非 MSVC 编译器环境下启用 F16C 指令集支持。MSVC 环境下 F16C 支持通常与 AVX2 或 AVX512 指令集一起启用。

### Other Instruction Sets
```cmake
option(GGML_LASX        "ggml: enable lasx"             ON)
option(GGML_LSX         "ggml: enable lsx"              ON)
option(GGML_SVE         "ggml: enable SVE"              OFF)
```
- **option(GGML_LASX and GGML_LSX...)**：启用 LASX 和 LSX 指令集，这些是特定的硬件指令集，可能用于某些特定的处理器架构。
- **option(GGML_SVE...)**：定义一个选项以启用 SVE (Scalable Vector Extension) 指令集，该指令集主要用于 ARM 架构。

### Windows Version
```cmake
if (WIN32)
    set(GGML_WIN_VER "0x602" CACHE STRING "ggml: Windows Version")
endif()
```
- **if (WIN32)**：检查是否在 Windows 平台上编译。
- **set(GGML_WIN_VER...)**：设置 Windows 版本标识符。这里 "0x602" 可能代表特定的 Windows 版本（例如 Windows Vista 或 Windows Server 2008），
此设置可能用于调整对应 Windows 版本的特定编译选项或功能支持。

这些 CMake 配置为项目提供了细致的控制，确保可以针对特定硬件和操作系统环境进行优化，以便利用各种现代处理器的高级功能。
]]
if (GGML_NATIVE OR NOT GGML_NATIVE_DEFAULT)
    set(INS_ENB OFF)
else()
    set(INS_ENB ON)
endif()

option(GGML_CPU_HBM     "ggml: use memkind for CPU HBM" OFF)

option(GGML_AVX         "ggml: enable AVX"              ${INS_ENB})
option(GGML_AVX2        "ggml: enable AVX2"             ${INS_ENB})
option(GGML_AVX512      "ggml: enable AVX512"           OFF)
option(GGML_AVX512_VBMI "ggml: enable AVX512-VBMI"      OFF)
option(GGML_AVX512_VNNI "ggml: enable AVX512-VNNI"      OFF)
option(GGML_AVX512_BF16 "ggml: enable AVX512-BF16"      OFF)
option(GGML_FMA         "ggml: enable FMA"              ${INS_ENB})
if (NOT MSVC)
    option(GGML_F16C    "ggml: enable F16C"             ${INS_ENB}) # in MSVC F16C is implied with AVX2/AVX512
endif()
option(GGML_LASX        "ggml: enable lasx"             ON)
option(GGML_LSX         "ggml: enable lsx"              ON)
option(GGML_SVE         "ggml: enable SVE"              OFF)

if (WIN32)
    set(GGML_WIN_VER "0x602" CACHE STRING "ggml: Windows Version")
endif()











# ggml core
#[[
这行 CMake 脚本是用来设置一个名为 `GGML_SCHED_MAX_COPIES` 的变量。下面是对这条命令的详细解释：

### Command Structure

```cmake
set(GGML_SCHED_MAX_COPIES "4" CACHE STRING "ggml: max input copies for pipeline parallelism")
```

- **set**: 这是 CMake 中用来定义或设置变量的命令。

- **GGML_SCHED_MAX_COPIES**: 这是变量的名称。

- **"4"**: 这是为变量设置的值，这里设置为 "4"。

- **CACHE**: 这个关键字用来指示该变量应被存储在 CMake 的缓存中，而不是仅在当前的 CMake 运行中有效。这意味着变量的值在未来的配置阶段也会被保留，
除非显示地被修改或删除。

- **STRING**: 指定缓存中此变量的类型为 STRING，即字符串。

- **"ggml: max input copies for pipeline parallelism"**: 这是变量的描述，用于解释这个设置的目的。描述说明这个变量控制在实现管道并行性时的
最大输入复制数。

### 作用和意义

- **GGML_SCHED_MAX_COPIES**: 这个变量在 ggml (一个机器学习库) 中可能用于配置管道并行处理的参数。管道并行处理是一种优化技术，通过在不同处理阶段
并行处理不同的数据集来加速计算。在这种设置中，`GGML_SCHED_MAX_COPIES` 定义了可以同时在管道中处理的输入数据副本的最大数量。

- **性能优化**: 通过允许配置最大副本数，开发者可以根据具体的硬件配置和性能需求调整并行级别，优化执行效率和资源利用率。

- **灵活性**: 将这种配置项放入 CMake 缓存中，使得用户在初次配置后，除非需要，通常不必在每次构建时重新指定这些参数。这为用户提供了更大的灵活性和
便利性。

### 总结

    通过这个设置，开发者能够调整并优化他们的应用程序在进行大规模数据处理时的并行策略，适应不同的运行时环境和硬件配置。这是软件配置管理的一个示例，展示
了如何通过构建系统灵活地管理和调优软件性能参数。
]]
set(GGML_SCHED_MAX_COPIES  "4" CACHE STRING "ggml: max input copies for pipeline parallelism")
















# 3rd party libs / backends
#[[
    这段 CMake 脚本包含了多个配置选项，用于控制第三方库和不同后端技术的启用状态。这些选项涵盖了从高性能计算后端到调试和优化特定的硬件加速功能。
下面逐行解释每个命令：

### 启用加速框架
```cmake
option(GGML_ACCELERATE "ggml: enable Accelerate framework" ON)
```
- **option(GGML_ACCELERATE ON)**：定义一个配置选项，用于决定是否启用 Apple 的 Accelerate 框架，它提供了高性能的数学运算库。默认设置为 `ON`。

### 使用 BLAS
```cmake
option(GGML_BLAS "ggml: use BLAS" ${GGML_BLAS_DEFAULT})
set(GGML_BLAS_VENDOR ${GGML_BLAS_VENDOR_DEFAULT} CACHE STRING "ggml: BLAS library vendor")
```
- **option(GGML_BLAS...)**：配置选项，用于控制是否使用 BLAS (基础线性代数子程序) 库。默认值由 `GGML_BLAS_DEFAULT` 提供。
- **set(GGML_BLAS_VENDOR...)**：设置 BLAS 库的供应商，默认值由 `GGML_BLAS_VENDOR_DEFAULT` 提供。

### CUDA 相关选项
```cmake
option(GGML_CUDA "ggml: use CUDA" OFF)
option(GGML_CUDA_FORCE_DMMV "ggml: use dmmv instead of mmvq CUDA kernels" OFF)
option(GGML_CUDA_FORCE_MMQ "ggml: use mmq kernels instead of cuBLAS" OFF)
option(GGML_CUDA_FORCE_CUBLAS "ggml: always use cuBLAS instead of mmq kernels" OFF)
set(GGML_CUDA_DMMV_X "32" CACHE STRING "ggml: x stride for dmmv CUDA kernels")
set(GGML_CUDA_MMV_Y "1" CACHE STRING "ggml: y block size for mmv CUDA kernels")
option(GGML_CUDA_F16 "ggml: use 16 bit floats for some calculations" OFF)
set(GGML_CUDA_KQUANTS_ITER "2" CACHE STRING "ggml: iters./thread per block for Q2_K/Q6_K")
set(GGML_CUDA_PEER_MAX_BATCH_SIZE "128" CACHE STRING "ggml: max. batch size for using peer access")
option(GGML_CUDA_NO_PEER_COPY "ggml: do not use peer to peer copies" OFF)
option(GGML_CUDA_NO_VMM "ggml: do not try to use CUDA VMM" OFF)
option(GGML_CUDA_FA_ALL_QUANTS "ggml: compile all quants for FlashAttention" OFF)
option(GGML_CUDA_USE_GRAPHS "ggml: use CUDA graphs (llama.cpp only)" OFF)
```
- **一系列 CUDA 相关选项**：这些选项控制如何使用 NVIDIA 的 CUDA 技术，包括是否启用 CUDA，使用哪种 CUDA 核心，以及是否启用 CUDA 的高级功能如图形
执行和浮点精度控制。

### 其他硬件加速和技术
```cmake
option(GGML_CURL "ggml: use libcurl to download model from an URL" OFF)
option(GGML_HIPBLAS "ggml: use hipBLAS" OFF)
option(GGML_HIP_UMA "ggml: use HIP unified memory architecture" OFF)
option(GGML_VULKAN "ggml: use Vulkan" OFF)
option(GGML_VULKAN_CHECK_RESULTS "ggml: run Vulkan op checks" OFF)
option(GGML_VULKAN_DEBUG "ggml: enable Vulkan debug output" OFF)
option(GGML_VULKAN_MEMORY_DEBUG "ggml: enable Vulkan memory debug output" OFF)
option(GGML_VULKAN_VALIDATE "ggml: enable Vulkan validation" OFF)
option(GGML_VULKAN_RUN_TESTS "ggml: run Vulkan tests" OFF)
option(GGML_KOMPUTE "ggml: use Kompute" OFF)
option(GGML_METAL "ggml: use Metal" ${GGML_METAL_DEFAULT})
option(GGML_METAL_NDEBUG "ggml: disable Metal debugging" OFF)
option(GGML_METAL_SHADER_DEBUG "ggml: compile Metal with -fno-fast-math" OFF)
option(GGML_METAL_EMBED_LIBRARY "ggml: embed Metal library" ${GGML_METAL})
set(GGML_METAL_MACOSX_VERSION_MIN "" CACHE STRING "ggml: metal minimum macOS version")
set(GGML_METAL_STD "" CACHE STRING "ggml: metal standard version (-std flag)")
option(GGML_OPENMP "ggml: use OpenMP" ON)
option(GGML_RPC "ggml: use RPC" OFF)
option(GGML_SYCL "ggml: use SYCL" OFF)
option(GGML_SYCL_F16 "ggml: use 16 bit floats for sycl calculations" OFF)
set(GGML_SYCL_TARGET "INTEL" CACHE STRING "ggml: sycl target device")
```
- **一系列技术和标准支持选项**：控制项目中使用的各种技术，包括 Vulkan 图形 API、Metal、OpenMP 并行编程框架、RPC 远程过程调用、SYCL 异构编程模型等。这些选项允许用户根据具体的硬件配置和性能需求来启用或禁用特定技术。

这些 CMake 选项展示了项目配置的复杂性和多样性，允许项目针对不同的编译环境、硬件和用户需求进行高度定制。
]]
option(GGML_ACCELERATE                      "ggml: enable Accelerate framework"               ON)
option(GGML_BLAS                            "ggml: use BLAS"                                  ${GGML_BLAS_DEFAULT})
set(GGML_BLAS_VENDOR ${GGML_BLAS_VENDOR_DEFAULT} CACHE STRING
                                            "ggml: BLAS library vendor")
option(GGML_LLAMAFILE                       "ggml: use LLAMAFILE"                             OFF)

option(GGML_CUDA                            "ggml: use CUDA"                                  OFF)
option(GGML_CUDA_FORCE_DMMV                 "ggml: use dmmv instead of mmvq CUDA kernels"     OFF)
option(GGML_CUDA_FORCE_MMQ                  "ggml: use mmq kernels instead of cuBLAS"         OFF)
option(GGML_CUDA_FORCE_CUBLAS               "ggml: always use cuBLAS instead of mmq kernels"  OFF)
set   (GGML_CUDA_DMMV_X   "32" CACHE STRING "ggml: x stride for dmmv CUDA kernels")
set   (GGML_CUDA_MMV_Y     "1" CACHE STRING "ggml: y block size for mmv CUDA kernels")
option(GGML_CUDA_F16                        "ggml: use 16 bit floats for some calculations"   OFF)
set   (GGML_CUDA_KQUANTS_ITER "2" CACHE STRING
                                            "ggml: iters./thread per block for Q2_K/Q6_K")
set   (GGML_CUDA_PEER_MAX_BATCH_SIZE "128" CACHE STRING
                                            "ggml: max. batch size for using peer access")
option(GGML_CUDA_NO_PEER_COPY               "ggml: do not use peer to peer copies"            OFF)
option(GGML_CUDA_NO_VMM                     "ggml: do not try to use CUDA VMM"                OFF)
option(GGML_CUDA_FA_ALL_QUANTS              "ggml: compile all quants for FlashAttention"     OFF)
option(GGML_CUDA_USE_GRAPHS                 "ggml: use CUDA graphs (llama.cpp only)"          OFF)

option(GGML_CURL                            "ggml: use libcurl to download model from an URL" OFF)
option(GGML_HIPBLAS                         "ggml: use hipBLAS"                               OFF)
option(GGML_HIP_UMA                         "ggml: use HIP unified memory architecture"       OFF)
option(GGML_VULKAN                          "ggml: use Vulkan"                                OFF)
option(GGML_VULKAN_CHECK_RESULTS            "ggml: run Vulkan op checks"                      OFF)
option(GGML_VULKAN_DEBUG                    "ggml: enable Vulkan debug output"                OFF)
option(GGML_VULKAN_MEMORY_DEBUG             "ggml: enable Vulkan memory debug output"         OFF)
option(GGML_VULKAN_VALIDATE                 "ggml: enable Vulkan validation"                  OFF)
option(GGML_VULKAN_RUN_TESTS                "ggml: run Vulkan tests"                          OFF)
option(GGML_KOMPUTE                         "ggml: use Kompute"                               OFF)
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
set   (GGML_SYCL_TARGET "INTEL" CACHE STRING
                                            "ggml: sycl target device")












# extra artifacts
#[[
    这两行 CMake 脚本定义了关于构建测试和示例的配置选项。这些选项使得开发者可以根据需要决定是否构建项目的测试用例和示例代码。这样的灵活性有助于
减少编译时间和管理构建输出，特别是在不需要测试或示例时。下面是对这两个命令的逐行解释：

### 1. GGML_BUILD_TESTS
```cmake
option(GGML_BUILD_TESTS "ggml: build tests" ${GGML_STANDALONE})
```
- **option**: 这是 CMake 中用来定义一个可由用户在构建时配置的布尔值选项的命令。
- **GGML_BUILD_TESTS**: 这是选项的名称，用于在 CMake 配置过程中引用。
- **"ggml: build tests"**: 这是选项的描述，它解释了这个选项的用途。描述指出此选项用于决定是否构建项目的测试套件。
- **${GGML_STANDALONE}**: 这是此配置选项的默认值。它使用一个变量 `GGML_STANDALONE` 来决定默认值，该变量的设置通常反映了项目是否作为独立项目
进行构建。如果项目是独立的，通常建议构建测试来验证功能；如果项目是作为一个库嵌入其他项目，可能不需要默认构建测试。

### 2. GGML_BUILD_EXAMPLES
```cmake
option(GGML_BUILD_EXAMPLES "ggml: build examples" ${GGML_STANDALONE})
```
- **option**: 同上，定义一个可配置的布尔值选项。
- **GGML_BUILD_EXAMPLES**: 选项的名称。
- **"ggml: build examples"**: 描述此选项用于启用项目的示例代码构建。
- **${GGML_STANDALONE}**: 默认值也使用 `GGML_STANDALONE` 变量。这反映了当项目独立运行时，默认构建示例代码，以帮助用户了解如何使用库。

### 总结
    这两个选项为项目提供了额外的构建控制，允许开发者或构建管理者根据项目的使用场景灵活地选择是否包括测试和示例。这种配置尤其在大型项目或库开发中
很有用，因为它们可以帮助开发者在开发和维护过程中节省资源，同时确保在需要时能够验证功能和演示用法。这些选项通常在开发阶段被启用，以便进行全面测试
和演示，而在生产或发行构建中被禁用以优化构建时间和输出大小。
]]
option(GGML_BUILD_TESTS    "ggml: build tests"    ${GGML_STANDALONE})
option(GGML_BUILD_EXAMPLES "ggml: build examples" ${GGML_STANDALONE})
















#
# dependencies
#
#[[
    这段 CMake 脚本设置了关于编程语言标准和线程库的配置选项。它确保了项目使用正确的 C 和 C++ 标准，并且正确地链接了线程库。以下是对这些命令的
逐行解释：

### 设置 C 语言标准
```cmake
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)
```
- **set(CMAKE_C_STANDARD 11)**: 这条命令设置了 C 语言标准为 C11。这是近年来 C 语言的一个主要版本，引入了许多新特性，如泛型、匿名结构、改进的
并行支持等。
- **set(CMAKE_C_STANDARD_REQUIRED true)**: 这条命令确保如果指定的标准不可用，CMake 配置过程将失败。这是为了确保构建环境符合代码的需求。

### 根据是否使用 SYCL 设置 C++ 语言标准
```cmake
if (GGML_SYCL)
    set(CMAKE_CXX_STANDARD 17)
else()
    set(CMAKE_CXX_STANDARD 11)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED true)
```
- **if (GGML_SYCL)**: 这个条件检查 `GGML_SYCL` 是否被设置为 `ON`，即是否启用了 SYCL，这是一个用于编写异构计算应用的 C++ 抽象层。
    - **set(CMAKE_CXX_STANDARD 17)**: 如果启用了 SYCL，C++ 标准被设置为 C++17。C++17 提供了许多现代化的特性，这可能是 SYCL 实现所需的。
    - **else**: 如果没有启用 SYCL，C++ 标准则设置为 C++11。
- **set(CMAKE_CXX_STANDARD_REQUIRED true)**: 与 C 语言标准设置类似，这确保了如果编译器不支持指定的 C++ 标准，CMake 配置将失败。

### 设置并查找线程库
```cmake
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
```
- **set(THREADS_PREFER_PTHREAD_FLAG ON)**: 这条命令指示 CMake 在可能的情况下优先使用 pthreads 标志。这是 Unix-like 系统中标准的线程库实现。
- **find_package(Threads REQUIRED)**: 这条命令让 CMake 查找并设置线程库的使用，这对于几乎所有使用线程的现代软件项目都是必需的。`REQUIRED` 
关键字指示这个包是必需的，如果找不到，CMake 配置将失败。

### 总结
    这些 CMake 配置项确保了项目使用适当的编程语言标凈，并且能够链接到必需的线程库。这是维护跨平台兼容性并利用现代编程语言特性的关键步骤，特别是
在涉及到并行和异构计算时。
]]
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)

if (GGML_SYCL)
    set(CMAKE_CXX_STANDARD 17)
else()
    set(CMAKE_CXX_STANDARD 11)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED true)

set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(Threads REQUIRED)













#
# build the library
#
#[[
    添加src子目录
]]
add_subdirectory(src)




















#
# tests and examples
#
#[[
1、如果GGML_BUILD_TESTS变量是TRUE的话，那么启动测试，并且将tests子目录添加进行build
2、如果GGML_BUILD_EXAMPLES变量是TRUE的话，那么将examples子目录添加进行build
]]
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
    这两行 CMake 脚本中的命令负责包含（引入）两个特定的 CMake 模块，这些模块是用来辅助项目的安装和包配置过程。以下是对这两个命令的逐行解释：

### 1. include(GNUInstallDirs)
```cmake
include(GNUInstallDirs)
```
- **作用**: 这条命令引入了 `GNUInstallDirs` 模块，它是 CMake 提供的一个用来定义安装目录的模块。
- **解释**: 使用 `GNUInstallDirs` 可以帮助确保项目的安装目录符合 GNU 安装目录标准，这对于确保在不同的系统上有一致的安装路径非常重要。该模块
为不同类型的安装文件定义了一系列变量（如 `CMAKE_INSTALL_BINDIR`, `CMAKE_INSTALL_LIBDIR`, `CMAKE_INSTALL_INCLUDEDIR` 等），这些变量代表了
二进制文件、库文件、头文件等的标准安装目录。

### 2. include(CMakePackageConfigHelpers)
```cmake
include(CMakePackageConfigHelpers)
```
- **作用**: 这条命令引入了 `CMakePackageConfigHelpers` 模块，它提供了一组函数来创建和安装配置文件，这些配置文件用于帮助外部项目找到并使用当前
项目作为库时的设置。
- **解释**: 使用 `CMakePackageConfigHelpers` 可以帮助生成和安装配置文件，如 `*Config.cmake` 和 `*ConfigVersion.cmake`。这些文件通常包含了
关于如何在外部项目中定位和使用当前项目的信息（如包含目录、库目录、依赖关系等）。该模块中的函数如 `configure_package_config_file` 和 
`write_basic_package_version_file` 等，用于生成适合安装的包配置文件，这对于将当前项目作为库分享或发布非常重要。

### 总结
    通过包含这两个模块，CMake 脚本为项目的安装和打包过程提供了强大的支持，确保了项目可以在其他项目中作为依赖项被方便地找到和正确地使用。这是任何
希望提供可重用库的 CMake 项目的重要组成部分，特别是在开源和大型项目中，这种标准化的安装和配置方法提高了项目的可移植性和易用性。
]]
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# all public headers
# 设置一个变量GGML_PUBLIC_HEADERS用来存储include子目录中相关指定的头文件，相当于是一个文件路径列表
set(GGML_PUBLIC_HEADERS
    include/ggml.h
    include/ggml-alloc.h
    include/ggml-backend.h
    include/ggml-blas.h
    include/ggml-cuda.h
    include/ggml.h
    include/ggml-kompute.h
    include/ggml-metal.h
    include/ggml-rpc.h
    include/ggml-sycl.h
    include/ggml-vulkan.h)

#[[
这些 CMake 脚本指令涉及为名为 `ggml` 的目标（通常是一个库）设置属性，并安排其安装过程。以下是对这些命令的逐行解释：

### 设置目标属性
```cmake
set_target_properties(ggml PROPERTIES PUBLIC_HEADER "${GGML_PUBLIC_HEADERS}")
```
- **set_target_properties**: 这条命令用于设置一个或多个目标的属性。这里，它被用来设置 `ggml` 目标的属性。
- **PROPERTIES PUBLIC_HEADER "${GGML_PUBLIC_HEADERS}"**: 这里指定 `ggml` 目标的 `PUBLIC_HEADER` 属性，即公共头文件列表。这些文件在安装时
会被特别处理，确保它们被放在正确的包含（include）目录中。
- **"${GGML_PUBLIC_HEADERS}"**: 是一个变量，包含了公共头文件的路径列表，这些头文件是要被安装的，以便其他项目在引用 `ggml` 库时可以正确地包含
这些头文件。

### 条件性设置资源属性（被注释掉）
```cmake
#if (GGML_METAL)
#    set_target_properties(ggml PROPERTIES RESOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/ggml-metal.metal")
#endif()
```
- 这些被注释掉的行原本用于设置 `ggml` 目标的 `RESOURCE` 属性，这在使用 Metal 框架时可能需要。如果启用 Metal 支持，则将 Metal 的源文件
（`.metal` 文件）指定为资源文件。这有助于确保在构建过程中正确处理 Metal 文件，例如在 macOS 应用程序包中正确放置它们。

### 安装目标的公共头文件
```cmake
install(TARGETS ggml PUBLIC_HEADER)
```
- **install**: 这条命令用于安装指定的目标。
- **TARGETS ggml**: 指定要安装的目标名称是 `ggml`。
- **PUBLIC_HEADER**: 指示 CMake 安装 `ggml` 目标的公共头文件。这将按照之前通过 `set_target_properties` 设置的 `PUBLIC_HEADER` 属性来执行。

### 条件性安装库文件
```cmake
if (BUILD_SHARED_LIBS)
    install(TARGETS ggml LIBRARY)
endif()
```
- **if (BUILD_SHARED_LIBS)**: 这是一个条件语句，检查 `BUILD_SHARED_LIBS` 变量是否被设置为真。这通常意味着项目配置为构建共享库（动态库）而不是
静态库。
- **install(TARGETS ggml LIBRARY)**: 如果条件为真，则执行安装库文件的命令。`LIBRARY` 指示 CMake 安装 `ggml` 目标作为一个库文件。这将确保在安
装过程中，库文件被放在合适的库目录中（例如 `/usr/lib` 或 `C:/Program Files/lib` 等）。

### 总结
    这些指令确保 `ggml` 目标（库）的头文件和库文件（如果构建为共享库的话）被正确地安装到目标系统的标准目录中。这对于确保库的用户能够方便地找到并
链接到这些文件至关重要，同时也保持了项目的模块化和可维护性。
]]
set_target_properties(ggml PROPERTIES PUBLIC_HEADER "${GGML_PUBLIC_HEADERS}")
#if (GGML_METAL)
#    set_target_properties(ggml PROPERTIES RESOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/ggml-metal.metal")
#endif()
install(TARGETS ggml PUBLIC_HEADER)

if (BUILD_SHARED_LIBS)
    install(TARGETS ggml LIBRARY)
endif()

#[[
这段 CMake 脚本涉及了一系列条件编译和安装步骤，特别是针对使用 Metal 和独立构建的情况。以下是对这些命令的逐行解释：

### 条件编译与安装 Metal 相关文件
```cmake
if (GGML_METAL)
    install(
        FILES src/ggml-metal.metal
        PERMISSIONS
            OWNER_READ
            OWNER_WRITE
            GROUP_READ
            WORLD_READ
        DESTINATION ${CMAKE_INSTALL_BINDIR})

    if (NOT GGML_METAL_EMBED_LIBRARY)
        install(
            FILES ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib
            DESTINATION ${CMAKE_INSTALL_BINDIR}
        )
    endif()
endif()
```
- **if (GGML_METAL)**: 检查是否启用了 Metal 支持。如果是，执行以下安装步骤。
- **install(FILES src/ggml-metal.metal...)**: 安装 `src/ggml-metal.metal` 文件到预定义的二进制安装目录 (`${CMAKE_INSTALL_BINDIR}`)。
该命令还设置了文件权限，保证文件拥有者具有读写权限，组成员和其他用户具有读权限。
- **if (NOT GGML_METAL_EMBED_LIBRARY)**: 如果不将 Metal 库嵌入到应用程序中，则单独安装 `.metallib` 文件。
  - **install(FILES ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib...)**: 将编译生成的 Metal 库文件 (`default.metallib`) 安装到
二进制安装目录。

### 处理和安装 pkg-config 文件
```cmake
if (GGML_STANDALONE)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ggml.pc.in
        ${CMAKE_CURRENT_BINARY_DIR}/ggml.pc
        @ONLY)

    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ggml.pc
        DESTINATION share/pkgconfig)
endif()
```
- **if (GGML_STANDALONE)**: 检查是否在独立模式下构建项目。如果是，执行以下步骤。
- **configure_file(...) @ONLY**: 使用 `configure_file` 命令从模板 (`ggml.pc.in`) 生成配置文件 (`ggml.pc`)。`@ONLY` 参数表示仅替换由 
`@<var>@` 标记的变量，而不处理 `#cmakedefine` 指令。
  - **${CMAKE_CURRENT_SOURCE_DIR}/ggml.pc.in**: 指定输入文件的路径，即模板文件。
  - **${CMAKE_CURRENT_BINARY_DIR}/ggml.pc**: 指定输出文件的路径，即最终生成的配置文件路径。
- **install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ggml.pc...)**: 将生成的 `ggml.pc` 文件安装到系统的 `pkgconfig` 目录下，这样其他项目可以
通过 pkg-config 工具找到并使用 `ggml` 库。

### 总结
    这些命令确保了在启用 Metal 支持时正确地安装 Metal 文件，并在独立构建模式下生成并安装 pkg-config 文件，以便其他项目或用户可以正确地发现
和链接到 `ggml` 库。这提高了库的可用性和易用性，同时也保持了构建过程的灵活性和可配置性。
]]
if (GGML_METAL)
    install(
        FILES src/ggml-metal.metal
        PERMISSIONS
            OWNER_READ
            OWNER_WRITE
            GROUP_READ
            WORLD_READ
        DESTINATION ${CMAKE_INSTALL_BINDIR})

    if (NOT GGML_METAL_EMBED_LIBRARY)
        install(
            FILES ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib
            DESTINATION ${CMAKE_INSTALL_BINDIR}
        )
    endif()
endif()

if (GGML_STANDALONE)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ggml.pc.in
        ${CMAKE_CURRENT_BINARY_DIR}/ggml.pc
        @ONLY)

    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ggml.pc
        DESTINATION share/pkgconfig)
endif()
