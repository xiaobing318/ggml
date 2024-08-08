#[[
1、option(GGML_ALL_WARNINGS "ggml: enable all compiler warnings" ON)命令
```cmake
option(<variable> "<description>" [initial_value])
```
- `<variable>`: 这是将被定义或修改的变量的名称。
- `<description>`: 这是对变量的描述，通常在图形界面中显示，帮助用户了解这个变量的用途。
- `[initial_value]`: 这是变量的初始值，通常是 `ON` 或 `OFF`。如果用户在命令行或者通过CMake图形界面未指定这个变量，它就会使用此初始值。

2、解释
  在顶层CMakeLists.txt文件中已经定义了名为GGML_ALL_WARNINGS的bool类型的变量，这个变量用来指示是否在编译时开启所有的编译器警告
]]
if (GGML_ALL_WARNINGS)
  if (NOT MSVC)
      set(cxx_flags
          # TODO(marella): Add other warnings.
          -Wpedantic
          -Wunused-variable
          -Wno-unused-function
          -Wno-multichar
      )
      add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>")
  endif()
endif()

#[[

### 解释CMake命令 `add_library` 和 `target_include_directories`

这两个命令是CMake中用来定义和管理库以及其包含目录的基础构件。下面是对每个命令的具体作用的详细解释：

#### 1. `add_library`

`add_library` 命令用于创建一个库（可以是静态库或动态库），并指定其源文件。命令格式如下：

```cmake
add_library(<name> [STATIC | SHARED | MODULE] [EXCLUDE_FROM_ALL] source1 [source2 ...])
```

- `<name>`: 库的名称，在项目中唯一。
- `STATIC | SHARED | MODULE`: 库的类型。`STATIC` 表示静态库，`SHARED` 表示动态库，`MODULE` 用于创建在运行时加载的模块（如插件）。
- `source1 [source2 ...]`: 构建库所需的源文件。

##### 应用示例

```cmake
add_library(common STATIC common.cpp)
```

  这行命令创建了一个名为 `common` 的静态库，它包含了 `common.cpp` 文件。静态库是在编译时与其他目标文件链接，形成一个单一的可执行文件。
使用静态库的好处包括链接时的性能优化和简化部署，因为所有代码都被包含在一个单一的可执行文件中。

#### 2. `target_include_directories`

这个命令用于为目标（如库或可执行文件）添加包含目录。这些目录会在编译该目标的文件时被编译器用来查找头文件。命令格式如下：

```cmake
target_include_directories(<target> <INTERFACE|PUBLIC|PRIVATE> [items1...])
```

- `<target>`: 目标的名称，需要已经通过 `add_library` 或 `add_executable` 定义。
- `<INTERFACE|PUBLIC|PRIVATE>`: 指定包含目录的范围。
  - `INTERFACE` 表示目录仅用于目标的消费者，即那些链接或使用此目标的其他目标。
  - `PUBLIC` 表示目录既用于此目标自身的构建，也用于链接此目标的其他目标。
  - `PRIVATE` 表示目录仅用于此目标自身的构建，不会影响使用此目标的其他目标。
- `[items1...]`: 一个或多个包含目录。

##### 应用示例

```cmake
target_include_directories(common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
```

  这行命令为名为 `common` 的库设置了包含目录。通过使用 `PUBLIC` 限定符，这意味着 `common` 库本身和任何链接了 `common` 的目标都将能够访问指定
的包含目录。`${CMAKE_CURRENT_SOURCE_DIR}` 变量代表包含当前处理的 `CMakeLists.txt` 文件的目录，确保了无论库文件放在何处，都能正确地找到头文件。

这样的设置是为了确保库的可移植性和易用性，允许开发者和使用者在不同环境下更轻松地构建和使用库。
]]
add_library(common STATIC common.cpp)
target_include_directories(common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})


#[[

在提供的CMake命令中，有三个主要的操作，每个都具有特定的作用和目的，用于库的构建和链接。下面是对每个命令的总结：

### 1. `add_library`

```cmake
add_library(common-ggml STATIC common-ggml.cpp)
```

**作用**：
- **创建库**：这个命令创建了一个名为 `common-ggml` 的静态库。
- **库类型**：指定为 `STATIC`，意味着库是静态链接的。静态库在编译时会被复制到最终的可执行文件中，通常用于创建不希望在运行时进行动态链接的代码。
- **源文件**：指定 `common-ggml.cpp` 为构建此库需要的源文件。

### 2. `target_link_libraries`

```cmake
target_link_libraries(common-ggml PRIVATE ggml)
```

**作用**：
- **目标库指定**：指定 `common-ggml` 为目标库，这是要被链接的库。
- **链接库**：将 `common-ggml` 库与另一个库 `ggml` 链接。
- **链接类型**：使用 `PRIVATE` 关键字，意味着 `ggml` 库只在 `common-ggml` 的编译和链接过程中被使用，对链接 `common-ggml` 的其他目标不可见。
这有助于封装，避免了依赖传递。

### 3. `target_include_directories`

```cmake
target_include_directories(common-ggml PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
```

**作用**：
- **目标指定**：为库 `common-ggml` 设置包含目录。
- **包含目录范围**：使用 `PUBLIC` 说明这些目录既对构建 `common-ggml` 本身有效，也对链接此库的其他目标有效。
- **目录变量**：`${CMAKE_CURRENT_SOURCE_DIR}` 表示当前处理的CMakeLists.txt文件的目录。这使得无论源文件和项目布局如何变化，编译器都能找到正确
的头文件路径。

  通过这些命令，可以有效地控制库的编译、链接及其头文件的访问权限，使得构建过程既灵活又具有良好的封装性。这些设置特别适合于具有复杂依赖和需要精确
控制链接行为的大型项目。
]]
add_library(common-ggml STATIC common-ggml.cpp)
target_link_libraries(common-ggml PRIVATE ggml)
target_include_directories(common-ggml PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

#添加子目录，递归运行子目录中的CMakeLists.txt，从而build项目
add_subdirectory(gpt-2)
add_subdirectory(gpt-j)
add_subdirectory(mnist)
add_subdirectory(sam)
add_subdirectory(yolo)
add_subdirectory(simple)
add_subdirectory(magika)
