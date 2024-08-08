#
# gpt-2

#[[
  这段CMake脚本创建了一个名为 `gpt-2-ctx` 的可执行文件，主要基于 `main-ctx.cpp` 源文件，并确保了需要的库（`ggml`, `common`, `common-ggml`）
被正确链接，提供了必要的功能和代码支持。
1、设置一个名为TEST_TARGET的变量
2、TEST_TARGET这个变量的值为gpt-2-ctx
3、创建一个名为gpt-2-ctx可执行文件，build文件由main-ctx.cpp组成
4、将可执行文件gpt-2-ctx同ggml\common\common-ggml三个库进行linking
]]
set(TEST_TARGET gpt-2-ctx)
add_executable(${TEST_TARGET} main-ctx.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)

set(TEST_TARGET gpt-2-alloc)
add_executable(${TEST_TARGET} main-alloc.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)

set(TEST_TARGET gpt-2-backend)
add_executable(${TEST_TARGET} main-backend.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)

set(TEST_TARGET gpt-2-sched)
add_executable(${TEST_TARGET} main-sched.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)

#
# gpt-2-quantize

set(TEST_TARGET gpt-2-quantize)
add_executable(${TEST_TARGET} quantize.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)

#
# gpt-2-batched

set(TEST_TARGET gpt-2-batched)
add_executable(${TEST_TARGET} main-batched.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)
