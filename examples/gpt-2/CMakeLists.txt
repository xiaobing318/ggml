#
# gpt-2

set(TEST_TARGET gpt-2-ctx)
add_executable(${TEST_TARGET} main-ctx.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)

set(TEST_TARGET gpt-2-alloc)
add_executable(${TEST_TARGET} main-alloc.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)

#[[
1. 创建一个普通变量用来存储目标名称，当前目标名称是 gpt-2-backend ，构建该目标使用到了名为 main-backend.cpp 源文件，并且该目标私有链接 ggml/common/common-ggml 三个目标。
]]
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
