#
# gpt-j

#[[
1、设置一个名为TEST_TARGET的变量
2、这个TEST_TARGET变量名为gpt-j
3、创建名为gpt-j的可执行文件（源文件集合：main.cpp）
4、将gpt-j这个可执行文件同ggml\common\common-ggml链接
]]
set(TEST_TARGET gpt-j)
add_executable(${TEST_TARGET} main.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)

#
# gpt-j-quantize

set(TEST_TARGET gpt-j-quantize)
add_executable(${TEST_TARGET} quantize.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)
