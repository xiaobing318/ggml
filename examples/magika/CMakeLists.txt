#
# magika

#[[
1、设置一个名为TEST_TARGET的变量
2、这个名为TEST_TARGET的变量值为magika
3、创建一个名为TEST_TARGET的可执行文件，源文件集合：main.cpp
4、将TEST_TARGET的可执行文件同ggml\common\common-ggml进行linking
]]
set(TEST_TARGET magika)
add_executable(${TEST_TARGET} main.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)

#
# For GPU offloading
#[[
1、如果定义了GGML_CUDA变量，那么执行其中的cmake命令
2、添加编译定义
]]
if (GGML_CUDA)
    add_compile_definitions(GGML_USE_CUDA)
endif()

if (GGML_METAL)
    add_compile_definitions(GGML_USE_METAL)
endif()
