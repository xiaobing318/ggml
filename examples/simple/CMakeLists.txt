#
# simple-ctx

#[[
1、设置一个名为TEST_TARGET的变量
2、TEST_TARGET这个变量的值是simple-ctx
3、创建一个名为TEST_TARGET的可执行文件（源文件集合：simple-ctx.cpp）
4、将可执行文件同ggml进行linking
]]
set(TEST_TARGET simple-ctx)
add_executable(${TEST_TARGET} simple-ctx.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)




#
# simple-backend

#[[
1、设置一个名为TEST_TARGET的变量
2、TEST_TARGET这个变量的值是simple-backend
3、创建一个名为TEST_TARGET的可执行文件（源文件集合：simple-backend.cpp）
4、将可执行文件同ggml进行linking
]]
set(TEST_TARGET simple-backend)
add_executable(${TEST_TARGET} simple-backend.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)

#如果GGML_CUDA是TRUE，那么添加编译定义
if (GGML_CUDA)
    add_compile_definitions(GGML_USE_CUDA)
endif()

if (GGML_METAL)
    add_compile_definitions(GGML_USE_METAL)
endif()
