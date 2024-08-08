#
# sam
#[[
1、设置一个名为TEST_TARGET的变量
2、TEST_TARGET这个变量的值为sam
3、创建名为TEST_TARGET的可执行文件（源文件集合：main.cpp）
4、将TEST_TARGET可执行文件同ggml\common两者进行linking
]]
set(TEST_TARGET sam)
add_executable(${TEST_TARGET} main.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common)

#
# sam-quantize

#set(TEST_TARGET sam-quantize)
#add_executable(${TEST_TARGET} quantize.cpp)
#target_link_libraries(${TEST_TARGET} PRIVATE ggml common)
