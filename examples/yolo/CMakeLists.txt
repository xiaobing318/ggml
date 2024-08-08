#
# yolov3-tiny

#[[
1、设置一个名为TEST_TARGET的变量
2、TEST_TARGET这个变量的值为yolov3-tiny
3、创建一个名为TEST_TARGET的可执行文件（源文件集合：yolov3-tiny.cpp、yolo-image.cpp）
4、将可执行文件同ggml\common两者进行linking
]]
set(TEST_TARGET yolov3-tiny)
add_executable(${TEST_TARGET} yolov3-tiny.cpp yolo-image.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common)
