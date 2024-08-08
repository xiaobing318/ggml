#
# mnist
#[[
1、设置一个名为TEST_TARGET的变量
2、TEST_TARGET变量的值为mnist
3、创建一个名为TEST_TARGET的可执行文件（源文件集合：main.cpp）
4、将TEST_TARGET可执行文件同ggml\common两者进行linking
]]
set(TEST_TARGET mnist)
add_executable(${TEST_TARGET} main.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common)

#
# mnist-cnn

set(TEST_TARGET mnist-cnn)
add_executable(${TEST_TARGET} main-cnn.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common)

#
# mnist-cpu

set(TEST_TARGET mnist-cpu)
add_executable(${TEST_TARGET} main-cpu.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)

if (APPLE)
    #
    # mnist-mtl

    find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
    find_library(METAL_FRAMEWORK    Metal      REQUIRED)
    find_library(METALKIT_FRAMEWORK MetalKit   REQUIRED)
    find_library(METALPERFORMANCE_FRAMEWORK MetalPerformanceShaders REQUIRED)

    set(TEST_TARGET mnist-mtl)
    add_executable(${TEST_TARGET} main-mtl.cpp main-mtl.h main-mtl.m)
    target_link_libraries(${TEST_TARGET} PRIVATE
        ggml
        ${FOUNDATION_LIBRARY}
        ${METAL_FRAMEWORK}
        ${METALKIT_FRAMEWORK}
        ${METALPERFORMANCE_FRAMEWORK}
    )
endif()
