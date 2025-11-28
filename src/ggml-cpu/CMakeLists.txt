#[[
1. 在当前 CMakeLists.txt 文件中定义和实现一个名为 ggml_add_cpu_backend_features 的函数，该函数将会作为该文件内部使用的一个函数。
2. 注释翻译：feature detection code 被编译成一个单独的目标，因此可以在不指定架构标志的情况下进行构建。由于同一个构建中可能包含多个 CPU 后端变体，因此无法使用 `set_source_files_properties()` 来设置架构标志。
  2.1 feature detection code:检测当前 CPU 是否支持 AVX/AVX2/AVX512 等等扩展指令，该代码将会被单独编译成一个独立的 CMake 目标。这个探测代码在编译时完全不加任何特殊架构编译选项（architecture flags），例如不加 -mavx2、-mavx512f、/arch:AVX2 等，只用“最保守、最通用”的指令集编译。因为如果在编译的时候添加了某些特殊架构编译选项，可能会导致编译后的代码在某些 CPU 上运行不起来。
  2.2 要完成“探测 CPU 是否支持 AVX2”，这段探测代码本身 绝对不能使用 AVX2 指令。否则你连“问一嘴你支不支持 AVX2”都来不及，程序已经因为非法指令挂了。
]]
function(ggml_add_cpu_backend_features cpu_name arch)
    # The feature detection code is compiled as a separate target so that
    # it can be built without the architecture flags
    # Since multiple variants of the CPU backend may be included in the same
    # build, using set_source_files_properties() to set the arch flags is not possible
    set(GGML_CPU_FEATS_NAME ${cpu_name}-feats)
    add_library(${GGML_CPU_FEATS_NAME} OBJECT ggml-cpu/arch/${arch}/cpu-feats.cpp)
    target_include_directories(${GGML_CPU_FEATS_NAME} PRIVATE . ../include)
    target_compile_definitions(${GGML_CPU_FEATS_NAME} PRIVATE ${ARGN})
    target_compile_definitions(${GGML_CPU_FEATS_NAME} PRIVATE GGML_BACKEND_DL GGML_BACKEND_BUILD GGML_BACKEND_SHARED)
    set_target_properties(${GGML_CPU_FEATS_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(${cpu_name} PRIVATE ${GGML_CPU_FEATS_NAME})
endfunction()

#[[
1. 目标命名
  1.1 CPU 后端：ggml_add_cpu_backend_variant_impl 生成 ggml-cpu 或 ggml-cpu-<tag>（x64、sse42、sandybridge、haswell、skylakex、icelake、alderlake、sapphirerapids；armv8.0_1 等 Linux/Android/Apple 变体；power0/power7_1/.../power11；z15、z16），并为每个变体附带对象库 <name>-feats。
  1.2 其它后端库：ggml-blas、ggml-cann、ggml-cuda、ggml-hip、ggml-metal、ggml-musa、ggml-rpc、ggml-sycl、ggml-vulkan、ggml-webgpu、ggml-zdnn、ggml-opencl、ggml-hexagon。
  1.3 Hexagon 辅助目标：对象库 htp_iface、外部构建的 ggml-htp-v73/v75/v79/v81，以及打包 Metal/Vulkan/WebGPU 的自定义目标 ggml-metal-lib、generate_shaders、vulkan-shaders-gen 等。
2. 源文件归属
  2.1 ggml-base：ggml.c, ggml.cpp, ggml-alloc.c, ggml-backend.cpp, ggml-opt.cpp, ggml-threading.*, ggml-quants.*, gguf.cpp 及公开头文件。
  2.2 ggml：仅 src/ggml-backend-reg.cpp，链接 ggml-base。
  2.3 CPU 后端（src/ggml-cpu/CMakeLists.txt）：固定加入 ggml-cpu/ggml-cpu.c|cpp, repack.*, hbm.*, quants.*, traits.*, amx/*.cpp|h, ggml-cpu-impl.h, common.h, binary-ops.*, unary-ops.*, simd-mappings.h, vec.*, ops.*；可选附加
    2.3.1 LLAMAFILE：llamafile/sgemm.cpp|h；
    2.3.2 ARM：arch/arm/quants.c, arch/arm/repack.cpp;
    2.3.3 x86：arch/x86/quants.c, arch/x86/repack.cpp;
    2.3.4 PowerPC：arch/powerpc/quants.c; loongarch：arch/loongarch/quants.c;
    2.3.5 riscv64：arch/riscv/quants.c, arch/riscv/repack.cpp，若 GGML_CPU_RISCV64_SPACEMIT 再加 spacemit/*;
    2.3.6 s390x：arch/s390/quants.c; wasm：arch/wasm/quants.c;
    2.3.7 KleidiAI：ggml-cpu/kleidiai/*.cpp|h 加上大量 kai/ukernels 源（按 dotprod/i8mm/SME 检测再拼接）。
  2.4 ggml-blas：ggml-blas.cpp。
  2.5 ggml-cann：子目录内所有 *.cpp（file(GLOB)）。
  2.6 ggml-cuda：全部 *.cuh、../../include/ggml-cuda.h、*.cu 与 template-instances/fattn-*, mmq*, mmf*，以及选择性的 fattn-vec* 变体。
  2.7 ggml-hip：复用 ../ggml-cuda/*.cuh|*.cu 与相同模板实例；源语言视是否 hipcc 设为 HIP 或 CXX。
  2.8 ggml-metal：ggml-metal.cpp, ggml-metal-device.{m,cpp}, ggml-metal-common.cpp, ggml-metal-context.m, ggml-metal-ops.cpp，可附加生成的汇编或 default.metallib。
  2.9 ggml-musa：复用 ../ggml-cuda/*.cuh|*.cu + mudnn.cuh，可选 ../ggml-musa/*.cu（MUDNN_COPY）。
  2.10 ggml-rpc：ggml-rpc.cpp。
  2.11 ggml-sycl：ggml-sycl.cpp, ../../include/ggml-sycl.h, 以及本目录所有 *.hpp|*.cpp。
  2.12 ggml-vulkan：ggml-vulkan.cpp, ../../include/ggml-vulkan.h，生成的 ggml-vulkan-shaders.hpp 与每个 .comp 对应的生成 .cpp。
  2.13 ggml-webgpu：ggml-webgpu.cpp, ../../include/ggml-webgpu.h, 生成的 generated/ggml-wgsl-shaders.hpp。
  2.14 ggml-zdnn：本目录所有 *.c|*.cpp|*.h|*.hpp。
  2.15 ggml-opencl：ggml-opencl.cpp, ../../include/ggml-opencl.h，以及按 GGML_OPENCL_KERNELS 列表嵌入或复制的 .cl 头文件。
  2.16 ggml-hexagon：ggml-hexagon.cpp, htp-utils.c|h, ../../include/ggml-hexagon.h；htp 子目录库含 main.c, htp_iface_skel.c, worker-pool.c, 各类 hvx/ops 源等。
3. 配置与自动探测
  3.1 ggml_add_backend_library：根据 GGML_BACKEND_DL 选择 MODULE 插件或普通库；所有后端链接 ggml-base、包含 src 上级头，若构建共享库则定义 GGML_BACKEND_BUILD/GGML_BACKEND_SHARED，并统一设置 VERSION/SOVERSION 与聚合列表 GGML_AVAILABLE_BACKENDS。
  3.2 核心库：ggml-base/ggml 要求 C11/C++17，ggml-base 私有包含 src，动态后端模式下公开 GGML_BACKEND_DL，按平台添加 _GNU_SOURCE/_DARWIN_C_SOURCE/_XOPEN_SOURCE 等；ggml 可带 GGML_BACKEND_DIR 宏，Linux 链接 dl。
  3.3 CPU 后端（src/ggml-cpu/CMakeLists.txt）：统一要求 C11/C++17、包含 . 与 ggml-cpu；可开启 Accelerate(GGML_USE_ACCELERATE 等)与 OpenMP(GGML_USE_OPENMP + 链接 OpenMP)；GGML_LLAMAFILE/CPU_HBM/CPU_REPACK/CPU_KLEIDIAI 分别加对应宏和依赖（memkind、KleidiAI 源及特定 include/编译选项）；Emscripten 强制 -msimd128，IntelLLVM 关闭 -fassociative-math。
  3.4 ARM：拒绝 MSVC 非 Clang；可探测 -mfp16-format=ieee；GGML_NATIVE 通过编译器输出来拼 -mcpu/-march 并逐项探测 dotprod/i8mm/sve/sme；非 native 时用 GGML_CPU_ARM_ARCH 或 GGML_CPU_ALL_VARIANTS 自动拼 -march=armv8.x + +dotprod/+fp16/+sve/+i8mm/+sve2/+nosve/+sme，并对变体定义 GGML_USE_*；可生成 <name>-feats 对象库并链接回主库。
  3.5 x86：MSVC 分支用 /arch:* 和显式 GGML_AVX512/AVX2/AVX/SSE42 等宏，AMX/AVX512 扩展手动定义，BMI2/AVX-VNNI 手动补宏；GGML_NATIVE 时可调用 FindSIMD；GCC/Clang 分支用 -march=native 或逐项 -msse4.2/-mf16c/-mfma/-mbmi2/-mavx/-mavx2/-mavxvnni/-mavx512*/-mamx-*，对应宏 GGML_*；若 GGML_BACKEND_DL 且非 GGML_NATIVE，额外生成 <name>-feats 对象库。
  3.6 PowerPC：可选 GGML_NATIVE 基于 /proc/cpuinfo 或 GGML_CPU_ALL_VARIANTS（POWER7-11/VSX 定义）或 GGML_CPU_POWERPC_CPUTYPE；变体模式生成 features 对象库。
  3.7 loongarch64：固定 -march=loongarch64，可加 -mlasx/-mlsx。
  3.8 riscv64：拼 -march=rv64gc[_zfh][_xtheadvector|_v[_zvfh]/][_zicbop] -mabi=lp64d；Spacemit 打开 GGML_USE_CPU_RISCV64_SPACEMIT 并加入 ime* 源。
  3.9 s390x：依据 /proc/cpuinfo 选 -march=z15/z16/z17 或 -march=native，跨平台变体范围 z15-z17；可选 VXE/VXE2/NNPA 宏并生成 features 对象库。
  3.10 wasm：仅加 wasm quants；未知架构则定义 -DGGML_CPU_GENERIC。最终把 ARCH_FLAGS 传给 target_compile_options，ARCH_DEFINITIONS 传给 target_compile_definitions。
  3.11 BLAS：探测 BLAS，依供应商添加 GGML_BLAS_USE_ACCELERATE/MKL/BLIS 等宏，应用 BLAS_LINKER_FLAGS、包含/库目录。
  3.12 CANN：校验平台与 SOC，定义 ASCEND_*（由 SOC_TYPE），可选 USE_ACL_GRAPH；链接 ascendcl/nnopbase/opapi/acl_op_compiler。
  3.13 CUDA：设置 CMAKE_CUDA_ARCHITECTURES（native 或多级虚拟/真实 SM），宏 GGML_CUDA_PEER_MAX_BATCH_SIZE，可选 GGML_CUDA_USE_GRAPHS/ FORCE_MMQ/ FORCE_CUBLAS/ NO_VMM/ NO_FA/ NO_PEER_COPY，GGML_CUDA_DEBUG 加 -lineinfo，GGML_CUDA_COMPRESSION_MODE 传给 -compress-mode，链接 cudart/cublas(+Lt)/cuda_driver。
  3.14 HIP：宏 GGML_USE_HIP，并（非动态后端时）对 ggml 补 GGML_USE_CUDA；传递 CUDA 相关宏/开关（FORCE_MMQ/CUBLAS→HIP等）、GGML_HIP_GRAPHS、GGML_HIP_NO_VMM、GGML_HIP_ROCWMMA_FATTN、GGML_HIP_NO_MMQ_MFMA、GGML_HIP_EXPORT_METRICS、GGML_CUDA_NO_FA；链接 hip::host/rocblas/hipblas。
  3.15 Metal：可选 GGML_METAL_NDEBUG、GGML_METAL_EMBED_LIBRARY 生成内嵌汇编或默认 default.metallib，支持 GGML_METAL_SHADER_DEBUG/GGML_METAL_MACOSX_VERSION_MIN/GGML_METAL_STD；链接 Foundation/Metal/MetalKit。
  3.16 MUSA：宏 GGML_USE_MUSA、GGML_CUDA_PEER_MAX_BATCH_SIZE，可选 GGML_MUSA_GRAPHS、GGML_CUDA_FORCE_MMQ/CUBLAS/NO_VMM/NO_FA/NO_PEER_COPY、GGML_MUSA_MUDNN_COPY；每源附 --cuda-gpu-arch=mp_<arch> 等编译标志；链接 musart/mublas/(mudnn)。
  3.17 RPC：仅 Windows 额外链接 ws2_32。
  3.18 SYCL：编译/链接 -fsycl，宏 GGML_SYCL_DNNL、GGML_SYCL_F16、GGML_SYCL_GRAPH、GGML_SYCL_WARP_SIZE（16/32 取决于目标）、GGML_SYCL_NVIDIA/AMD/GENERIC；可用 oneDNN 或自动 Fetch oneMath（配置 GPU backend/arch），NVIDIA/AMD 还加 -fsycl-targets 与 -Xsycl-target-backend --offload-arch=…。
  3.19 Vulkan：根据 glslc 测试自动定义 GGML_VULKAN_COOPMAT_GLSLC_SUPPORT 等扩展宏，支持 GGML_VULKAN_CHECK_RESULTS/DEBUG/MEMORY_DEBUG/SHADER_DEBUG_INFO/VALIDATE/RUN_TESTS；外部项目生成并安装着色器代码。
  3.20 WebGPU：可选 GGML_WEBGPU_DEBUG/CPU_PROFILE/GPU_PROFILE，依平台链接 Dawn 或 emdawn 端口；Python 嵌入 WGSL。
  3.21 zDNN：定义 GGML_USE_ZDNN 并链接探测到的 zdnn 库/包含。
  3.22 OpenCL：宏 GGML_OPENCL_SOA_Q、GGML_OPENCL_TARGET_VERSION，可选 GGML_OPENCL_PROFILING/USE_ADRENO_KERNELS/EMBED_KERNELS，Python 生成内嵌 kernel 头。
  3.23 Hexagon：htp_iface 按 GGML_HEXAGON_HTP_DEBUG 设 HTP_DEBUG/NDEBUG，链接 cdsprpc/rpcmem/-ldl 或 -llog 等；外部项目为 v73/v75/v79/v81 构建 DSP skel，ggml-hexagon 链接 htp_iface。
]]
function(ggml_add_cpu_backend_variant_impl tag_name)
    if (tag_name)
        set(GGML_CPU_NAME ggml-cpu-${tag_name})
    else()
        set(GGML_CPU_NAME ggml-cpu)
    endif()

    ggml_add_backend_library(${GGML_CPU_NAME})

    list (APPEND GGML_CPU_SOURCES
        ggml-cpu/ggml-cpu.c
        ggml-cpu/ggml-cpu.cpp
        ggml-cpu/repack.cpp
        ggml-cpu/repack.h
        ggml-cpu/hbm.cpp
        ggml-cpu/hbm.h
        ggml-cpu/quants.c
        ggml-cpu/quants.h
        ggml-cpu/traits.cpp
        ggml-cpu/traits.h
        ggml-cpu/amx/amx.cpp
        ggml-cpu/amx/amx.h
        ggml-cpu/amx/mmq.cpp
        ggml-cpu/amx/mmq.h
        ggml-cpu/ggml-cpu-impl.h
        ggml-cpu/common.h
        ggml-cpu/binary-ops.h
        ggml-cpu/binary-ops.cpp
        ggml-cpu/unary-ops.h
        ggml-cpu/unary-ops.cpp
        ggml-cpu/simd-mappings.h
        ggml-cpu/vec.h
        ggml-cpu/vec.cpp
        ggml-cpu/ops.h
        ggml-cpu/ops.cpp
        )

    target_compile_features(${GGML_CPU_NAME} PRIVATE c_std_11 cxx_std_17)
    target_include_directories(${GGML_CPU_NAME} PRIVATE . ggml-cpu)

    if (APPLE AND GGML_ACCELERATE)
        find_library(ACCELERATE_FRAMEWORK Accelerate)
        if (ACCELERATE_FRAMEWORK)
            message(STATUS "Accelerate framework found")

            target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_ACCELERATE)
            target_compile_definitions(${GGML_CPU_NAME} PRIVATE ACCELERATE_NEW_LAPACK)
            target_compile_definitions(${GGML_CPU_NAME} PRIVATE ACCELERATE_LAPACK_ILP64)

            target_link_libraries(${GGML_CPU_NAME} PRIVATE ${ACCELERATE_FRAMEWORK})
        else()
            message(WARNING "Accelerate framework not found")
        endif()
    endif()

    if (GGML_OPENMP)
        find_package(OpenMP)
        if (OpenMP_FOUND)
            set(GGML_OPENMP_ENABLED "ON" CACHE INTERNAL "")
            target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_OPENMP)

            target_link_libraries(${GGML_CPU_NAME} PRIVATE OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
        else()
            set(GGML_OPENMP_ENABLED "OFF" CACHE INTERNAL "")
            message(WARNING "OpenMP not found")
        endif()
    endif()

    if (GGML_LLAMAFILE)
        target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_LLAMAFILE)

        list(APPEND GGML_CPU_SOURCES
                    ggml-cpu/llamafile/sgemm.cpp
                    ggml-cpu/llamafile/sgemm.h)
    endif()

    if (GGML_CPU_HBM)
        find_library(memkind memkind REQUIRED)

        message(STATUS "Using memkind for CPU HBM")

        target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_CPU_HBM)

        target_link_libraries(${GGML_CPU_NAME} PUBLIC memkind)
    endif()

    if (GGML_SYSTEM_ARCH STREQUAL "ARM")
        message(STATUS "ARM detected")
        list(APPEND GGML_CPU_SOURCES
            ggml-cpu/arch/arm/quants.c
            ggml-cpu/arch/arm/repack.cpp
            )

        if (MSVC AND NOT CMAKE_C_COMPILER_ID STREQUAL "Clang")
            message(FATAL_ERROR "MSVC is not supported for ARM, use clang")
        else()
            check_cxx_compiler_flag(-mfp16-format=ieee GGML_COMPILER_SUPPORTS_FP16_FORMAT_I3E)
            if (NOT "${GGML_COMPILER_SUPPORTS_FP16_FORMAT_I3E}" STREQUAL "")
                list(APPEND ARCH_FLAGS -mfp16-format=ieee)
            endif()

            if (GGML_NATIVE)
                # -mcpu=native does not always enable all the features in some compilers,
                # so we check for them manually and enable them if available

                execute_process(
                    COMMAND ${CMAKE_C_COMPILER} -mcpu=native -E -v -
                    INPUT_FILE "/dev/null"
                    OUTPUT_QUIET
                    ERROR_VARIABLE ARM_MCPU
                    RESULT_VARIABLE ARM_MCPU_RESULT
                )
                if (NOT ARM_MCPU_RESULT)
                    string(REGEX MATCH "-mcpu=[^ ']+" ARM_MCPU_FLAG "${ARM_MCPU}")
                    string(REGEX MATCH "-march=[^ ']+" ARM_MARCH_FLAG "${ARM_MCPU}")

                    # on some old GCC we need to read -march=
                    if (ARM_MARCH_FLAG AND NOT "${ARM_MARCH_FLAG}" STREQUAL "-march=native")
                        set(ARM_NATIVE_FLAG "${ARM_MARCH_FLAG}")
                    elseif(ARM_MCPU_FLAG AND NOT "${ARM_MCPU_FLAG}" STREQUAL "-mcpu=native")
                        set(ARM_NATIVE_FLAG "${ARM_MCPU_FLAG}")
                    endif()
                endif()

                if ("${ARM_NATIVE_FLAG}" STREQUAL "")
                    set(ARM_NATIVE_FLAG -mcpu=native)
                    message(WARNING "ARM -march/-mcpu not found, -mcpu=native will be used")
                else()
                    message(STATUS "ARM detected flags: ${ARM_NATIVE_FLAG}")
                endif()

                include(CheckCXXSourceRuns)

                function(check_arm_feature tag code)
                    set(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
                    set(CMAKE_REQUIRED_FLAGS "${ARM_NATIVE_FLAG}+${tag}")
                    check_cxx_source_runs("${code}" GGML_MACHINE_SUPPORTS_${tag})
                    if (GGML_MACHINE_SUPPORTS_${tag})
                        set(ARM_NATIVE_FLAG_FIX "${ARM_NATIVE_FLAG_FIX}+${tag}" PARENT_SCOPE)
                    else()
                        set(CMAKE_REQUIRED_FLAGS "${ARM_NATIVE_FLAG}+no${tag}")
                        check_cxx_source_compiles("int main() { return 0; }" GGML_MACHINE_SUPPORTS_no${tag})
                        if (GGML_MACHINE_SUPPORTS_no${tag})
                            set(ARM_NATIVE_FLAG_FIX "${ARM_NATIVE_FLAG_FIX}+no${tag}" PARENT_SCOPE)
                        endif()
                    endif()
                    set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})
                endfunction()

                check_arm_feature(dotprod "#include <arm_neon.h>\nint main() { int8x16_t _a, _b; volatile int32x4_t _s = vdotq_s32(_s, _a, _b); return 0; }")
                check_arm_feature(i8mm    "#include <arm_neon.h>\nint main() { int8x16_t _a, _b; volatile int32x4_t _s = vmmlaq_s32(_s, _a, _b); return 0; }")
                check_arm_feature(sve     "#include <arm_sve.h>\nint main()  { svfloat32_t _a, _b; volatile svfloat32_t _c = svadd_f32_z(svptrue_b8(), _a, _b); return 0; }")
                check_arm_feature(sme     "#include <arm_sme.h>\n__arm_locally_streaming int main() { __asm__ volatile(\"smstart; smstop;\"); return 0; }")

                list(APPEND ARCH_FLAGS "${ARM_NATIVE_FLAG}${ARM_NATIVE_FLAG_FIX}")
            else()
                if (GGML_CPU_ARM_ARCH)
                    list(APPEND ARCH_FLAGS -march=${GGML_CPU_ARM_ARCH})
                elseif(GGML_CPU_ALL_VARIANTS)
                    # Begin with the lowest baseline
                    set(ARM_MCPU "armv8-a")
                    set(ARCH_TAGS "")
                    set(ARCH_DEFINITIONS "")

                    # When a feature is selected, bump the MCPU to the first
                    # version that supported it
                    if (GGML_INTERNAL_DOTPROD)
                        set(ARM_MCPU "armv8.2-a")
                        set(ARCH_TAGS "${ARCH_TAGS}+dotprod")
                        list(APPEND ARCH_DEFINITIONS GGML_USE_DOTPROD)
                    endif()
                    if (GGML_INTERNAL_FP16_VECTOR_ARITHMETIC)
                        set(ARM_MCPU "armv8.2-a")
                        set(ARCH_TAGS "${ARCH_TAGS}+fp16")
                        list(APPEND ARCH_DEFINITIONS GGML_USE_FP16_VECTOR_ARITHMETIC)
                    endif()
                    if (GGML_INTERNAL_SVE)
                        set(ARM_MCPU "armv8.2-a")
                        set(ARCH_TAGS "${ARCH_TAGS}+sve")
                        list(APPEND ARCH_DEFINITIONS GGML_USE_SVE)
                    endif()
                    if (GGML_INTERNAL_MATMUL_INT8)
                        set(ARM_MCPU "armv8.6-a")
                        set(ARCH_TAGS "${ARCH_TAGS}+i8mm")
                        list(APPEND ARCH_DEFINITIONS GGML_USE_MATMUL_INT8)
                    endif()
                    if (GGML_INTERNAL_SVE2)
                        set(ARM_MCPU "armv8.6-a")
                        set(ARCH_TAGS "${ARCH_TAGS}+sve2")
                        list(APPEND ARCH_DEFINITIONS GGML_USE_SVE2)
                    endif()
                    if (GGML_INTERNAL_NOSVE)
                        set(ARCH_TAGS "${ARCH_TAGS}+nosve")
                    endif()
                    if (GGML_INTERNAL_SME)
                        set(ARM_MCPU "armv9.2-a")
                        set(ARCH_TAGS "${ARCH_TAGS}+sme")
                        list(APPEND ARCH_DEFINITIONS GGML_USE_SME)
                    endif()
                    list(APPEND ARCH_FLAGS "-march=${ARM_MCPU}${ARCH_TAGS}")
                    ggml_add_cpu_backend_features(${GGML_CPU_NAME} arm ${ARCH_DEFINITIONS})
                endif()
            endif()

            # show enabled features
            if (CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
                set(FEAT_INPUT_FILE "NUL")
            else()
                set(FEAT_INPUT_FILE "/dev/null")
            endif()

            execute_process(
                COMMAND ${CMAKE_C_COMPILER} ${ARCH_FLAGS} -dM -E -
                INPUT_FILE ${FEAT_INPUT_FILE}
                OUTPUT_VARIABLE ARM_FEATURE
                RESULT_VARIABLE ARM_FEATURE_RESULT
            )
            if (ARM_FEATURE_RESULT)
                message(WARNING "Failed to get ARM features")
            else()
                foreach(feature DOTPROD SVE MATMUL_INT8 FMA FP16_VECTOR_ARITHMETIC SME)
                    string(FIND "${ARM_FEATURE}" "__ARM_FEATURE_${feature} 1" feature_pos)
                    if (NOT ${feature_pos} EQUAL -1)
                        # Special handling for MATMUL_INT8 when machine doesn't support i8mm
                        if ("${feature}" STREQUAL "MATMUL_INT8" AND GGML_MACHINE_SUPPORTS_noi8mm)
                            message(STATUS "ARM feature ${feature} detected but unsetting due to machine not supporting i8mm")
                            list(APPEND ARCH_FLAGS -U__ARM_FEATURE_MATMUL_INT8)
                        else()
                            message(STATUS "ARM feature ${feature} enabled")
                        endif()
                    endif()
                endforeach()
            endif()
        endif()
    elseif (GGML_SYSTEM_ARCH STREQUAL "x86")
        message(STATUS "x86 detected")
        list(APPEND GGML_CPU_SOURCES
            ggml-cpu/arch/x86/quants.c
            ggml-cpu/arch/x86/repack.cpp
            )

        if (MSVC)
            # instruction set detection for MSVC only
            if (GGML_NATIVE)
                include(ggml-cpu/cmake/FindSIMD.cmake)
            endif ()
            if (GGML_AVX512)
                list(APPEND ARCH_FLAGS /arch:AVX512)
                # /arch:AVX512 includes: __AVX512F__, __AVX512CD__, __AVX512BW__, __AVX512DQ__, and __AVX512VL__
                # MSVC has no compile-time flags enabling specific
                # AVX512 extensions, neither it defines the
                # macros corresponding to the extensions.
                # Do it manually.
                list(APPEND ARCH_DEFINITIONS GGML_AVX512)
                if (GGML_AVX512_VBMI)
                    list(APPEND ARCH_DEFINITIONS __AVX512VBMI__)
                    if (CMAKE_C_COMPILER_ID STREQUAL "Clang")
                        list(APPEND ARCH_FLAGS -mavx512vbmi)
                    endif()
                endif()
                if (GGML_AVX512_VNNI)
                    list(APPEND ARCH_DEFINITIONS __AVX512VNNI__ GGML_AVX512_VNNI)
                    if (CMAKE_C_COMPILER_ID STREQUAL "Clang")
                        list(APPEND ARCH_FLAGS -mavx512vnni)
                    endif()
                endif()
                if (GGML_AVX512_BF16)
                    list(APPEND ARCH_DEFINITIONS __AVX512BF16__ GGML_AVX512_BF16)
                    if (CMAKE_C_COMPILER_ID STREQUAL "Clang")
                        list(APPEND ARCH_FLAGS -mavx512bf16)
                    endif()
                endif()
                if (GGML_AMX_TILE)
                    list(APPEND ARCH_DEFINITIONS __AMX_TILE__ GGML_AMX_TILE)
                endif()
                if (GGML_AMX_INT8)
                    list(APPEND ARCH_DEFINITIONS __AMX_INT8__ GGML_AMX_INT8)
                endif()
                if (GGML_AMX_BF16)
                    list(APPEND ARCH_DEFINITIONS __AMX_BF16__ GGML_AMX_BF16)
                endif()
            elseif (GGML_AVX2)
                list(APPEND ARCH_FLAGS /arch:AVX2)
                list(APPEND ARCH_DEFINITIONS GGML_AVX2 GGML_FMA GGML_F16C)
            elseif (GGML_AVX)
                list(APPEND ARCH_FLAGS /arch:AVX)
                list(APPEND ARCH_DEFINITIONS GGML_AVX)
            elseif (GGML_SSE42)
                list(APPEND ARCH_FLAGS /arch:SSE4.2)
                list(APPEND ARCH_DEFINITIONS GGML_SSE42)
            endif()
            if (GGML_AVX_VNNI)
                list(APPEND ARCH_DEFINITIONS __AVXVNNI__ GGML_AVX_VNNI)
            endif()
            if (GGML_BMI2)
                # MSVC does not define macro __BMI2__
                list(APPEND ARCH_DEFINITIONS __BMI2__ GGML_BMI2)
            endif()
        else ()
            if (GGML_NATIVE)
                list(APPEND ARCH_FLAGS -march=native)
            else ()
                if (GGML_SSE42)
                    list(APPEND ARCH_FLAGS -msse4.2)
                    list(APPEND ARCH_DEFINITIONS GGML_SSE42)
                endif()
                if (GGML_F16C)
                    list(APPEND ARCH_FLAGS -mf16c)
                    list(APPEND ARCH_DEFINITIONS GGML_F16C)
                endif()
                if (GGML_FMA)
                    list(APPEND ARCH_FLAGS -mfma)
                    list(APPEND ARCH_DEFINITIONS GGML_FMA)
                endif()
                if (GGML_BMI2)
                    list(APPEND ARCH_FLAGS -mbmi2)
                    list(APPEND ARCH_DEFINITIONS GGML_BMI2)
                endif()
                if (GGML_AVX)
                    list(APPEND ARCH_FLAGS -mavx)
                    list(APPEND ARCH_DEFINITIONS GGML_AVX)
                endif()
                if (GGML_AVX2)
                    list(APPEND ARCH_FLAGS -mavx2)
                    list(APPEND ARCH_DEFINITIONS GGML_AVX2)
                endif()
                if (GGML_AVX_VNNI)
                    list(APPEND ARCH_FLAGS -mavxvnni)
                    list(APPEND ARCH_DEFINITIONS GGML_AVX_VNNI)
                endif()
                if (GGML_AVX512)
                    list(APPEND ARCH_FLAGS -mavx512f)
                    list(APPEND ARCH_FLAGS -mavx512cd)
                    list(APPEND ARCH_FLAGS -mavx512vl)
                    list(APPEND ARCH_FLAGS -mavx512dq)
                    list(APPEND ARCH_FLAGS -mavx512bw)
                    list(APPEND ARCH_DEFINITIONS GGML_AVX512)
                endif()
                if (GGML_AVX512_VBMI)
                    list(APPEND ARCH_FLAGS -mavx512vbmi)
                    list(APPEND ARCH_DEFINITIONS GGML_AVX512_VBMI)
                endif()
                if (GGML_AVX512_VNNI)
                    list(APPEND ARCH_FLAGS -mavx512vnni)
                    list(APPEND ARCH_DEFINITIONS GGML_AVX512_VNNI)
                endif()
                if (GGML_AVX512_BF16)
                    list(APPEND ARCH_FLAGS -mavx512bf16)
                    list(APPEND ARCH_DEFINITIONS GGML_AVX512_BF16)
                endif()
                if (GGML_AMX_TILE)
                    list(APPEND ARCH_FLAGS -mamx-tile)
                    list(APPEND ARCH_DEFINITIONS GGML_AMX_TILE)
                endif()
                if (GGML_AMX_INT8)
                    list(APPEND ARCH_FLAGS -mamx-int8)
                    list(APPEND ARCH_DEFINITIONS GGML_AMX_INT8)
                endif()
                if (GGML_AMX_BF16)
                    list(APPEND ARCH_FLAGS -mamx-bf16)
                    list(APPEND ARCH_DEFINITIONS GGML_AMX_BF16)
                endif()
            endif()
        endif()

        if (GGML_BACKEND_DL)
            if (GGML_NATIVE)
                # the feature check relies on ARCH_DEFINITIONS, but it is not set with GGML_NATIVE
                message(FATAL_ERROR "GGML_NATIVE is not compatible with GGML_BACKEND_DL, consider using GGML_CPU_ALL_VARIANTS")
            endif()
            ggml_add_cpu_backend_features(${GGML_CPU_NAME} x86 ${ARCH_DEFINITIONS})
        endif()
    elseif (GGML_SYSTEM_ARCH STREQUAL "PowerPC")
        message(STATUS "PowerPC detected")
        list(APPEND GGML_CPU_SOURCES ggml-cpu/arch/powerpc/quants.c)
        if (GGML_NATIVE)
            if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64")
                file(READ "/proc/cpuinfo" POWER10_M)
            elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "powerpc")
                execute_process(COMMAND bash -c "prtconf |grep 'Implementation' | head -n 1" OUTPUT_VARIABLE POWER10_M)
            endif()

            string(TOUPPER "${POWER10_M}" POWER10_M_UPPER)
            string(REGEX MATCHALL "POWER *([0-9]+)" MATCHED_STRING "${POWER10_M_UPPER}")
            string(REGEX REPLACE "POWER *([0-9]+)" "\\1" EXTRACTED_NUMBER "${MATCHED_STRING}")

            if (EXTRACTED_NUMBER GREATER_EQUAL 10)
                list(APPEND ARCH_FLAGS -mcpu=power10 -mpowerpc64)
            elseif (EXTRACTED_NUMBER EQUAL 9)
                list(APPEND ARCH_FLAGS -mcpu=power9 -mpowerpc64)
            elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64le")
                list(APPEND ARCH_FLAGS -mcpu=powerpc64le -mtune=native)
            else()
                list(APPEND ARCH_FLAGS -mcpu=native -mtune=native -mpowerpc64)
            endif()
        elseif(GGML_CPU_ALL_VARIANTS)
            # Begin with the lowest baseline
            set(ARCH_DEFINITIONS "")

            # When a feature is selected, bump the MCPU to the first
            # version that supported it
            foreach(PVER RANGE 7 11)
                if(DEFINED GGML_INTERNAL_POWER${PVER})
                    set(POWERPC_MCPU "power${PVER}")
                    list(APPEND ARCH_DEFINITIONS GGML_USE_POWER${PVER})
                endif()
            endforeach()
            if (GGML_INTERNAL_VSX)
                list(APPEND ARCH_DEFINITIONS GGML_USE_VSX)
                list(APPEND ARCH_FLAGS -mvsx)
            endif()

            if (DEFINED POWERPC_MCPU)
                list(APPEND ARCH_FLAGS -mcpu=${POWERPC_MCPU})
            endif()
            ggml_add_cpu_backend_features(${GGML_CPU_NAME} powerpc ${ARCH_DEFINITIONS})
        else()
            if (GGML_CPU_POWERPC_CPUTYPE)
                list(APPEND ARCH_FLAGS -mcpu=${GGML_CPU_POWERPC_CPUTYPE})
            endif()
        endif()
    elseif (GGML_SYSTEM_ARCH STREQUAL "loongarch64")
        message(STATUS "loongarch64 detected")
        list(APPEND GGML_CPU_SOURCES ggml-cpu/arch/loongarch/quants.c)

        list(APPEND ARCH_FLAGS -march=loongarch64)
        if (GGML_LASX)
            list(APPEND ARCH_FLAGS -mlasx)
        endif()
        if (GGML_LSX)
            list(APPEND ARCH_FLAGS -mlsx)
        endif()
    elseif (GGML_SYSTEM_ARCH STREQUAL "riscv64")
        message(STATUS "riscv64 detected")
        list(APPEND GGML_CPU_SOURCES
            ggml-cpu/arch/riscv/quants.c
            ggml-cpu/arch/riscv/repack.cpp
            )
        if (GGML_CPU_RISCV64_SPACEMIT)
            target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_CPU_RISCV64_SPACEMIT ${RISCV64_SPACEMIT_IME_SPEC})
            list(APPEND GGML_CPU_SOURCES
                ggml-cpu/spacemit/ime.cpp
                ggml-cpu/spacemit/ime.h
                ggml-cpu/spacemit/ime1_kernels.cpp
                ggml-cpu/spacemit/ime_kernels.h
            )
        endif()
        set(MARCH_STR "rv64gc")
        if (GGML_RV_ZFH)
            string(APPEND MARCH_STR "_zfh")
        endif()
        if (GGML_XTHEADVECTOR)
            string(APPEND MARCH_STR "_xtheadvector")
        elseif (GGML_RVV)
            string(APPEND MARCH_STR "_v")
            if (GGML_RV_ZVFH)
                string(APPEND MARCH_STR "_zvfh")
            endif()
        endif()
        if (GGML_RV_ZICBOP)
            string(APPEND MARCH_STR "_zicbop")
        endif()
        list(APPEND ARCH_FLAGS "-march=${MARCH_STR}" -mabi=lp64d)
    elseif (GGML_SYSTEM_ARCH STREQUAL "s390x")
        message(STATUS "s390x detected")
        list(APPEND GGML_CPU_SOURCES
            ggml-cpu/arch/s390/quants.c)

        # for native compilation
        if (GGML_NATIVE)
            # check machine level to determine target
            file(READ "/proc/cpuinfo" CPUINFO_CONTENTS)
            string(REGEX REPLACE "machine[ \t\r\n]*=[ \t\r\n]*([0-9]+)" "\\1" S390X_M ${CPUINFO_CONTENTS})

            # TODO: Separation to determine activation of VX/VXE/VXE2
            if (${S390X_M} MATCHES "8561|8562")
                message(STATUS "z15 target")
                list(APPEND ARCH_FLAGS -march=z15)
            elseif (${S390X_M} MATCHES "3931")
                message(STATUS "z16 target")
                list(APPEND ARCH_FLAGS -march=z16)
            elseif (${S390X_M} MATCHES "9175|9176")
                # NOTE: Only available from GCC 15.1.0 onwards. Any z17 machine with compile issues must first verify their GCC version.
                #       binutils must also be updated to the latest for the -march=z17 flag to work. Otherwise, use -march=arch15.
                message(STATUS "z17 target")
                list(APPEND ARCH_FLAGS -march=arch15)
            else()
                message(STATUS "Unknown target")
                message(WARNING "Unknown target. If you are compiling for z14 and earlier, you might have to add -DGGML_VXE=OFF.")
                list(APPEND ARCH_FLAGS -march=native -mtune=native)
            endif()
        # for cross-compilation
        elseif(GGML_CPU_ALL_VARIANTS)
            # range through IBM z15 to z17
            # NOTE: update when a new hardware level is released
            foreach (ZHW RANGE 15 17)
                if(DEFINED GGML_INTERNAL_Z${ZHW})
                    message(STATUS "z${ZHW} cross-compile target")
                    list(APPEND ARCH_FLAGS -march=z${ZHW})
                endif()
            endforeach()
        endif()

        if (GGML_VXE OR GGML_INTERNAL_VXE2)
            message(STATUS "VXE2 enabled")
            list(APPEND ARCH_FLAGS -mvx -mzvector)
            list(APPEND ARCH_DEFINITIONS GGML_USE_VXE2)
        endif()

        if (GGML_INTERNAL_NNPA)
            message(STATUS "NNPA enabled")
            list(APPEND ARCH_DEFINITIONS GGML_USE_NNPA)
        endif()

        ggml_add_cpu_backend_features(${GGML_CPU_NAME} s390 ${ARCH_DEFINITIONS})
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "wasm")
        message(STATUS "Wasm detected")
        list (APPEND GGML_CPU_SOURCES ggml-cpu/arch/wasm/quants.c)
    else()
        message(WARNING "Unknown CPU architecture. Falling back to generic implementations.")
        list(APPEND ARCH_FLAGS -DGGML_CPU_GENERIC)
    endif()

    if (GGML_CPU_REPACK)
        target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_CPU_REPACK)
    endif()

    if (GGML_CPU_KLEIDIAI)
        message(STATUS "Using KleidiAI optimized kernels if applicable")

        # Disable the KleidiAI tests
        set(KLEIDIAI_BUILD_TESTS  OFF)

        # Fetch KleidiAI sources:
        include(FetchContent)
        set(KLEIDIAI_COMMIT_TAG "v1.14.0")
        set(KLEIDIAI_DOWNLOAD_URL "https://github.com/ARM-software/kleidiai/archive/refs/tags/${KLEIDIAI_COMMIT_TAG}.tar.gz")
        set(KLEIDIAI_ARCHIVE_MD5  "45e110675d93f99f82c23a1afcca76bc")

        if (POLICY CMP0135)
            cmake_policy(SET CMP0135 NEW)
        endif()

        FetchContent_Declare(KleidiAI_Download
            URL ${KLEIDIAI_DOWNLOAD_URL}
            DOWNLOAD_EXTRACT_TIMESTAMP NEW
            URL_HASH MD5=${KLEIDIAI_ARCHIVE_MD5})

        FetchContent_MakeAvailable(KleidiAI_Download)
        FetchContent_GetProperties(KleidiAI_Download
            SOURCE_DIR  KLEIDIAI_SRC
            POPULATED   KLEIDIAI_POPULATED)

        if (NOT KLEIDIAI_POPULATED)
            message(FATAL_ERROR "KleidiAI source downloaded failed.")
        endif()

        add_compile_definitions(GGML_USE_CPU_KLEIDIAI)

        # Remove kleidiai target after fetching it
        if (TARGET kleidiai)
            set_target_properties(kleidiai PROPERTIES EXCLUDE_FROM_ALL TRUE)
        endif()

        list(APPEND GGML_CPU_SOURCES
            ggml-cpu/kleidiai/kleidiai.cpp
            ggml-cpu/kleidiai/kernels.cpp
            ggml-cpu/kleidiai/kleidiai.h
            ggml-cpu/kleidiai/kernels.h
            )

        # KleidiAI
        include_directories(
            ${KLEIDIAI_SRC}/
            ${KLEIDIAI_SRC}/kai/
            ${KLEIDIAI_SRC}/kai/ukernels/
            ${KLEIDIAI_SRC}/kai/ukernels/matmul/
            ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/
            ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/
            ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_fp32_bf16p_bf16p/
            ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/)

        set(ARCH_FLAGS_TEMP "${ARCH_FLAGS}")
        if (NOT ARCH_FLAGS_TEMP)
            string(REGEX MATCH "-march=[^ ]+" ARCH_FLAGS_TEMP "${CMAKE_C_FLAGS}")
        endif()
        string(FIND "${ARCH_FLAGS_TEMP}" "+dotprod" DOTPROD_ENABLED)
        string(FIND "${ARCH_FLAGS_TEMP}" "+i8mm" I8MM_ENABLED)
        string(FIND "${ARCH_FLAGS_TEMP}" "+sme" SME_ENABLED)

        set(PRIVATE_ARCH_FLAGS ${ARCH_FLAGS_TEMP})

        list(APPEND GGML_KLEIDIAI_SOURCES
            ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32.c
            ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p4x8sb_f32_neon.c
            ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon.c
            ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32_neon.c
            ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.c
            ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.c
            ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.c)

        if (NOT DOTPROD_ENABLED MATCHES -1)
            list(APPEND GGML_KLEIDIAI_SOURCES
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.c
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod.c
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod.c
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod.c
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod.c
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod.c)
        endif()

        if (NOT I8MM_ENABLED MATCHES -1)
            list(APPEND GGML_KLEIDIAI_SOURCES
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm.c
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.c)
        endif()

        if (NOT SME_ENABLED MATCHES -1)
            list(APPEND GGML_KLEIDIAI_SOURCES
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa.c
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot.c
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa.c
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa_asm.S
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot.c
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot_asm.S
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_fp32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa.c
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_fp32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa_asm.S
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_lhs_pack_bf16p2vlx2_f32_sme.c
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme.c
                ${KLEIDIAI_SRC}/kai/kai_common_sme_asm.S)
            set(PRIVATE_ARCH_FLAGS "-fno-tree-vectorize;${PRIVATE_ARCH_FLAGS}+sve+sve2")
        endif()

        set_source_files_properties(${GGML_KLEIDIAI_SOURCES} PROPERTIES COMPILE_OPTIONS "${PRIVATE_ARCH_FLAGS}")
        list(APPEND GGML_CPU_SOURCES ${GGML_KLEIDIAI_SOURCES})
    endif()

    message(STATUS "Adding CPU backend variant ${GGML_CPU_NAME}: ${ARCH_FLAGS} ${ARCH_DEFINITIONS}")
    target_sources(${GGML_CPU_NAME} PRIVATE ${GGML_CPU_SOURCES})
    target_compile_options(${GGML_CPU_NAME} PRIVATE ${ARCH_FLAGS})
    target_compile_definitions(${GGML_CPU_NAME} PRIVATE ${ARCH_DEFINITIONS})

    if (EMSCRIPTEN)
        set_target_properties(${GGML_CPU_NAME} PROPERTIES COMPILE_FLAGS "-msimd128")
    endif()

    if (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        # The compiler automatically enables "-ffast-math" which can cause NaNs in tests due to "-fassociative-math"
        target_compile_options(${GGML_CPU_NAME} PRIVATE "-fno-associative-math")
    endif()
endfunction()
