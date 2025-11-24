# ggml

```c
/*
1. ggml 是 Georgi Gerganov Machine Learning 的缩写，其中 Georgi Gerganov 是 Github 上的一个开发者名称。
2. ggml 仓库使用下列编程语言开发，可以发现 C/C++/CUDA 三种编程语言占据了大多数比例：
  2.1 C     22%
  2.2 C++   57.8%
  2.3 CUDA  10.9%
  2.4 Metal 3.6%
  2.5 GLSL  2.1%
  2.6 CMake 1.4%
  2.7 Other 2.2%
*/
```
[Roadmap](https://github.com/users/ggerganov/projects/7) / [Manifesto](https://github.com/ggerganov/llama.cpp/discussions/205)
```c
/*
1. Roadmap 中记录了 TODO/DONE 列表即打算完成哪些事项/已经完成了哪些事项。
2. Manifesto 可以翻译成宣言/声明书，其目的是说明一些理念、原则、方向等等，例如 llama.cpp 宣言中的一条是希望能够使得人人可以在本地运行 large language model 。
*/
```

Tensor library for machine learning
```c
/*
1. ggml 是一个专为 machine learning 场景开发的 tensor libray 。
2. 理解
  2.1 从数学的角度来理解：tensor library 是一个 multi-dimension array ，其包含 operator 和 operand 两个方面。
  2.2 从计算机实现的角度：tensor library 是一个 multi-dimension array ，其不仅仅包含 operator 和 operand 两个方面，还有很多相匹配的其它内容，例如内存管理、算子调度等等。
*/
```

***Note that this project is under active development. \
Some of the development is currently happening in the [llama.cpp](https://github.com/ggerganov/llama.cpp) and [whisper.cpp](https://github.com/ggerganov/whisper.cpp) repos***
```c
/*
1. 需要注意的是一些开发工作正在 llama.cpp 项目中开展，也就是说有些是在 ggml 中开发实现的而另外一些则是在 llama.cpp/whisper 中开发实现的，llama.cpp/whisper 中开发实现之后再同步到 ggml 项目中，之前我理解的是只有 ggml 到 llama.cpp/whisper 这单个方向。
*/
```

## Features

- Low-level cross-platform implementation
- Integer quantization support
- Broad hardware support
- Automatic differentiation
- ADAM and L-BFGS optimizers
- No third-party dependencies
- Zero memory allocations during runtime
```c
/*
1. ggml 特性：
  1.1 使用接近低层硬件的语言开发的跨平台实现
  1.2 支持多种整数量化
  1.3 支持不同 CPU/GPU/Accelerator 架构
  1.4 支持自动微分
  1.5 支持 ADAM/L-BFGS 优化器
  1.6 无第三方库依赖
  1.7 运行时零内存分配
*/
```

## Build

```bash
git clone https://github.com/ggml-org/ggml
cd ggml

# install python dependencies in a virtual environment
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# build the examples
mkdir build && cd build
cmake ..
cmake --build . --config Release -j 8
```

## GPT inference (example)

```bash
# run the GPT-2 small 117M model
../examples/gpt-2/download-ggml-model.sh 117M
./bin/gpt-2-backend -m models/gpt-2-117M/ggml-model.bin -p "This is an example"
```

For more information, checkout the corresponding programs in the [examples](examples) folder.

## Using CUDA

```bash
# fix the path to point to your CUDA compiler
cmake -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.1/bin/nvcc ..
```

## Using hipBLAS

```bash
cmake -DCMAKE_C_COMPILER="$(hipconfig -l)/clang" -DCMAKE_CXX_COMPILER="$(hipconfig -l)/clang++" -DGGML_HIP=ON
```

## Using SYCL

```bash
# linux
source /opt/intel/oneapi/setvars.sh
cmake -G "Ninja" -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL=ON ..

# windows
"C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
cmake -G "Ninja" -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=icx -DGGML_SYCL=ON ..
```
```c
/*
1. setvars.sh 脚本文件和 setvars.bat 脚本文件目的都是为了配置 SYCL 环境，只不过操作系统平台是不同的，通过这个对比可以更好的理解 bash 脚本文件和 Windows patch 脚本文件的共同点了。
*/
```

## Compiling for Android

Download and unzip the NDK from this download [page](https://developer.android.com/ndk/downloads). Set the NDK_ROOT_PATH environment variable or provide the absolute path to the CMAKE_ANDROID_NDK in the command below.

```bash
cmake .. \
   -DCMAKE_SYSTEM_NAME=Android \
   -DCMAKE_SYSTEM_VERSION=33 \
   -DCMAKE_ANDROID_ARCH_ABI=arm64-v8a \
   -DCMAKE_ANDROID_NDK=$NDK_ROOT_PATH
   -DCMAKE_ANDROID_STL_TYPE=c++_shared
```

```bash
# create directories
adb shell 'mkdir /data/local/tmp/bin'
adb shell 'mkdir /data/local/tmp/models'

# push the compiled binaries to the folder
adb push bin/* /data/local/tmp/bin/

# push the ggml library
adb push src/libggml.so /data/local/tmp/

# push model files
adb push models/gpt-2-117M/ggml-model.bin /data/local/tmp/models/

adb shell
cd /data/local/tmp
export LD_LIBRARY_PATH=/data/local/tmp
./bin/gpt-2-backend -m models/ggml-model.bin -p "this is an example"
```

## Resources

- [Introduction to ggml](https://huggingface.co/blog/introduction-to-ggml)
- [The GGUF file format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
