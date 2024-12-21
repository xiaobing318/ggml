# ggml
```c
/*
Note:杨小兵-2024-12-21

1、这个项目的名称叫做ggml
2、ggml的作用是为了能够推理LLMs（对于这部分的理解不是很深入）
*/
```

[Roadmap](https://github.com/users/ggerganov/projects/7) / [Manifesto](https://github.com/ggerganov/llama.cpp/discussions/205)
```c
/*
Note:杨小兵-2024-12-21

1、通过点击文件便可以跳转到对应的网页链接上，在markdown文件中通过[](web url)便可以实现这种效果
2、两个网址
   2.1 该项目的Roadmap
   2.2 该项目的‘宣言’：Inference at the edge（这部分内容还是比较有意思的，可以了解一下）
*/
```

Tensor library for machine learning
```c
/*
Note:杨小兵-2024-12-21

1、ggml是一个库，这个库具体来说就是tensor library for machine learning
2、tensor的解释
   2.1 Tensor是一个多维数组，广泛用于机器学习和深度学习中。它可以表示标量（0维）、向量（1维）、矩阵（2维）以及更高维度的数据结构。Tensor是深度学习框架（如TensorFlow和PyTorch）的核心数据结构，用于存储和操作数据。
   2.2 Tensor可以用于各种数学运算，如加法、减法、乘法和矩阵乘法。
*/
```

***Note that this project is under active development. \
Some of the development is currently happening in the [llama.cpp](https://github.com/ggerganov/llama.cpp) and [whisper.cpp](https://github.com/ggerganov/whisper.cpp) repos***
```c
/*
Note:杨小兵-2024-12-21

1、注意：ggml这个项目还处于积极开发状态
2、一些开发工作正在下列两个项目中开展
   2.1 llama.cpp
   2.2 whisper.cpp
3、llama.cpp和whisper.cpp两个项目应该都用到了ggml这个库的成果
*/
```

## 1 Features

- Low-level cross-platform implementation
- Integer quantization support
- Broad hardware support
- Automatic differentiation
- ADAM and L-BFGS optimizers
- No third-party dependencies
- Zero memory allocations during runtime
```c
/*
Note:杨小兵-2024-12-21

1、这部分内容将会介绍ggml的一些features
2、低级别的跨平台实现
   ggml库提供了低级别的跨平台实现，这意味着它可以在不同的操作系统和硬件平台上运行，而无需进行大量修改。
3、整数量化支持
   ggml库支持整数量化，这意味着可以将浮点数表示的模型参数转换为整数表示，从而减少模型的存储空间和计算开销。
3、广泛的硬件支持
   ggml库支持多种硬件平台，包括CPU、GPU和其他加速器。这意味着无论是在个人电脑、服务器还是嵌入式设备上，都可以高效地运行ggml库。
4、自动微分
   ggml库支持自动微分，这是一种计算函数导数的技术，广泛用于机器学习中的反向传播算法。自动微分可以自动计算出模型参数的梯度，从而简化了模型训练过程。
5、ADAM和L-BFGS优化器
   ggml库内置了多种优化器，包括ADAM和L-BFGS。ADAM是一种自适应学习率优化算法，适用于处理稀疏梯度和大规模数据集。L-BFGS是一种拟牛顿法优化算法，适用于处理高维优化问题。
6、无第三方依赖
   ggml库不依赖任何第三方库，这意味着在构建和部署时不需要额外安装其他依赖项，从而简化了安装和使用过程。
7、运行时零内存分配
   ggml库在运行时不进行任何内存分配，这意味着在模型推理和训练过程中不会产生额外的内存开销，从而提高了性能和效率。
*/
```

## 2 Build
```c
/*
Note:杨小兵-2024-12-21

1、这部分内容将会解释如何buid ggml library
2、这个库需要跨平台支持，因此应该会针对不同的平台存在不同的build method
*/
```

```bash
git clone https://github.com/ggerganov/ggml
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
```c
/*
Note:杨小兵-2024-12-21

1、整个操作是在linux环境中执行的
2、git clone https://github.com/ggerganov/ggml
   使用git将ggml库从github上clone到本地
3、cd ggml
   进入到ggml文件夹中
4、python3.10 -m venv .venv
   创建python的虚拟环境
5、source .venv/bin/activate
   配置终端环境
6、pip install -r requirements.txt
   在python环境中安装依赖的库
7、mkdir build && cd build
   创建build目录并且进入到build文件夹中
8、cmake ..
   使用camke在build中构建外层的项目从而隔离源代码和构建项目
9、cmake --build . --config Release -j 8
   9.1 --build .           指定构建目录
   9.2 --config Release    指定项目构建类型
   9.3 -j 8                使用8线程并行构建
*/
```

## 3 GPT inference (example)
```c
/*
Note:杨小兵-2024-12-21

1、这部分内容给出一个示例来演示ggml的作用
2、这里给出的例子是GPT inference
*/
```

```bash
# run the GPT-2 small 117M model
../examples/gpt-2/download-ggml-model.sh 117M
./bin/gpt-2-backend -m models/gpt-2-117M/ggml-model.bin -p "This is an example"
```
```c
/*
Note:杨小兵-2024-12-21

1、命令解释
   1.1 使用download-ggml-model.sh下载GPT模型
   1.2 使用gpt-2-backend运行下载的GPT模型
*/
```

For more information, checkout the corresponding programs in the [examples](examples) folder.
```c
/*
Note:杨小兵-2024-12-21

1、对于GPT inference的更多信息，查看examples文件夹中对应的示例程序
*/
```

## 4 Using CUDA

```bash
# fix the path to point to your CUDA compiler
cmake -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.1/bin/nvcc ..
```
```c
/*
Note:杨小兵-2024-12-21

1、在构建GGML库的时候通过添加宏定义来控制构建选项
2、宏定义
   2.1 GGML_CUDA              使用CUDA构建GGML
   2.2 CMAKE_CUDA_COMPILER    指定CUDA编译器的路径
*/
```

## 5 Using hipBLAS

```bash
cmake -DCMAKE_C_COMPILER="$(hipconfig -l)/clang" -DCMAKE_CXX_COMPILER="$(hipconfig -l)/clang++" -DGGML_HIP=ON
```
```c
/*
Note:杨小兵-2024-12-21

1、在构建GGML库的时候通过添加宏定义来控制构建选项
2、宏定义
   2.1 CMAKE_C_COMPILER       指定C编译器的路径
   2.2 CMAKE_CXX_COMPILER     指定C++编译器的路径
   2.3 GGML_HIP               使用HIP构建GGML
*/
```

## 6 Using SYCL

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
Note:杨小兵-2024-12-21

1、在构建GGML库的时候通过添加宏定义来控制构建选项
2、宏定义
   2.1 CMAKE_C_COMPILER       指定C编译器
   2.2 CMAKE_CXX_COMPILER     指定C++编译器
   2.3 GGML_SYCL               使用HSYCL构建GGML
*/
```

## 7 Compiling for Android

Download and unzip the NDK from this download [page](https://developer.android.com/ndk/downloads). Set the NDK_ROOT_PATH environment variable or provide the absolute path to the CMAKE_ANDROID_NDK in the command below.
```c
/*
Note:杨小兵-2024-12-21

1、从指定的网址中下在NDK并且将下载的zip包进行解压
2、设置NDK_ROOT_PATH环境变量或者在下列的命令行中提供CMAKE_ANDROID_NDK的绝对路径
3、这部分内容不熟悉，等到需要的时候在弄清楚
*/
```

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

## 8 Resources

- [Introduction to ggml](https://huggingface.co/blog/introduction-to-ggml)
- [The GGUF file format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
```c
/*
Note:杨小兵-2024-12-21

1、[Introduction to ggml]对GGML介绍有关资料
2、[The GGUF file format]对GGUF文件格式介绍有关资料
*/
```