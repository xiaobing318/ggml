1. **“BLAS 供应商”其实就是“给 BLAS/CBLAS 接口提供实现的那家库”**，比如 Apple Accelerate、OpenBLAS、BLIS、Intel oneMKL、ArmPL、AOCL 等。
2. CMake 的 `FindBLAS` 用 `BLA_VENDOR` 这串字符串来控制“优先找哪一家的 BLAS”，`GGML_BLAS_VENDOR_DEFAULT` 只是给 `BLA_VENDOR` 预填的默认值，最终成不成功取决于：**系统上有没有装对应的库 + 你的 CMake 选项**。([cmake.org][1])
3. GPU 侧的 cuBLAS/rocBLAS 不在 `FindBLAS` 这套 CPU BLAS 供应商体系里，在 ggml 里是通过 `GGML_CUDA/GGML_HIP` 等单独开关控制的。([huggingface.co][2])

---

## 一、BLAS 供应商 & BLA_VENDOR 是什么？

* **BLAS（Basic Linear Algebra Subprograms）**：一套*标准接口*，做矩阵乘、向量运算等数值线代操作。Netlib 给了一个“参考实现（reference implementation）”，性能不高但定义了“长什么样”。([netlib.org][3])

* **BLAS 供应商（vendor）**：谁提供了一个“符合 BLAS 接口”的库，比如：

  * **Apple**：Accelerate/vecLib
  * **OpenBLAS**
  * **BLIS**
  * **Intel oneMKL**
  * **Arm Performance Libraries**
  * **AMD AOCL**
  * **FlexiBLAS / libblastrampoline** 这类“多路复用包装器”

* **CMake FindBLAS 模块**：负责去系统上找一个实现 BLAS 接口的库。你可以写：

  ```cmake
  # 想用 OpenBLAS
  set(BLA_VENDOR OpenBLAS)
  find_package(BLAS REQUIRED)
  target_link_libraries(myapp PRIVATE BLAS::BLAS)
  ```

  里面 `BLA_VENDOR` 必须是 CMake 文档列出的某个字符串，比如 `Apple` / `OpenBLAS` / `Intel10_64lp` / `Arm` / `Generic` 等。([cmake.org][1])

* **Generic**：在 CMake 语境中是“不要特指某家，随便找一个实现（常常就是 Netlib BLAS 或发行版默认那一套）”。([cmake.org][1])

**关键点：**`BLA_VENDOR` 只是“筛选规则”，**不会帮你安装任何库**；找不到就 `BLAS_FOUND=FALSE` 报错。

---

## 二、按平台看常见 BLAS 组合

### 1. macOS：Apple Accelerate 为主

1. **Apple Accelerate / vecLib**

   * 在 CMake 里对应 `BLA_VENDOR=Apple` 或 `NAS`。([cmake.org][1])

   * Apple 在 Accelerate framework 里提供了 BLAS + LAPACK，和 LAPACK 3.9.1 保持一致。([Apple Developer][4])

   * 链接方式一般是：

     ```cmake
     find_package(BLAS)
     target_link_libraries(myapp PRIVATE BLAS::BLAS)  # 实际上就是 -framework Accelerate
     ```

   * 优点：系统自带、零配置、对 Intel + Apple Silicon 都做过优化。

   * 缺点：只有 Apple 系统上有，不能移植到 Linux/Windows。

2. **macOS 上的其它选择**

   * **OpenBLAS**：例如通过 Homebrew 装 `brew install openblas`，然后设 `BLA_VENDOR=OpenBLAS`。([Wikipedia][5])
   * **Intel oneMKL**：Intel oneAPI on macOS（新一点的 macOS/硬件支持有限），`BLA_VENDOR` 用 `Intel10_64lp` 等。([cmake.org][1])

**简单经验：**

* 想省事：**直接用 Accelerate**（BLA_VENDOR 不设，让 CMake 自动找到 Apple）。
* 想统一和 Linux/Windows 上的环境：可以手动装 OpenBLAS，然后 `BLA_VENDOR=OpenBLAS`。

---

### 2. Linux：最“物种丰富”的生态

这里按“家族/用途”来分。

#### (1) Netlib Reference BLAS —— Generic

* 来自 Netlib 的“参考实现”，设计目标是：**正确 & 可移植，不追求极致性能**。([netlib.org][3])
* 在 CMake 里一般对应 `BLA_VENDOR=Generic`。([cmake.org][1])
* 很多发行版早期/极简安装环境里就只提供这一套。

**适用场景：**

* 算法验证、教学代码、对性能要求不高的工具。

#### (2) OpenBLAS —— Linux 世界的常用默认

* OpenBLAS 是从 GotoBLAS2 fork 出来的开源高性能 BLAS+LAPACK 实现，对多种 CPU 微架构（Sandy Bridge、Loongson 等）做了手工优化。([Wikipedia][5])
* 支持 x86_64、ARM、POWER 等平台，线程级并行、多种指令集（AVX/AVX2…）。([Wikipedia][5])
* CMake 中 `BLA_VENDOR=OpenBLAS`，是 doc 里显式列出的 vendor。([cmake.org][1])

**发行版情况：**

* Ubuntu/Debian/Fedora 等都提供 `libopenblas-dev` 等包。很多 Python/NumPy 打包也默认用 OpenBLAS。([hpcc.umd.edu][6])

#### (3) BLIS（FLAME）—— 更现代的框架

* **BLIS（BLAS-like Library Instantiation Software）**：一个可扩展的高性能 BLAS 框架，支持 BLAS / CBLAS 接口，还扩展出自己的 object API。([GitHub][7])
* CMake 中把它叫 `BLA_VENDOR=FLAME`。([cmake.org][1])
* 目标是：用少量内核代码 + 缓存/寄存器优化，实现多平台高性能；还有 ILP64/LP64、多线程支持。([cs.utexas.edu][8])

很多科学计算环境（特别是 RHEL/CentOS 派生的）开始提供 BLIS 做为可选 BLAS 后端。([Quansight Labs][9])

#### (4) Arm Performance Libraries（ArmPL）

* `BLA_VENDOR=Arm / Arm_mp / Arm_ilp64 / Arm_ilp64_mp`。([cmake.org][1])
* Arm 官方的高性能数学库套件，专门针对 AArch64 进行指令级优化（SVE、NEON 等）。([community.arm.com][10])
* 常见于 HPC 集群、Arm 服务器（Ampere、Fujitsu A64FX 等）。

#### (5) AMD AOCL / ACML

* 老一代：**ACML（AMD Core Math Library）**，对应 `ACML`/`ACML_MP`/`ACML_GPU`。
* 新一代：**AOCL（AMD Optimizing CPU Libraries）**，`BLA_VENDOR=AOCL / AOCL_mt`。([cmake.org][1])
* 目标类似 MKL/ArmPL，为自家 CPU 做最优内核。

#### (6) ATLAS

* Automatically Tuned Linear Algebra Software，安装时会做“自适应调优”，自动选择 tile 和内核。([cmake.org][1])
* CMake 中 `BLA_VENDOR=ATLAS`。
* 在现代发行版里存在感逐渐被 OpenBLAS/BLIS 取代，但老系统里还常见。

#### (7) FlexiBLAS：运行时可切换的“多路复用器”

* FlexiBLAS 是一个 **wrapper library**，本身实现的是 BLAS/LAPACK 接口，但内部再转发到真正的后端（OpenBLAS/BLIS/Netlib/MKL…），并允许 *运行时切换*。([GitHub][11])
* CMake 中 `BLA_VENDOR=FlexiBLAS`。([cmake.org][1])
* Quansight 之类的 Python 科学栈打包经常用 FlexiBLAS 作为统一入口，再后挂多个实现。([Quansight Labs][9])

**优点**：不用重新编译/链接，就能换 BLAS back-end，非常适合发行版/科学平台。
**代价**：多一层间接调用，性能损失一般很小（函数调用级别）。

#### (8) libblastrampoline（LBT）

* `BLA_VENDOR=libblastrampoline`，是新近加入的一个 vendor。([cmake.org][1])
* 这也是一个“demux”库，利用 PLT trampolines 在运行时跳转到真正的 BLAS/LAPACK。这种方式在 Julia、现代 SciPy 生态里非常流行。([Quansight Labs][9])

---

### 3. Windows：MKL + OpenBLAS 为主

#### (1) Intel oneMKL

* Intel 数学核心库，一般是 oneAPI 的一部分。

* CMake 文档里列出了一堆 vendor：`Intel`, `Intel10_32`, `Intel10_64lp`, `Intel10_64lp_seq`, `Intel10_64ilp`, `Intel10_64ilp_seq`, `Intel10_64_dyn` 等。([cmake.org][1])

* 使用方式（Linux 示例）：

  ```bash
  . /opt/intel/oneapi/setvars.sh
  cmake -DBLA_VENDOR=Intel10_64lp ..
  ```

  CMake 文档里要求先加载 oneAPI 环境脚本来设置 `MKLROOT`、LD_LIBRARY_PATH 等。([cmake.org][1])

* 在 Windows 上类似，要用 Intel 提供的 env 脚本或 VS 集成环境。

**特点**：对 Intel CPU 极致优化、多线程、支持各种矩阵格式和扩展接口；闭源，但免费社区版广泛可用。

#### (2) OpenBLAS / BLIS 在 Windows

* OpenBLAS 可以通过：

  * vcpkg / MSYS2 / conda-forge 安装，对应 `BLA_VENDOR=OpenBLAS`。([Wikipedia][5])
* BLIS 可以手动编译或通过部分包管理器获取，对应 `BLA_VENDOR=FLAME`。([GitHub][7])

**实际情况**：
在 Windows 上，**最常见组合就是 oneMKL 或 OpenBLAS**；BLIS/ATLAS 在 Windows 的预编译包并不普遍。

---

### 4. GPU 侧库：cuBLAS / rocBLAS / NVPL

* cuBLAS（NVIDIA）、rocBLAS（AMD）、oneMKL GPU backend 等属于 **GPU BLAS**，通常不会通过 `find_package(BLAS)` 来找，而是 `find_package(CUDA)`、`find_package(rocblas)` 或供应商自己的 CMake 模块。
* CMake `BLAS/LAPACK Vendors` 里有 `NVPL`（NVIDIA Performance Libraries），支持线程模型选择（`BLA_THREAD=OMP/SEQ`），但本质上仍是“BLAS 接口层”的 vendor，和 ggml 里 `GGML_CUDA` 开关是两个维度。([cmake.org][1])

在 ggml / llama.cpp 里：

* CPU BLAS：由 `GGML_BLAS` + `GGML_BLAS_VENDOR` 控制。([huggingface.co][2])
* GPU 后端：`GGML_CUDA` / `GGML_HIP` / `GGML_VULKAN` / `GGML_METAL` 等单独开关。([huggingface.co][2])

---

## 三、CMake FindBLAS 行为 + ggml 的选项关系

`FindBLAS` 的关键输入变量：([cmake.org][1])

* `BLA_VENDOR`：指定供应商（上面那张 vendor 表的字符串）。
* `BLA_STATIC`：是否只找静态库（`.a` / `.lib`）。
* `BLA_SIZEOF_INTEGER`：选择 32-bit（4）、64-bit（8）还是 ANY 的 BLAS 整数接口（LP64 vs ILP64）。
* `BLA_THREAD`：在支持的 vendor（当前主要是 NVPL）上选 SEQ / OMP / ANY。

对应到 ggml（llama.cpp）：

```cmake
option(GGML_BLAS "ggml: use BLAS" ${GGML_BLAS_DEFAULT})
set(GGML_BLAS_VENDOR ${GGML_BLAS_VENDOR_DEFAULT} CACHE STRING
    "ggml: BLAS library vendor")
```

在它的 `ggml-blas` 子目录里，会根据 `GGML_BLAS_VENDOR` 的值去匹配 Apple / OpenBLAS / Intel 等，然后调用 `find_package(BLAS)` 并做一些特殊链接处理（比如 Apple 要加 `Accelerate`、OpenBLAS 要加额外库等）。([fossies.org][12])

**要点：**

* `GGML_BLAS_VENDOR_DEFAULT` 只是在不同平台上给一个“合理猜测”的默认值，实际生效的是你最终 cache 里的 `GGML_BLAS_VENDOR`/`BLA_VENDOR`。
* 如果你手动设 `-DGGML_BLAS_VENDOR=OpenBLAS`，但系统上 OpenBLAS 没装好，`FindBLAS` 就会像不少 issue 里那样报 `Could NOT find BLAS (missing: BLAS_LIBRARIES)`。([GitHub][13])

---

## 四、实战：不同平台怎么选 vendor？

给几个典型组合，方便你在 ggml/llama.cpp 里脑补：

1. **Ubuntu + Intel/AMD x86_64 桌面机**

   * 只图方便：

     * 安装 `libopenblas-dev`，`-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS`。
   * HPC 场景、已经安装 oneMKL：

     * `-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Intel10_64lp`（lp64，多线程）或 `Intel10_64lp_seq`（单线程）。([cmake.org][1])

2. **Ubuntu / Debian + Ampere/Arm 服务器**

   * 手上有 ArmPL：

     * `-DGGML_BLAS_VENDOR=Arm` 或 `Arm_mp`。([cmake.org][1])
   * 没有 ArmPL，就用 OpenBLAS：

     * `-DGGML_BLAS_VENDOR=OpenBLAS`。

3. **macOS + Apple Silicon (M1/M2/M3)**

   * 推荐直接用 Accelerate：

     * 不用显式设 vendor，让 CMake 自动找到 Apple；如果想写死就 `-DGGML_BLAS_VENDOR=Apple`。([Apple Developer][4])

4. **Windows + Intel CPU**

  * 有 oneAPI：
    * 在 oneAPI 控制台里打开，`-DGGML_BLAS_VENDOR=Intel10_64lp`。([cmake.org][1])
  * 想开源 + 方便部署：
    * 用 vcpkg/MSYS2 安装 OpenBLAS，`-DGGML_BLAS_VENDOR=OpenBLAS`。

---

## 五、性能 & 设计取舍简单聊一嘴

大部分 LLM 推理场景里，BLAS 主要影响：

* **大 batch prompt 阶段的矩阵乘**（例如 llama.cpp 文档就明确说，开启 BLAS 主要改善大 batch 的提示处理阶段性能，对 token-by-token 的生成提升有限）。([GitHub][14])

设计上的几个权衡：

1. **参考实现 vs 高性能实现**

   * Netlib BLAS：简单、可移植、慢。
   * OpenBLAS/BLIS/MKL/AOCL/ArmPL：复杂、平台相关，但能吃满 SIMD 和多核心。

2. **单线程 vs 多线程**

   * 单线程 BLAS：线程安全简单，不会和你应用自己的多线程抢核。
   * 多线程 BLAS：单进程算力更满，但在应用本身也多线程时容易“线程超订阅”，需要小心设置 `OMP_NUM_THREADS` 等环境变量。

3. **LP64 vs ILP64（32-bit vs 64-bit BLAS 整数）**

   * 大部分应用用 LP64（`int` 32-bit）。
   * 超大矩阵（N > 2^31）才需要 ILP64（`long long` 64-bit），这时要一致设置 `BLA_SIZEOF_INTEGER=8`，并保证你自己的代码也用 64 位整型来调用 BLAS。([cmake.org][1])

4. **直接链接 vs 通过 FlexiBLAS / libblastrampoline**

   * 直接链接：最少开销，但升级/切换实现需要重链。
   * Wrapper：一点点额外调用开销，换来“运行时切后端”的灵活性，方便发行版/科学平台编排。([GitHub][11])

---

## 六、常见坑点（自检用）

1. **指定了错误的 BLA_VENDOR**

   表现：CMake 报 `Could NOT find BLAS (missing: BLAS_LIBRARIES)`，而你以为装好了。
   自检：

   * 你写的是 `OpenBLAS` 还是 `openblas`？（大小写和拼写必须完全匹配 vendor 名）([cmake.org][1])
   * 发型版里到底有没有装对应 dev 包？（例如 `libopenblas-dev`）

2. **装了库，但没在搜索路径**

   * Windows：忘了把 DLL/库路径加到 PATH / LIB 环境变量。
   * Linux：自己编的 OpenBLAS 装到 `/opt/openblas`，没加 `LD_LIBRARY_PATH` 或 `CMAKE_PREFIX_PATH`。

3. **多线程超订阅**

   * 应用本身开了多线程（比如 ggml 本身用多线程），BLAS 也在用 OpenMP/Threading，CPU 核心被“翻倍占用”，结果性能反而退步。
   * 自检：运行时看 `htop` / `top` 里的线程数，把 `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` 调成合理值。

4. **LP64 / ILP64 不匹配**

   * 你链接了 ILP64 MKL（`Intel10_64ilp`），但自己的代码仍然用 `int`，会导致参数传递错位、内存 corruption。
   * 自检：确认 `BLA_SIZEOF_INTEGER`、MKL 配置和你代码里的 BLAS 接口签名一致。([cmake.org][1])

5. **macOS 上乱混 Accelerate 和外部 BLAS**

   * 部分场景下，如果既链接 Accelerate 又链接 OpenBLAS，可能出现符号重复、行为不确定。
   * 自检：确保最终链接命令里只保留一个主要 BLAS 实现。

---

## 七、检查理解（小练习）

1. 口头解释下面这句话的含义：

   > “在 Ubuntu 上编译 ggml 时，我用 `-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS`。”

   要点包括：

   * 这表示你期望用什么库？
   * CMake 实际上会做什么？
   * 什么时候会失败？

2. 设想三个环境，给出你会选的 `BLA_VENDOR`/`GGML_BLAS_VENDOR`：

   * macOS + M2，不想折腾第三方库；
   * Ubuntu + Ampere Altra（Arm 服务器），已安装 ArmPL；
   * Windows + Intel CPU，有 oneAPI 且只跑你这一个计算程序。

---

## 八、进阶阅读 / 下一步实验

* **CMake FindBLAS 文档**：完整 vendor 列表、`BLA_VENDOR`/`BLA_SIZEOF_INTEGER` 等解释。([cmake.org][1])
* **OpenBLAS 项目与文档**：了解它对不同 CPU 的优化思路。([Wikipedia][5])
* **BLIS 论文 & 框架介绍**：了解一个“现代 BLAS 框架”如何拆内核、做抽象。([cs.utexas.edu][8])
* **FlexiBLAS 与 BLAS/LAPACK 打包文章**：观察大型发行版/科学平台如何用 FlexiBLAS / libblastrampoline 统一管理 BLAS 后端。([GitLab][15])

---

**小练习建议：**
在你现在的开发机上，把 `llama.cpp` 或 ggml 单独拉下来，分别试一试：

* 不设 `GGML_BLAS_VENDOR`，只 `GGML_BLAS=ON`；
* 设成 `OpenBLAS`；
* 如果是 macOS，再设成 `Apple`；

对比 CMake 配置输出和最终 `main` 运行时打印的“using BLAS backend”信息，顺便看一眼 `ldd` / `otool -L` 显示到底链接了哪个库。这样你对“vendor 字符串和实际链接结果的关系”就会有很直观的感觉。


[1]: https://cmake.org/cmake/help/latest/module/FindBLAS.html "FindBLAS — CMake 4.2.0 Documentation"
[2]: https://huggingface.co/spaces/Steven10429/apply_lora_and_quantize/blob/main/llama.cpp/ggml/CMakeLists.txt?utm_source=chatgpt.com "llama.cpp/ggml/CMakeLists.txt · Steven10429/ ..."
[3]: https://www.netlib.org/blas/?utm_source=chatgpt.com "BLAS (Basic Linear Algebra Subprograms)"
[4]: https://developer.apple.com/documentation/accelerate/blas-library?utm_source=chatgpt.com "BLAS | Apple Developer Documentation"
[5]: https://en.wikipedia.org/wiki/OpenBLAS?utm_source=chatgpt.com "OpenBLAS"
[6]: https://hpcc.umd.edu/software/packages/openblas/?utm_source=chatgpt.com "openblas: Optimized BLAS libraries"
[7]: https://github.com/flame/blis?utm_source=chatgpt.com "flame/blis: BLAS-like Library Instantiation Software ..."
[8]: https://www.cs.utexas.edu/users/flame/pubs/BLISTOMSrev2.pdf?utm_source=chatgpt.com "BLIS: A Framework for Rapidly Instantiating BLAS Functionality"
[9]: https://labs.quansight.org/blog/blas-lapack-packaging?utm_source=chatgpt.com "BLAS/LAPACK packaging"
[10]: https://community.arm.com/support-forums/f/high-performance-computing-forum/56375/find_package-blas-fails-to-find-arm-performance-libraries?utm_source=chatgpt.com "find_package(BLAS) fails to find Arm performance libraries"
[11]: https://github.com/mpimd-csc/flexiblas?utm_source=chatgpt.com "GitHub - mpimd-csc/flexiblas: FlexiBLAS - A BLAS and ..."
[12]: https://fossies.org/linux/llama.cpp/ggml/src/ggml-blas/CMakeLists.txt?m=b&utm_source=chatgpt.com "CMakeLists.txt"
[13]: https://github.com/ggerganov/llama.cpp/issues/9039?utm_source=chatgpt.com "OpenBLAS compile for Android doesn't work in Ubuntu ..."
[14]: https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/build.md?utm_source=chatgpt.com "Vulkan - GitHub"
[15]: https://gitlab.mpi-magdeburg.mpg.de/software/flexiblas-release/-/blob/master/README.md?utm_source=chatgpt.com "README.md · master · Software / FlexiBLAS"
