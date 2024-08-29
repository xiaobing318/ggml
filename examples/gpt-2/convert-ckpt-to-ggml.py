# Convert a model checkpoint to a ggml compatible file(将一个model checkpoint转化为ggml兼容性文件格式)
#
# Load the model using TensorFlow.（使用tensorflow加载model）
# Iterate over all variables and write them to a binary file.（迭代所有的变量并将这些变量写入到一个二进制文件中）
#
# For each variable, write the following:（对于每一个变量，写入下列内容）
#   - Number of dimensions (int)（维度的数量）
#   - Name length (int)（变量名称的长度）
#   - Dimensions (int[n_dims])（维度）
#   - Name (char[name_length])（名称）
#   - Data (float[n_dims])（数据）
#
# By default, the bigger matrices are converted to 16-bit floats.（默认情况下，更大的矩阵被转化为16-bit的浮点数）
# This can be disabled by adding the "use-f32" CLI argument.（这个可以通过添加use0f32命令行参数禁止使用）
#
# At the start of the ggml file we write the model parameters（在ggml文件的开始我们写入model的参数和vocabulary）
# and vocabulary.
#

# 导入python的标准库sys，这个库提供与操作系统直接相关的操作，例如操作文件、目录
import sys
# 导入json标准库，这个库提供json格式相关的操作
import json
import struct
# 导入numpy第三方库，提供矩阵相关的内容
import numpy as np
# 导入tensorflow库
import tensorflow as tf

# 引用openai提供的encoder.py相关代码
# ref: https://github.com/openai/gpt-2/blob/master/src/encoder.py
'''
    这个函数 `bytes_to_unicode()` 的主要目的是创建一个从UTF-8字节到Unicode字符串的查找表。这个查找表使得每一个字节都与一个唯一的Unicode字符
相关联，有助于在使用字节对编码（BPE）时避免未知字符（UNK）的产生，特别是在处理大型数据集时。下面是函数具体的工作流程和逻辑：

1. **初始化字节列表（bs）**：函数首先创建一个包含可打印ASCII字符范围（从"!"到"~"）以及扩展的Latin字符（从"¡"到"¬"和从"®"到"ÿ"）的列表。这些
范围主要包含了大部分常用的非控制字符。

2. **复制字节列表到字符列表（cs）**：这个列表是Unicode字符的起始点，初始时与 `bs` 相同。

3. **填充其他字节**：函数遍历所有可能的256个UTF-8字节值（从0到255）。对于不在初始列表 `bs` 中的每个字节，它会添加这个字节到 `bs`，同时在 `cs` 
中添加一个新的Unicode字符。这个新字符是由基础值 `2**8` 加上一个递增的索引 `n` 组成的，确保了每个字节都能被映射到一个唯一的、非控制的Unicode字符。

4. **转换字符编码**：列表 `cs` 中的数字被转换成对应的Unicode字符。

5. **创建查找表**：使用 `zip` 函数将 `bs` 和转换后的 `cs` 组合成一个字典，实现从字节到字符的映射。

    总结来说，这个函数通过创建从字节到特定Unicode字符的映射，帮助避免在大规模文本数据处理中出现字符映射错误，特别是在使用BPE进行文本编码的场景中
非常有用。这种方法可以显著减少在大型词汇表中需要处理的未知字符数量。
'''
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

'''
    这个函数 `convert_to_ftype` 的主要作用是将给定的 numpy 数组转换成指定的浮点类型。目前，该函数支持将数据转换为 `np.float16` 类型，这是一种
半精度浮点格式，常用于减少数据的存储大小和处理时的计算资源需求，尤其在深度学习和高性能计算领域。

函数的工作流程如下：

1. **参数解释**：
   - `data`：一个 numpy 数组，表示要转换的数据。
   - `ftype`：一个标识符，用于指定转换的浮点类型。目前，如果 `ftype` 等于1，数据会被转换为 `np.float16`。

2. **类型转换**：
   - 如果 `ftype` 等于1，函数使用 numpy 的 `astype` 方法将 `data` 数组的数据类型转换为 `np.float16`。

3. **错误处理**：
   - 如果 `ftype` 不等于支持的类型（目前只有1），函数将断言失败，并抛出一个包含错误信息的异常，指出 `ftype` 是无效的。

    总结来说，这个函数的目的是提供一种灵活的方法来根据需要将数据数组转换为不同的浮点数类型。虽然当前只实现了对 `np.float16` 的支持，但函数的结构
允许未来扩展，可以加入更多的浮点类型支持。
'''
# helper method to convert a numpy array to different float types
def convert_to_ftype(data, ftype):
    # fp16
    if ftype == 1:
        return data.astype(np.float16)

    assert False, "Invalid ftype: " + str(ftype)

'''
    这段代码主要用于检查运行Python脚本时是否提供了足够的命令行参数，并给出正确的使用说明。具体来说，代码的作用如下：

1. **参数检查**：
   - 使用 `len(sys.argv)` 来获取命令行参数的数量。`sys.argv` 是一个包含命令行参数的列表，其中 `sys.argv[0]` 是脚本名称，之后的元素是传递给
   脚本的参数。
   - `if len(sys.argv) < 3:` 检查是否至少提供了两个参数（除了脚本名之外）。这意味着用户需要提供两个参数：`dir-model` 和 `ftype`。

2. **使用说明输出**：
   - 如果参数不足，脚本会通过 `print` 函数输出使用说明，提示用户如何正确运行脚本。它指出需要两个参数：模型目录 (`dir-model`) 和浮点类型标识 
   (`ftype`)。
   - `ftype == 0 -> float32` 和 `ftype == 1 -> float16` 为用户提供了 `ftype` 参数的可选值及其对应的数据类型，即 `float32` 或 `float16`。

3. **终止执行**：
   - 使用 `sys.exit(1)` 来终止程序执行。这里传递的参数 `1` 通常表示程序因错误而非正常结束。这是一个常见的做法，用于在发现输入错误时立即停止程
   序运行，并返回错误代码。

    总的来说，这段代码用于确保用户在运行脚本时提供了必要的参数，并在参数缺失时提供清晰的指示，帮助用户正确使用脚本。这样的参数检查是脚本健壮性的
重要组成部分，有助于避免运行时错误和潜在的问题。
'''
if len(sys.argv) < 3:
    print("Usage: convert-ckpt-to-ggml.py dir-model ftype\n")
    print("  ftype == 0 -> float32")
    print("  ftype == 1 -> float16")
    sys.exit(1)

'''
    这段代码的主要作用是从指定目录中读取并加载两个JSON文件，这些文件通常包含模型的编码器配置和超参数设置。以下是代码的具体功能和工作流程：

1. **定义输出文件的目录和名称**：
   - `dir_model = sys.argv[1]`：从命令行参数获取模型的目录路径。这是脚本的第一个参数（`sys.argv[1]`），它指定了包含模型文件的目录。
   - `fname_out = sys.argv[1] + "/ggml-model.bin"`：设置输出文件的完整路径，该文件将位于与模型同一目录下，并命名为 `ggml-model.bin`。

2. **读取编码器配置**：
   - `with open(dir_model + "/encoder.json", "r", encoding="utf-8") as f`：以只读模式打开 `encoder.json` 文件。这个文件通常包含模型的编码器
   配置，如字符到索引的映射等。通过指定 `encoding="utf-8"` 确保文件以UTF-8编码读取，这是处理文本数据的标准方式。
   - `encoder = json.load(f)`：使用 `json.load` 方法从文件中加载JSON数据，将其转换为Python字典，这个字典现在包含编码器的配置信息。

3. **读取模型的超参数**：
   - `with open(dir_model + "/hparams.json", "r", encoding="utf-8") as f`：同样以只读模式打开 `hparams.json` 文件。这个文件包含模型训练和
   操作所需的超参数，如学习率、批次大小等。
   - `hparams = json.load(f)`：使用 `json.load` 方法加载超参数配置，同样将其转换为Python字典以便进一步处理。

    总结来说，这段代码用于加载和处理与机器学习模型相关的关键配置文件。通过读取这些文件，脚本可以获得必要的信息来进一步操作模型，例如进行训练、评估或
应用模型。这种配置文件的使用是机器学习和深度学习项目中常见的实践，有助于模型配置的标准化和复用。
'''
# output in the same directory as the model
dir_model = sys.argv[1]
fname_out = sys.argv[1] + "/ggml-model.bin"

with open(dir_model + "/encoder.json", "r", encoding="utf-8") as f:
    encoder = json.load(f)

with open(dir_model + "/hparams.json", "r", encoding="utf-8") as f:
    hparams = json.load(f)


# 这段代码的主要作用是根据命令行参数确定文件的输出类型和名称，同时确保输入的类型标识（ftype）是有效的
# possible data types
#   ftype == 0 -> float32
#   ftype == 1 -> float16
#
# map from ftype to string
ftype_str = ["f32", "f16"]

ftype = 1
if len(sys.argv) > 2:
    ftype = int(sys.argv[2])
    if ftype < 0 or ftype > 1:
        print("Invalid ftype: " + str(ftype))
        sys.exit(1)
    fname_out = sys.argv[1] + "/ggml-model-" + ftype_str[ftype] + ".bin"


'''

    这段代码的作用是将特定模型的参数和编码器信息写入一个二进制文件中。具体过程涉及了使用TensorFlow读取模型变量、构建输出文件、写入头部信息以及编码器
数据。以下是每个步骤的详细说明：

1. **读取模型变量**：
   - `list_vars = tf.train.list_variables(dir_model)`：这一行代码调用 TensorFlow 的 `list_variables` 函数，列出位于 `dir_model` 指定的目录下
的所有模型变量。

2. **打开输出文件**：
   - `fout = open(fname_out, "wb")`：以二进制写模式打开指定的输出文件 `fname_out`。

3. **写入文件头部信息**：
   - `fout.write(struct.pack("i", 0x67676d6c))`：写入一个魔数（magic number），这是一个用于标识文件类型或格式的特定数字，这里用 "ggml" 的十六
   进制表示。
   - 接下来的几行使用 `struct.pack("i", value)` 来写入超参数，如词汇表大小 (`n_vocab`)、上下文大小 (`n_ctx`)、嵌入维度 (`n_embd`)、头的数量 
   (`n_head`)、层数 (`n_layer`) 和浮点类型 (`ftype`)。这些参数对于模型的配置非常关键。

4. **处理和写入编码器数据**：
   - `byte_encoder = bytes_to_unicode()`：调用前面定义的函数，创建一个从字节到Unicode字符的映射。
   - `byte_decoder = {v: k for k, v in byte_encoder.items()}`：生成逆映射，用于将Unicode字符映射回原始字节。
   - `fout.write(struct.pack("i", len(encoder)))`：写入编码器的长度，即编码器中条目的数量。
   - 循环遍历编码器中的每个键（`key`）：
     - 将键字符串中的每个字符转换为对应的字节，使用 `byte_decoder` 完成这一转换，并构造一个 `bytearray`。
     - 首先写入这个 `bytearray` 的长度，然后写入 `bytearray` 本身。

5. **文件写入完成**：
   - 循环结束后，所有的编码器键及其对应的字节数据都已写入文件。最后，文件通过 `fout.close()` （如果使用上下文管理器自动处理则不需要显式调用）关闭，
确保所有数据都正确保存并完成资源的释放。

    总结来说，这段代码的目的是为了将模型的配置参数和编码信息以一种特定的格式（二进制）保存到文件中，这对于模型的部署和使用，特别是在需要加载预训练模型
到不同环境中时非常有用。
'''
# 使用tensorflow中的相关的函数将输入的model中的variables罗列出来
list_vars = tf.train.list_variables(dir_model)
# 打开文件，将会向这个文件中写入二进制文件流
fout = open(fname_out, "wb")

fout.write(struct.pack("i", 0x67676d6c)) # magic: ggml in hex
fout.write(struct.pack("i", hparams["n_vocab"]))
fout.write(struct.pack("i", hparams["n_ctx"]))
fout.write(struct.pack("i", hparams["n_embd"]))
fout.write(struct.pack("i", hparams["n_head"]))
fout.write(struct.pack("i", hparams["n_layer"]))
fout.write(struct.pack("i", ftype))

byte_encoder = bytes_to_unicode()
byte_decoder = {v:k for k, v in byte_encoder.items()}

fout.write(struct.pack("i", len(encoder)))

for key in encoder:
    text = bytearray([byte_decoder[c] for c in key])
    fout.write(struct.pack("i", len(text)))
    fout.write(text)

for name, shape in list_vars:
    print("Processing variable: " + name + " with shape: ", shape)

    data = tf.train.load_variable(dir_model, name).squeeze()
    n_dims = len(data.shape);

    # for efficiency - transpose the projection matrices
    # "model/h.*/attn/c_attn/w"
    # "model/h.*/attn/c_proj/w"
    # "model/h.*/mlp/c_fc/w"
    # "model/h.*/mlp/c_proj/w"
    if name[-14:] == "/attn/c_attn/w" or \
       name[-14:] == "/attn/c_proj/w" or \
       name[-11:] == "/mlp/c_fc/w" or \
       name[-13:] == "/mlp/c_proj/w":
        print("  Transposing")
        data = data.transpose()

    dshape = data.shape

    ftype_cur = 0
    if ftype != 0:
        # match name:
        #  "model/wte"
        #  "model/h.*/attn/c_attn/w"
        #  "model/h.*/attn/c_proj/w"
        #  "model/h.*/mlp/c_fc/w"
        #  "model/h.*/mlp/c_proj/w"
        if name == "model/wte" or name[-2:] == "/w":
            print("  Converting to " + ftype_str[ftype])
            data = convert_to_ftype(data, ftype)
            ftype_cur = ftype
        else:
            print("  Converting to float32")
            data = data.astype(np.float32)
            ftype_cur = 0

    # header
    str = name.encode('utf-8')
    fout.write(struct.pack("iii", n_dims, len(str), ftype_cur))
    for i in range(n_dims):
        fout.write(struct.pack("i", dshape[n_dims - 1 - i]))
    fout.write(str);

    # data
    data.tofile(fout)

fout.close()

print("Done. Output file: " + fname_out)
print("")
