# GGUF（GGUF是一种文件格式）

GGUF is a file format for storing models for inference with GGML and executors based on GGML. GGUF is a binary format that is designed for fast loading and saving of models, and for ease of reading. Models are traditionally developed using PyTorch or another framework, and then converted to GGUF for use in GGML.（GGUF 是一种文件格式，用于存储使用 GGML 进行推理的模型以及基于 GGML 的执行器。GGUF 是一种二进制格式，旨在快速加载和保存模型，并且易于读取。传统上，模型是使用 PyTorch 或其他框架开发的，然后转换为 GGUF 以在 GGML 中使用。）

 - GGUF是一种文件格式，这种文件内部存储的是可以被GGML进行推理的模型文件
 - GGUF是一种二进制格式，GGUF为了能够快速加载和保存模型
 - 一般而言模型都是使用PyTorch或者其他框架开发的，这些模型转化成GGUF后再GGML中使用

It is a successor file format to GGML, GGMF and GGJT, and is designed to be unambiguous by containing all the information needed to load a model. It is also designed to be extensible, so that new information can be added to models without breaking compatibility.（它是 GGML、GGMF 和 GGJT 的后继文件格式，旨在通过包含加载模型所需的所有信息来确保明确性。它还具有可扩展性，因此可以在不破坏兼容性的情况下将新信息添加到模型中。）

 - GGUF是GGML、GGMF、GGJT文件格式的后继者
 - GGUF包含加载模型所需要的所有信息来确保明确性，为了能够运行整个模型，加载一个GGUF文件就可以了，因为在这个GGUF文件中包含了所有的信息

For more information about the motivation behind GGUF, see [Historical State of Affairs](#historical-state-of-affairs).（有关 GGUF 背后动机的更多信息，请参阅[历史事态](#historical-state-of-affairs)。）





















## Specification（规范）

GGUF is a format based on the existing GGJT, but makes a few changes to the format to make it more extensible and easier to use. The following features are desired（GGUF 是一种基于现有 GGJT 的格式，但对该格式进行了一些更改，使其更具可扩展性和更易于使用。需要以下功能）:

- Single-file deployment: they can be easily distributed and loaded, and do not require any external files for additional information.（单文件部署：它们可以轻松分发和加载，并且不需要任何外部文件来获取附加信息。）
- Extensible: new features can be added to GGML-based executors/new information can be added to GGUF models without breaking compatibility with existing models.（可扩展：可以向基于 GGML 的执行器添加新功能/可以向 GGUF 模型添加新信息，而不会破坏与现有模型的兼容性。）
- `mmap` compatibility: models can be loaded using `mmap` for fast loading and saving.（`mmap` 兼容性：可以使用 `mmap` 加载模型，以便快速加载和保存）
- Easy to use: models can be easily loaded and saved using a small amount of code, with no need for external libraries, regardless of the language used.（易于使用：无论使用何种语言，都可以使用少量代码轻松加载和保存模型，无需外部库。）
- Full information: all information needed to load a model is contained in the model file, and no additional information needs to be provided by the user.（信息完整：加载模型所需的所有信息都包含在模型文件中，无需用户提供任何额外信息。）

The key difference between GGJT and GGUF is the use of a key-value structure for the hyperparameters (now referred to as metadata), rather than a list of untyped values. This allows for new metadata to be added without breaking compatibility with existing models, and to annotate the model with additional information that may be useful for inference or for identifying the model.（GGJT 和 GGUF 之间的主要区别在于，GGJT 使用键值结构来表示超参数（现在称为元数据），而不是无类型值列表。这样就可以在不破坏与现有模型兼容性的情况下添加新的元数据，并使用可能对推理或识别模型有用的其他信息来注释模型。）


GGUF（Generic Graph Universal Format）是基于现有的GGJT格式开发的，旨在通过一些关键的改进使得格式更加可扩展和易于使用。以下是对GGUF文件格式和其主要特性的详细解释：

### 特点一：单文件部署

GGUF设计为单文件格式，这意味着整个模型的所有必要信息都被包含在一个文件内：
- **便于分发和加载**：单文件使得模型的部署、传输和加载变得更加简单直接，因为不需要管理多个依赖文件。
- **独立性**：模型不依赖任何外部文件来补充信息，降低了部署和维护的复杂性。

### 特点二：可扩展性

GGUF格式特别强调可扩展性：
- **向前兼容**：可以在不破坏与现有模型兼容性的前提下，向模型文件中添加新的信息或特性。
- **适应性**：这种设计支持随着机器学习技术的发展，逐步引入新的算法特性或性能优化措施。

### 特点三：`mmap` 兼容性

GGUF支持通过内存映射（`mmap`）的方式加载模型：
- **快速加载和保存**：`mmap` 允许应用程序以接近内存速度的方式直接读写磁盘上的文件，这对于大型模型的加载和保存非常有效率。
- **低内存占用**：使用`mmap`加载模型可以避免将整个模型载入RAM，从而节省内存资源。

### 特点四：易用性

GGUF设计考虑到跨语言的易用性：
- **简化API**：无论使用哪种编程语言，都能够通过少量的代码实现模型的加载和保存，无需依赖外部库。
- **通用性**：设计API时注重简洁性和通用性，使得不同编程环境中的开发者都能轻松使用。

### 特点五：完整信息

GGUF文件包含加载模型所需的全部信息：
- **无需额外输入**：使用者在加载模型时无需提供额外的参数或配置，所有必要的信息都已经嵌入在文件中。
- **模型自描述**：文件格式支持包含丰富的元数据，这些元数据描述了模型的配置、训练过程以及可能需要的任何执行信息。

### 特点六：元数据的关键变化

与GGJT格式相比，GGUF在处理超参数（现称为元数据）时采用键值对结构，而非无类型的值列表：
- **灵活的元数据扩展**：键值对结构使得添加新的元数据更加灵活，且不会影响现有模型的兼容性。
- **增强的自描述能力**：键值对允许在模型文件中注释更多有用的信息，这对于推理过程或模型识别特别有价值。

通过这些设计改进，GGUF文件格式提供了一个强大的、适用于现代机器学习应用的模型存储和部署解决方案，特别注重性能、扩展性和用户体验。




















### GGUF Naming Convention（GGUF 命名约定）

GGUF follow a naming convention of `<BaseName><SizeLabel><FineTune><Version><Encoding><Type><Shard>.gguf` where each component is delimitated by a `-` if present. Ultimately this is intended to make it easier for humans to at a glance get the most important details of a model. It is not intended to be perfectly parsable in the field due to the diversity of existing gguf filenames.

 - GGUF 遵循命名约定 `<BaseName><SizeLabel><FineTune><Version><Encoding><Type><Shard>.gguf`，其中每个组件如果存在则用 `-` 分隔。最终，这是为了让人们更容易一目了然地了解模型的最重要细节。由于现有 gguf 文件名的多样性，它并不打算在现场完美解析。
 - 这一命名格式旨在便于人类阅读和理解，而不是为了在现场完美解析，因为现有的gguf文件名具有多样性。


The components are（每个组件部分的解释）:
1. **BaseName**: A descriptive name for the model base type or architecture.
    - This can be derived from gguf metadata `general.basename` substituting spaces for dashes.
2. **SizeLabel**: Parameter weight class (useful for leader boards) represented as `<expertCount>x<count><scale-prefix>`
    - This can be derived from gguf metadata `general.size_label` if available or calculated if missing.
    - Rounded decimal point is supported in count with a single letter scale prefix to assist in floating point exponent shown below
      - `Q`: Quadrillion parameters.
      - `T`: Trillion parameters.
      - `B`: Billion parameters.
      - `M`: Million parameters.
      - `K`: Thousand parameters.
    - Additional `-<attributes><count><scale-prefix>` can be appended as needed to indicate other attributes of interest
3. **FineTune**: A descriptive name for the model fine tuning goal (e.g. Chat, Instruct, etc...)
    - This can be derived from gguf metadata `general.finetune` substituting spaces for dashes.
4. **Version**: (Optional) Denotes the model version number, formatted as `v<Major>.<Minor>`
    - If model is missing a version number then assume `v1.0` (First Public Release)
    - This can be derived from gguf metadata `general.version`
5. **Encoding**: Indicates the weights encoding scheme that was applied to the model. Content, type mixture and arrangement however are determined by user code and can vary depending on project needs.
6. **Type**: Indicates the kind of gguf file and the intended purpose for it
     - If missing, then file is by default a typical gguf tensor model file
     - `LoRA` : GGUF file is a LoRA adapter
     - `vocab` : GGUF file with only vocab data and metadata
7. **Shard**: (Optional) Indicates and denotes that the model has been split into multiple shards, formatted as `<ShardNum>-of-<ShardTotal>`.
    - *ShardNum* : Shard position in this model. Must be 5 digits padded by zeros.
      - Shard number always starts from `00001` onwards (e.g. First shard always starts at `00001-of-XXXXX` rather than `00000-of-XXXXX`).
    - *ShardTotal* : Total number of shards in this model. Must be 5 digits padded by zeros.
  

### 命名组成部分

1. **BaseName（基础名称）**:
   - 描述模型基础类型或架构的名称。
   - 可以从gguf元数据的`general.basename`字段获取，将空格替换为破折号。

2. **SizeLabel（尺寸标签）**:
   - 参数权重等级，格式为`<expertCount>x<count><scale-prefix>`，用于排行榜等。
   - 可以从gguf元数据的`general.size_label`获取，如果缺失则需计算。
   - 支持使用单字母的规模前缀来帮助显示浮点数的指数：
     - `Q`：千兆（Quadrillion）参数。
     - `T`：万亿（Trillion）参数。
     - `B`：十亿（Billion）参数。
     - `M`：百万（Million）参数。
     - `K`：千（Thousand）参数。
   - 如有需要，可以追加其他属性，格式为`-<attributes><count><scale-prefix>`。

3. **FineTune（微调目标）**:
   - 描述模型微调目的的名称（例如 Chat, Instruct 等）。
   - 可以从gguf元数据的`general.finetune`获取，将空格替换为破折号。

4. **Version（版本）**:
   - （可选）表示模型的版本号，格式为`v<Major>.<Minor>`。
   - 如果模型缺少版本号，则默认为`v1.0`（首次公开发布）。
   - 可以从gguf元数据的`general.version`获取。

5. **Encoding（编码）**:
   - 表示应用于模型的权重编码方案。
   - 具体内容、类型和排列方式根据用户代码的不同而有所不同，视项目需求而定。

6. **Type（类型）**:
   - 指示gguf文件的种类及其预期用途。
   - 如果未指明，则默认为典型的gguf张量模型文件。
   - `LoRA`：GGUF文件是LoRA适配器。
   - `vocab`：GGUF文件仅包含词汇数据和元数据。

7. **Shard（分片）**:
   - （可选）表示模型已被分割成多个分片，格式为`<ShardNum>-of-<ShardTotal>`。
   - *ShardNum*：该模型中的分片位置，必须是五位数且由零填充。
     - 分片编号始终从`00001`开始（例如，第一个分片始终为`00001-of-XXXXX`，而非`00000-of-XXXXX`）。
   - *ShardTotal*：该模型的分片总数，必须是五位数且由零填充。

### 总结

这种命名约定的设计是为了使模型的关键信息（如模型的类型、尺寸、微调目的、版本和编码方案）对于维护者和用户来说更加清晰和易于理解。通过这样的方式，即使在文件名多样化的情况下，也能快速识别和操作这些模型文件。











#### Validating Above Naming Convention（验证上述命名约定）

At a minimum all model files should have at least BaseName, SizeLabel, Version, in order to be easily validated as a file that is keeping with the GGUF Naming Convention. An example of this issue is that it is easy for Encoding to be mistaken as a FineTune if Version is omitted.（所有模型文件至少应具有 BaseName、SizeLabel 和 Version，以便轻松验证文件是否符合 GGUF 命名约定。此问题的一个例子是，如果省略 Version，则很容易将 Encoding 误认为 FineTune。）

To validate you can use this regular expression `^(?<BaseName>[A-Za-z0-9\s]*(?:(?:-(?:(?:[A-Za-z\s][A-Za-z0-9\s]*)|(?:[0-9\s]*)))*))-(?:(?<SizeLabel>(?:\d+x)?(?:\d+\.)?\d+[A-Za-z](?:-[A-Za-z]+(\d+\.)?\d+[A-Za-z]+)?)(?:-(?<FineTune>[A-Za-z0-9\s-]+))?)?-(?:(?<Version>v\d+(?:\.\d+)*))(?:-(?<Encoding>(?!LoRA|vocab)[\w_]+))?(?:-(?<Type>LoRA|vocab))?(?:-(?<Shard>\d{5}-of-\d{5}))?\.gguf$` which will check that you got the minimum BaseName, SizeLabel and Version present in the correct order.

For example:

  * `Mixtral-8x7B-v0.1-KQ2.gguf`:
    - Model Name: Mixtral
    - Expert Count: 8
    - Parameter Count: 7B
    - Version Number: v0.1
    - Weight Encoding Scheme: KQ2

  * `Hermes-2-Pro-Llama-3-8B-F16.gguf`:
    - Model Name: Hermes 2 Pro Llama 3
    - Expert Count: 0
    - Parameter Count: 8B
    - Version Number: v1.0
    - Weight Encoding Scheme: F16
    - Shard: N/A

  * `Grok-100B-v1.0-Q4_0-00003-of-00009.gguf`
    - Model Name: Grok
    - Expert Count: 0
    - Parameter Count: 100B
    - Version Number: v1.0
    - Weight Encoding Scheme: Q4_0
    - Shard: 3 out of 9 total shards


<details><summary>Example Node.js Regex Function</summary>

```js
#!/usr/bin/env node
const ggufRegex = /^(?<BaseName>[A-Za-z0-9\s]*(?:(?:-(?:(?:[A-Za-z\s][A-Za-z0-9\s]*)|(?:[0-9\s]*)))*))-(?:(?<SizeLabel>(?:\d+x)?(?:\d+\.)?\d+[A-Za-z](?:-[A-Za-z]+(\d+\.)?\d+[A-Za-z]+)?)(?:-(?<FineTune>[A-Za-z0-9\s-]+))?)?-(?:(?<Version>v\d+(?:\.\d+)*))(?:-(?<Encoding>(?!LoRA|vocab)[\w_]+))?(?:-(?<Type>LoRA|vocab))?(?:-(?<Shard>\d{5}-of-\d{5}))?\.gguf$/;

function parseGGUFFilename(filename) {
  const match = ggufRegex.exec(filename);
  if (!match)
    return null;
  const {BaseName = null, SizeLabel = null, FineTune = null, Version = "v1.0", Encoding = null, Type = null, Shard = null} = match.groups;
  return {BaseName: BaseName, SizeLabel: SizeLabel, FineTune: FineTune, Version: Version, Encoding: Encoding, Type: Type, Shard: Shard};
}

const testCases = [
  {filename: 'Mixtral-8x7B-v0.1-KQ2.gguf',                         expected: { BaseName: 'Mixtral',              SizeLabel: '8x7B',     FineTune: null, Version: 'v0.1',   Encoding: 'KQ2',  Type: null, Shard: null}},
  {filename: 'Grok-100B-v1.0-Q4_0-00003-of-00009.gguf',            expected: { BaseName: 'Grok',                 SizeLabel: '100B',     FineTune: null, Version: 'v1.0',   Encoding: 'Q4_0', Type: null, Shard: "00003-of-00009"}},
  {filename: 'Hermes-2-Pro-Llama-3-8B-v1.0-F16.gguf',              expected: { BaseName: 'Hermes-2-Pro-Llama-3', SizeLabel: '8B', FineTune: null, Version: 'v1.0',   Encoding: 'F16',  Type: null, Shard: null}},
  {filename: 'Phi-3-mini-3.8B-ContextLength4k-instruct-v1.0.gguf', expected: { BaseName: 'Phi-3-mini',   SizeLabel: '3.8B-ContextLength4k', FineTune: 'instruct', Version: 'v1.0',   Encoding: null,  Type: null, Shard: null}},
  {filename: 'not-a-known-arrangement.gguf',                       expected: null},
];

testCases.forEach(({ filename, expected }) => {
  const result = parseGGUFFilename(filename);
  const passed = JSON.stringify(result) === JSON.stringify(expected);
  console.log(`${filename}: ${passed ? "PASS" : "FAIL"}`);
  if (!passed) {
      console.log(result);
      console.log(expected);
  }
});
```

</details>

上述内容描述了GGUF文件命名约定的进一步规范化，以及通过正则表达式验证GGUF文件名是否符合规定格式的方法。这个命名约定包含多个组件，以确保文件名可以提供关于模型的重要信息，同时规范化的格式有助于自动化处理和管理模型文件。以下是各部分的详细解释：

### 文件命名组成部分

1. **BaseName**（基本名称）: 模型的基础类型或架构的描述性名称。
2. **SizeLabel**（尺寸标签）: 描述模型参数量的标签，可能包括专家数（如果适用）和参数规模的缩写（如M表示百万）。
3. **FineTune**（微调目标）: 描述模型微调目标的名称，如`Chat`、`Instruct`等。
4. **Version**（版本）: 模型的版本号，标准格式为`vX.X`。
5. **Encoding**（编码方式）: 指示应用于模型的权重编码方案。
6. **Type**（类型）: 表示文件类型，例如是否为LoRA适配器或仅包含词汇数据和元数据的文件。
7. **Shard**（分片）: 如果模型被分割成多个文件，此部分标识分片的序号和总数。

### 正则表达式的使用

提供的正则表达式用于验证GGUF文件名是否包含至少包括BaseName、SizeLabel和Version的必要信息，并且这些信息是否按正确的顺序排列。正则表达式的各组成部分如下：

- `BaseName`: 匹配以字母、数字开始，可能包含破折号分隔的多个词组。
- `SizeLabel`: 匹配表示参数数量的标签，可能包含专家数前缀和参数规模后缀。
- `FineTune`: 可选组件，匹配模型的具体微调目标。
- `Version`: 匹配模型版本号，这是必须存在的组件。
- `Encoding`: 可选组件，匹配权重编码方案。
- `Type`: 可选组件，匹配文件类型标记。
- `Shard`: 可选组件，匹配模型的分片信息。

### 示例和Node.js脚本

提供了一些文件名示例来说明如何应用这个命名约定。同时，附带了一个Node.js脚本示例，该脚本使用上述正则表达式来验证文件名是否符合规定的格式。脚本定义了一个`parseGGUFFilename`函数，该函数使用正则表达式解析文件名，并返回一个包含文件名各部分信息的对象。通过测试案例，可以检查函数是否能正确解析符合和不符合规范的文件名。

这种命名和验证方法的设计旨在确保GGUF文件名能清晰、一致地提供关于模型的关键信息，同时简化文件管理和自动化处理的复杂性。




















### File Structure（GGUF文件结构）

![image](https://github.com/ggerganov/ggml/assets/1991296/c3623641-3a1d-408e-bfaf-1b7c4e16aa63)
*diagram by [@mishig25](https://github.com/mishig25) (GGUF v3)*

GGUF files are structured as follows. They use a global alignment specified in the `general.alignment` metadata field, referred to as `ALIGNMENT` below. Where required, the file is padded with `0x00` bytes to the next multiple of `general.alignment`.（GGUF 文件的结构如下。它们使用 `general.alignment` 元数据字段中指定的全局对齐，下面称为 `ALIGNMENT`。如果需要，文件将用 `0x00` 字节填充到 `general.alignment` 的下一个倍数。）



Fields, including arrays, are written sequentially without alignment unless otherwise specified.（除非另有说明，字段（包括数组）均按顺序写入且不对齐。）

Models are little-endian by default. They can also come in big-endian for use with big-endian computers; in this case, all values (including metadata values and tensors) will also be big-endian. At the time of writing, there is no way to determine if a model is big-endian; this may be rectified in future versions. If no additional information is provided, assume the model is little-endian.（默认情况下，模型是小端的。它们也可以采用大端，以便在大端计算机中使用；在这种情况下，所有值（包括元数据值和张量）也将是大端的。在撰写本文时，无法确定模型是否是大端的；这可能会在未来版本中得到纠正。如果没有提供其他信息，则假设模型是小端的。）

```c
enum ggml_type: uint32_t {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    // GGML_TYPE_Q4_2 = 4, support has been removed
    // GGML_TYPE_Q4_3 = 5, support has been removed
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_COUNT,
};

enum gguf_metadata_value_type: uint32_t {
    // The value is a 8-bit unsigned integer.
    GGUF_METADATA_VALUE_TYPE_UINT8 = 0,
    // The value is a 8-bit signed integer.
    GGUF_METADATA_VALUE_TYPE_INT8 = 1,
    // The value is a 16-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT16 = 2,
    // The value is a 16-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT16 = 3,
    // The value is a 32-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT32 = 4,
    // The value is a 32-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT32 = 5,
    // The value is a 32-bit IEEE754 floating point number.
    GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6,
    // The value is a boolean.
    // 1-byte value where 0 is false and 1 is true.
    // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.（其他任何值均无效，应视为模型无效或读取器存在错误）
    GGUF_METADATA_VALUE_TYPE_BOOL = 7,
    // The value is a UTF-8 non-null-terminated string, with length prepended.
    GGUF_METADATA_VALUE_TYPE_STRING = 8,
    // The value is an array of other values, with the length and type prepended.（该值是其他值的数组，其长度和类型在前面。）
    ///
    // Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.（数组可以嵌套，并且数组的长度是数组元素个数，而不是字节数）
    GGUF_METADATA_VALUE_TYPE_ARRAY = 9,
    // The value is a 64-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT64 = 10,
    // The value is a 64-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT64 = 11,
    // The value is a 64-bit IEEE754 floating point number.
    GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12,
};

// A string in GGUF.
struct gguf_string_t {
    // The length of the string, in bytes.
    uint64_t len;
    // The string as a UTF-8 non-null-terminated string.（字符串数据以UTF-8编码方式存储，并且不以空字符（null-terminated，即'\0'）结尾。UTF-8是一种广泛使用的字符编码方式，可以表示全世界绝大多数的书写系统。不使用空字符结束可以让字符串包含任何二进制数据，包括中间可能出现的零值，这在某些应用场景中非常有用）
    char string[len];
};

union gguf_metadata_value_t {
    uint8_t uint8;
    int8_t int8;
    uint16_t uint16;
    int16_t int16;
    uint32_t uint32;
    int32_t int32;
    float float32;
    uint64_t uint64;
    int64_t int64;
    double float64;
    bool bool_;
    gguf_string_t string;
    struct {
        // Any value type is valid, including arrays.
        gguf_metadata_value_type type;
        // Number of elements, not bytes
        uint64_t len;
        // The array of values.
        gguf_metadata_value_t array[len];
    } array;
};

struct gguf_metadata_kv_t {
    // The key of the metadata. It is a standard GGUF string, with the following caveats（元数据的键。它是一个标准的 GGUF 字符串，但有以下注意事项）:
    // - It must be a valid ASCII string.（它必须是有效的 ASCII 字符串）
    // - It must be a hierarchical key, where each segment is `lower_snake_case` and separated by a `.`.（它必须是一个分层键，其中每个段都是“小写蛇形字母”，并由“.”分隔。）
    // - It must be at most 2^16-1/65535 bytes long.（长度最多为 2^16-1/65535 字节）
    // Any keys that do not follow these rules are invalid.（任何不遵守这些规则的key均无效）
    gguf_string_t key;

    // The type of the value.
    // Must be one of the `gguf_metadata_value_type` values.
    gguf_metadata_value_type value_type;
    // The value.
    gguf_metadata_value_t value;
};

struct gguf_header_t {
    // Magic number to announce that this is a GGUF file.
    // Must be `GGUF` at the byte level: `0x47` `0x47` `0x55` `0x46`.
    // Your executor might do little-endian byte order, so it might be
    // check for 0x46554747 and letting the endianness cancel out.
    // Consider being *very* explicit about the byte order here.
    uint32_t magic;
    // The version of the format implemented.
    // Must be `3` for version described in this spec, which introduces big-endian support.
    //
    // This version should only be increased for structural changes to the format.
    // Changes that do not affect the structure of the file should instead update the metadata
    // to signify the change.
    uint32_t version;
    // The number of tensors in the file.
    // This is explicit, instead of being included in the metadata, to ensure it is always present
    // for loading the tensors.
    uint64_t tensor_count;
    // The number of metadata key-value pairs.
    uint64_t metadata_kv_count;
    // The metadata key-value pairs.
    gguf_metadata_kv_t metadata_kv[metadata_kv_count];
};

uint64_t align_offset(uint64_t offset) {
    return offset + (ALIGNMENT - (offset % ALIGNMENT)) % ALIGNMENT;
}

struct gguf_tensor_info_t {
    // The name of the tensor. It is a standard GGUF string, with the caveat that. it must be at most 64 bytes long.(张量的名称。它是一个标准的 GGUF 字符串，但需要注意的是这个字符串长度最多为64 bytes)
    gguf_string_t name;
    // The number of dimensions in the tensor. Currently at most 4, but this may change in the future.(张量的维数。目前最多为 4，但将来可能会发生变化。)
    uint32_t n_dimensions;
    // The dimensions of the tensor.（tensor的维度）
    uint64_t dimensions[n_dimensions];
    // The type of the tensor.（tensor的类型）
    ggml_type type;
    // The offset of the tensor's data in this file in bytes.This offset is relative to `tensor_data`, not to the start of the file, to make it easier for writers to write the file. Readers should consider exposing this offset relative to the file to make it easier to read the data. Must be a multiple of `ALIGNMENT`. That is, `align_offset(offset) == offset`.（此文件中张量数据的偏移量（以字节为单位）。此偏移量相对于 `tensor_data`，而不是相对于文件开头，以便于编写者写入文件。读者应考虑公开相对于文件的偏移量，以便于读取数据。必须是 `ALIGNMENT` 的倍数。即 `align_offset(offset) == offset`。）
    uint64_t offset;
};

struct gguf_file_t {
    // The header of the file.
    gguf_header_t header;

    // Tensor infos, which can be used to locate the tensor data.
    gguf_tensor_info_t tensor_infos[header.tensor_count];

    // Padding to the nearest multiple of `ALIGNMENT`.
    //
    // That is, if `sizeof(header) + sizeof(tensor_infos)` is not a multiple of `ALIGNMENT`,
    // this padding is added to make it so.
    //
    // This can be calculated as `align_offset(position) - position`, where `position` is
    // the position of the end of `tensor_infos` (i.e. `sizeof(header) + sizeof(tensor_infos)`).
    uint8_t _padding[];

    // Tensor data.
    //
    // This is arbitrary binary data corresponding to the weights of the model. This data should be close
    // or identical to the data in the original model file, but may be different due to quantization or
    // other optimizations for inference. Any such deviations should be recorded in the metadata or as
    // part of the architecture definition.
    //
    // Each tensor's data must be stored within this array, and located through its `tensor_infos` entry.
    // The offset of each tensor's data must be a multiple of `ALIGNMENT`, and the space between tensors
    // should be padded to `ALIGNMENT` bytes.
    uint8_t tensor_data[];
};
```







## Standardized key-value pairs（标准化后的key-value对）

The following key-value pairs are standardized. This list may grow in the future as more use cases are discovered. Where possible, names are shared with the original model definitions to make it easier to map between the two.（以下键值对是标准化的。随着更多用例的发现，此列表将来可能会增长。在可能的情况下，名称与原始模型定义共享，以便更轻松地在两者之间进行映射。）

Not all of these are required, but they are all recommended. Keys that are required are bolded. For omitted pairs, the reader should assume that the value is unknown and either default or error as appropriate.（并非所有这些都是必需的，但它们都是推荐的。必需的键以粗体显示。对于省略的对，读者应假设该值是未知的，并且是默认值或错误（视情况而定）。）

The community can develop their own key-value pairs to carry additional data. However, these should be namespaced with the relevant community name to avoid collisions. For example, the `rustformers` community might use `rustformers.` as a prefix for all of their keys.（社区可以开发自己的键值对来承载更多数据。但是，这些键值对应使用相关社区名称进行命名空间划分，以避免冲突。例如，“rustformers”社区可能会使用“rustformers.”作为其所有键的前缀。）

If a particular community key is widely used, it may be promoted to a standardized key.（如果某个特定社区密钥被广泛使用，则它可能会被提升为标准化密钥。）

By convention, most counts/lengths/etc are `uint64` unless otherwise specified. This is to allow for larger models to be supported in the future. Some models may use `uint32` for their values; it is recommended that readers support both.（按照惯例，除非另有说明，大多数计数/长度等都是“uint64”。这是为了让将来支持更大的模型。某些模型可能使用“uint32”作为其值；建议读者同时支持两者。）

### General

#### Required（必须存在的）

- **`general.architecture: string`**: describes what architecture this model implements. All lowercase ASCII, with only `[a-z0-9]+` characters allowed. Known values include（描述此模型实现的架构。全部小写 ASCII，仅允许使用 `[a-z0-9]+` 字符。已知值包括）:
  - `llama`
  - `mpt`
  - `gptneox`
  - `gptj`
  - `gpt2`
  - `bloom`
  - `falcon`
  - `mamba`
  - `rwkv`
- **`general.quantization_version: uint32`**: The version of the quantization format. Not required if the model is not quantized (i.e. no tensors are quantized). If any tensors are quantized, this _must_ be present. This is separate to the quantization scheme of the tensors itself; the quantization version may change without changing the scheme's name (e.g. the quantization scheme is Q5_K, and the quantization version is 4).（量化格式的版本。如果模型未量化（即没有量化张量），则不需要。如果量化了任何张量，则必须存在此版本。这与张量本身的量化方案无关；量化版本可能会更改，但方案名称不会更改（例如，量化方案为 Q5_K，量化版本为 4）。）
- **`general.alignment: uint32`**: the global alignment to use, as described above. This can vary to allow for different alignment schemes, but it must be a multiple of 8. Some writers may not write the alignment. If the alignment is **not** specified, assume it is `32`.（要使用的全局对齐，如上所述。这可以变化以允许不同的对齐方案，但它必须是 8 的倍数。有些作者可能不会写出对齐。如果未指定对齐，则假定它是“32”。）

#### General metadat）

- `general.name: string`: The name of the model. This should be a human-readable name that can be used to identify the model. It should be unique within the community that the model is defined in.（模型的名称。这应该是人类可读的名称，可用于识别模型。它在定义该模型的社区内应该是唯一的。）
- `general.author: string`: The author of the model.（模型的作者）
- `general.version: string`: The version of the model.（模型的版本）
- `general.organization: string`: The organization of the model.（模型属于的组织）
- `general.basename: string`: The base model name / architecture of the model（基础模型名称/模型架构）
- `general.finetune: string`: What has the base model been optimized toward.（基础模型针对什么进行了优化？）
- `general.description: string`: free-form description of the model including anything that isn't covered by the other fields（模型的自由形式描述，包括其他字段未涵盖的任何内容）
- `general.quantized_by: string`: The name of the individual who quantized the model（量化模型的个人姓名）
- `general.size_label: string`: Size class of the model, such as number of weights and experts. (Useful for leader boards)（模型的大小类别，例如权重和专家的数量。（对排行榜有用））
- `general.license: string`: License of the model, expressed as a [SPDX license expression](https://spdx.github.io/spdx-spec/v2-draft/SPDX-license-expressions/) (e.g. `"MIT OR Apache-2.0`). Do not include any other information, such as the license text or the URL to the license.（模型的许可证，以 [SPDX 许可证表达式](https://spdx.github.io/spdx-spec/v2-draft/SPDX-license-expressions/) 表示（例如“MIT OR Apache-2.0”）。请勿包含任何其他信息，例如许可证文本或许可证的 URL。）
- `general.license.name: string`: Human friendly license name（人性化的许可证名称）
- `general.license.link: string`: URL to the license.（许可证的 URL）
- `general.url: string`: URL to the model's homepage. This can be a GitHub repo, a paper, etc.（模型主页的 URL。可以是 GitHub repo、论文等）
- `general.doi: string`: Digital Object Identifier (DOI) https://www.doi.org/（数字对象标识符 (DOI) https://www.doi.org/）
- `general.uuid: string`: [Universally unique identifier](https://en.wikipedia.org/wiki/Universally_unique_identifier)（[通用唯一标识符]（https://en.wikipedia.org/wiki/Universally_unique_identifier））
- `general.repo_url: string`: URL to the model's repository such as a GitHub repo or HuggingFace repo（模型存储库的 URL，例如 GitHub repo 或 HuggingFace repo）
- `general.tags: string[]`: List of tags that can be used as search terms for a search engine or social media（可用作搜索引擎或社交媒体搜索词的标签列表）
- `general.languages: string[]`: What languages can the model speak. Encoded as [ISO 639](https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes) two letter codes（模型可以使用哪些语言。编码为 [ISO 639](https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes) 两个字母的代码）
- `general.datasets: string[]`: Links or references to datasets that the model was trained upon（模型训练所依据的数据集的链接或引用）
- `general.file_type: uint32`: An enumerated value describing the type of the majority of the tensors in the file. Optional; can be inferred from the tensor types.（描述文件中大多数张量的类型的枚举值。可选；可以根据张量类型推断。）
  - `ALL_F32 = 0`
  - `MOSTLY_F16 = 1`
  - `MOSTLY_Q4_0 = 2`
  - `MOSTLY_Q4_1 = 3`
  - `MOSTLY_Q4_1_SOME_F16 = 4`
  - `MOSTLY_Q4_2 = 5` (support removed)
  - `MOSTLY_Q4_3 = 6` (support removed)
  - `MOSTLY_Q8_0 = 7`
  - `MOSTLY_Q5_0 = 8`
  - `MOSTLY_Q5_1 = 9`
  - `MOSTLY_Q2_K = 10`
  - `MOSTLY_Q3_K_S = 11`
  - `MOSTLY_Q3_K_M = 12`
  - `MOSTLY_Q3_K_L = 13`
  - `MOSTLY_Q4_K_S = 14`
  - `MOSTLY_Q4_K_M = 15`
  - `MOSTLY_Q5_K_S = 16`
  - `MOSTLY_Q5_K_M = 17`
  - `MOSTLY_Q6_K = 18`

#### Source metadata

Information about where this model came from. This is useful for tracking the provenance of the model, and for finding the original source if the model is modified. For a model that was converted from GGML, for example, these keys would point to the model that was converted from.（有关此模型来源的信息。这对于追踪模型的出处以及在模型被修改时查找原始来源非常有用。例如，对于从 GGML 转换而来的模型，这些键将指向转换自的模型。）

- `general.source.url: string`: URL to the source of the model's homepage. This can be a GitHub repo, a paper, etc.（模型主页的来源 URL。可以是 GitHub repo、论文等。）
- `general.source.doi: string`: Source Digital Object Identifier (DOI) https://www.doi.org/（源数字对象标识符 (DOI) https://www.doi.org/）
- `general.source.uuid: string`: Source [Universally unique identifier](https://en.wikipedia.org/wiki/Universally_unique_identifier)（来源 [通用唯一标识符](https://en.wikipedia.org/wiki/Universally_unique_identifier)）
- `general.source.repo_url: string`: URL to the source of the model's repository such as a GitHub repo or HuggingFace repo（模型存储库源的 URL，例如 GitHub repo 或 HuggingFace repo）

- `general.base_model.count: uint32`: Number of parent models
- `general.base_model.{id}.name: string`: The name of the parent model.
- `general.base_model.{id}.author: string`: The author of the parent model.
- `general.base_model.{id}.version: string`: The version of the parent model.
- `general.base_model.{id}.organization: string`: The organization of the parent model.
- `general.base_model.{id}.url: string`: URL to the source of the parent model's homepage. This can be a GitHub repo, a paper, etc.
- `general.base_model.{id}.doi: string`: Parent Digital Object Identifier (DOI) https://www.doi.org/
- `general.base_model.{id}.uuid: string`: Parent [Universally unique identifier](https://en.wikipedia.org/wiki/Universally_unique_identifier)
- `general.base_model.{id}.repo_url: string`: URL to the source of the parent model's repository such as a GitHub repo or HuggingFace repo



### LLM

In the following, `[llm]` is used to fill in for the name of a specific LLM architecture. For example, `llama` for LLaMA, `mpt` for MPT, etc. If mentioned in an architecture's section, it is required for that architecture, but not all keys are required for all architectures. Consult the relevant section for more information.（下文中，`[llm]` 用于填写特定 LLM 架构的名称。例如，`llama` 代表 LLaMA，`mpt` 代表 MPT，等等。如果在架构部分中提到，则该架构需要该键，但并非所有架构都需要所有键。请参阅相关部分以获取更多信息。）

- `[llm].context_length: uint64`: Also known as `n_ctx`. length of the context (in tokens) that the model was trained on. For most architectures, this is the hard limit on the length of the input. Architectures, like RWKV, that are not reliant on transformer-style attention may be able to handle larger inputs, but this is not guaranteed.（也称为“n_ctx”。模型训练上下文的长度（以 token 为单位）。对于大多数架构而言，这是输入长度的硬性限制。不依赖于 transformer 式注意力机制的架构（如 RWKV）可能能够处理更大的输入，但这并不能保证。）
- `[llm].embedding_length: uint64`: Also known as `n_embd`. Embedding layer size.
- `[llm].block_count: uint64`: The number of blocks of attention+feed-forward layers (i.e. the bulk of the LLM). Does not include the input or embedding layers.（注意力+前馈层块的数量（即 LLM 的大部分）。不包括输入层或嵌入层。）
- `[llm].feed_forward_length: uint64`: Also known as `n_ff`. The length of the feed-forward layer.
- `[llm].use_parallel_residual: bool`: Whether or not the parallel residual logic should be used.（是否应该使用并行残差逻辑。）
- `[llm].tensor_data_layout: string`: When a model is converted to GGUF, tensors may be rearranged to improve performance. This key describes the layout of the tensor data. This is not required; if not present, it is assumed to be `reference`.（当模型转换为 GGUF 时，可能会重新排列张量以提高性能。此键描述张量数据的布局。这不是必需的；如果不存在，则假定为“引用”。）
  - `reference`: tensors are laid out in the same order as the original model（张量的排列顺序与原始模型相同）
  - further options can be found for each architecture in their respective sections（可以在每个架构的相应部分中找到更多选项）
- `[llm].expert_count: uint32`: Number of experts in MoE models (optional for non-MoE arches).
- `[llm].expert_used_count: uint32`: Number of experts used during each token token evaluation (optional for non-MoE arches).

#### Attention

- `[llm].attention.head_count: uint64`: Also known as `n_head`. Number of attention heads.
- `[llm].attention.head_count_kv: uint64`: The number of heads per group used in Grouped-Query-Attention. If not present or if present and equal to `[llm].attention.head_count`, the model does not use GQA.（Grouped-Query-Attention 中使用的每组 head 数量。如果不存在，或者存在且等于 `[llm].attention.head_count`，则模型不使用 GQA。）
- `[llm].attention.max_alibi_bias: float32`: The maximum bias to use for ALiBI.
- `[llm].attention.clamp_kqv: float32`: Value (`C`) to clamp the values of the `Q`, `K`, and `V` tensors between (`[-C, C]`).
- `[llm].attention.layer_norm_epsilon: float32`: Layer normalization epsilon.
- `[llm].attention.layer_norm_rms_epsilon: float32`: Layer RMS normalization epsilon.
- `[llm].attention.key_length: uint32`: The optional size of a key head, $d_k$. If not specified, it will be `n_embd / n_head`.
- `[llm].attention.value_length: uint32`: The optional size of a value head, $d_v$. If not specified, it will be `n_embd / n_head`.

#### RoPE

- `[llm].rope.dimension_count: uint64`: The number of rotary dimensions for RoPE.（RoPE 的旋转维度数。）
- `[llm].rope.freq_base: float32`: The base frequency for RoPE.（RoPE 的基准频率。）

##### Scaling

The following keys describe RoPE scaling parameters:

- `[llm].rope.scaling.type: string`: Can be `none`, `linear`, or `yarn`.
- `[llm].rope.scaling.factor: float32`: A scale factor for RoPE to adjust the context length.
- `[llm].rope.scaling.original_context_length: uint32_t`: The original context length of the base model.
- `[llm].rope.scaling.finetuned: bool`: True if model has been finetuned with RoPE scaling.

Note that older models may not have these keys, and may instead use the following key:

- `[llm].rope.scale_linear: float32`: A linear scale factor for RoPE to adjust the context length.

It is recommended that models use the newer keys if possible, as they are more flexible and allow for more complex scaling schemes. Executors will need to support both indefinitely.

#### SSM

- `[llm].ssm.conv_kernel: uint32`: The size of the rolling/shift state.
- `[llm].ssm.inner_size: uint32`: The embedding size of the states.
- `[llm].ssm.state_size: uint32`: The size of the recurrent state.
- `[llm].ssm.time_step_rank: uint32`: The rank of time steps.

#### Models

The following sections describe the metadata for each model architecture. Each key specified _must_ be present.（以下部分描述了每个模型架构的元数据。指定的每个键都必须存在。）

##### LLaMA

- `llama.context_length`
- `llama.embedding_length`
- `llama.block_count`
- `llama.feed_forward_length`
- `llama.rope.dimension_count`
- `llama.attention.head_count`
- `llama.attention.layer_norm_rms_epsilon`

###### Optional

- `llama.rope.scale`
- `llama.attention.head_count_kv`
- `llama.tensor_data_layout`:
  - `Meta AI original pth`:
    ```python
    def permute(weights: NDArray, n_head: int) -> NDArray:
        return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
                    .swapaxes(1, 2)
                    .reshape(weights.shape))
    ```
- `llama.expert_count`
- `llama.expert_used_count`

##### MPT

- `mpt.context_length`
- `mpt.embedding_length`
- `mpt.block_count`
- `mpt.attention.head_count`
- `mpt.attention.alibi_bias_max`
- `mpt.attention.clip_kqv`
- `mpt.attention.layer_norm_epsilon`

##### GPT-NeoX

- `gptneox.context_length`
- `gptneox.embedding_length`
- `gptneox.block_count`
- `gptneox.use_parallel_residual`
- `gptneox.rope.dimension_count`
- `gptneox.attention.head_count`
- `gptneox.attention.layer_norm_epsilon`

###### Optional

- `gptneox.rope.scale`

##### GPT-J

- `gptj.context_length`
- `gptj.embedding_length`
- `gptj.block_count`
- `gptj.rope.dimension_count`
- `gptj.attention.head_count`
- `gptj.attention.layer_norm_epsilon`

###### Optional

- `gptj.rope.scale`

##### GPT-2

- `gpt2.context_length`
- `gpt2.embedding_length`
- `gpt2.block_count`
- `gpt2.attention.head_count`
- `gpt2.attention.layer_norm_epsilon`

##### BLOOM

- `bloom.context_length`
- `bloom.embedding_length`
- `bloom.block_count`
- `bloom.feed_forward_length`
- `bloom.attention.head_count`
- `bloom.attention.layer_norm_epsilon`

##### Falcon

- `falcon.context_length`
- `falcon.embedding_length`
- `falcon.block_count`
- `falcon.attention.head_count`
- `falcon.attention.head_count_kv`
- `falcon.attention.use_norm`
- `falcon.attention.layer_norm_epsilon`

###### Optional

- `falcon.tensor_data_layout`:

  - `jploski` (author of the original GGML implementation of Falcon):

    ```python
    # The original query_key_value tensor contains n_head_kv "kv groups",
    # each consisting of n_head/n_head_kv query weights followed by one key
    # and one value weight (shared by all query heads in the kv group).
    # This layout makes it a big pain to work with in GGML.
    # So we rearrange them here,, so that we have n_head query weights
    # followed by n_head_kv key weights followed by n_head_kv value weights,
    # in contiguous fashion.

    if "query_key_value" in src:
        qkv = model[src].view(
            n_head_kv, n_head // n_head_kv + 2, head_dim, head_dim * n_head)

        q = qkv[:, :-2 ].reshape(n_head * head_dim, head_dim * n_head)
        k = qkv[:, [-2]].reshape(n_head_kv * head_dim, head_dim * n_head)
        v = qkv[:, [-1]].reshape(n_head_kv * head_dim, head_dim * n_head)

        model[src] = torch.cat((q,k,v)).reshape_as(model[src])
    ```

##### Mamba

- `mamba.context_length`
- `mamba.embedding_length`
- `mamba.block_count`
- `mamba.ssm.conv_kernel`
- `mamba.ssm.inner_size`
- `mamba.ssm.state_size`
- `mamba.ssm.time_step_rank`
- `mamba.attention.layer_norm_rms_epsilon`

##### RWKV

The vocabulary size is the same as the number of rows in the `head` matrix.

- `rwkv.architecture_version: uint32`: The only allowed value currently is 4. Version 5 is expected to appear some time in the future.
- `rwkv.context_length: uint64`: Length of the context used during training or fine-tuning. RWKV is able to handle larger context than this limit, but the output quality may suffer.
- `rwkv.block_count: uint64`
- `rwkv.embedding_length: uint64`
- `rwkv.feed_forward_length: uint64`

##### Whisper

Keys that do not have types defined should be assumed to share definitions with `llm.` keys.
(For example, `whisper.context_length` is equivalent to `llm.context_length`.)
This is because they are both transformer models.

- `whisper.encoder.context_length`
- `whisper.encoder.embedding_length`
- `whisper.encoder.block_count`
- `whisper.encoder.mels_count: uint64`
- `whisper.encoder.attention.head_count`

- `whisper.decoder.context_length`
- `whisper.decoder.embedding_length`
- `whisper.decoder.block_count`
- `whisper.decoder.attention.head_count`

#### Prompting

**TODO**: Include prompt format, and/or metadata about how it should be used (instruction, conversation, autocomplete, etc).（包括提示格式和/或有关如何使用它的元数据（指令、对话、自动完成等））

### LoRA

**TODO**: Figure out what metadata is needed for LoRA. Probably desired features（弄清楚 LoRA 需要哪些元数据。可能需要的功能）:

- match an existing model exactly, so that it can't be misapplied（与现有模型完全匹配，以免被误用）
- be marked as a LoRA so executors won't try to run it by itself（被标记为 LoRA，因此执行器不会尝试自行运行它）

Should this be an architecture, or should it share the details of the original model with additional fields to mark it as a LoRA?（这应该是一个架构吗，还是应该与附加字段共享原始模型的细节以将其标记为 LoRA？）

### Tokenizer

The following keys are used to describe the tokenizer of the model. It is recommended that model authors support as many of these as possible, as it will allow for better tokenization quality with supported executors.（以下键用于描述模型的标记器。建议模型作者尽可能多地支持这些键，因为这将允许使用受支持的执行器实现更好的标记化质量。）

#### GGML

GGML supports an embedded vocabulary that enables inference of the model, but implementations of tokenization using this vocabulary (i.e. `llama.cpp`'s tokenizer) may have lower accuracy than the original tokenizer used for the model. When a more accurate tokenizer is available and supported, it should be used instead.（GGML 支持嵌入词汇表，可实现模型推理，但使用此词汇表（即 `llama.cpp` 的标记器）的标记化实现的准确率可能低于用于模型的原始标记器。当有更准确的标记器可用且受支持时，应改用它。）

It is not guaranteed to be standardized across models, and may change in the future. It is recommended that model authors use a more standardized tokenizer if possible.（它不能保证在各个模型之间实现标准化，并且将来可能会发生变化。建议模型作者尽可能使用更标准化的标记器。）

- `tokenizer.ggml.model: string`: The name of the tokenizer model.
  - `llama`: Llama style SentencePiece (tokens and scores extracted from HF `tokenizer.model`)
  - `replit`: Replit style SentencePiece (tokens and scores extracted from HF `spiece.model`)
  - `gpt2`: GPT-2 / GPT-NeoX style BPE (tokens extracted from HF `tokenizer.json`)
  - `rwkv`: RWKV tokenizer
- `tokenizer.ggml.tokens: array[string]`: A list of tokens indexed by the token ID used by the model.
- `tokenizer.ggml.scores: array[float32]`: If present, the score/probability of each token. If not present, all tokens are assumed to have equal probability. If present, it must have the same length and index as `tokens`.
- `tokenizer.ggml.token_type: array[int32]`: The token type (1=normal, 2=unknown, 3=control, 4=user defined, 5=unused, 6=byte). If present, it must have the same length and index as `tokens`.
- `tokenizer.ggml.merges: array[string]`: If present, the merges of the tokenizer. If not present, the tokens are assumed to be atomic.
- `tokenizer.ggml.added_tokens: array[string]`: If present, tokens that were added after training.

##### Special tokens

- `tokenizer.ggml.bos_token_id: uint32`: Beginning of sequence marker
- `tokenizer.ggml.eos_token_id: uint32`: End of sequence marker
- `tokenizer.ggml.unknown_token_id: uint32`: Unknown token
- `tokenizer.ggml.separator_token_id: uint32`: Separator token
- `tokenizer.ggml.padding_token_id: uint32`: Padding token

#### Hugging Face

Hugging Face maintains their own `tokenizers` library that supports a wide variety of tokenizers. If your executor uses this library, it may be able to use the model's tokenizer directly.（Hugging Face 维护着自己的“tokenizers”库，该库支持各种 tokenizer。如果您的执行器使用此库，则它可能能够直接使用模型的 tokenizer。）

- `tokenizer.huggingface.json: string`: the entirety of the HF `tokenizer.json` for a given model (e.g. <https://huggingface.co/mosaicml/mpt-7b-instruct/blob/main/tokenizer.json>). Included for compatibility with executors that support HF tokenizers directly.

#### Other

Other tokenizers may be used, but are not necessarily standardized. They may be executor-specific. They will be documented here as they are discovered/further developed.（可以使用其他标记器，但不一定标准化。它们可能是特定于执行器的。它们将在发现/进一步开发时在此处记录。）

- `tokenizer.rwkv.world: string`: a RWKV World tokenizer, like [this](https://github.com/BlinkDL/ChatRWKV/blob/main/tokenizer/rwkv_vocab_v20230424.txt). This text file should be included verbatim.
- `tokenizer.chat_template : string`: a Jinja template that specifies the input format expected by the model. For more details see: <https://huggingface.co/docs/transformers/main/en/chat_templating>

### Computation graph

This is a future extension and still needs to be discussed, and may necessitate a new GGUF version. At the time of writing, the primary blocker is the stabilization of the computation graph format.（这是未来的扩展，仍需讨论，并且可能需要新的 GGUF 版本。在撰写本文时，主要阻碍因素是计算图格式的稳定性。）

A sample computation graph of GGML nodes could be included in the model itself, allowing an executor to run the model without providing its own implementation of the architecture. This would allow for a more consistent experience across executors, and would allow for more complex architectures to be supported without requiring the executor to implement them.（GGML 节点的计算图示例可以包含在模型本身中，从而允许执行器运行模型而无需提供其自己的架构实现。这将允许跨执行器获得更一致的体验，并且允许支持更复杂的架构而无需执行器实现它们。）

## Standardized tensor names

To minimize complexity and maximize compatibility, it is recommended that models using the transformer architecture use the following naming convention for their tensors（为了最大限度地降低复杂性并最大限度地提高兼容性，建议使用 Transformer 架构的模型对其张量使用以下命名约定）:

### Base layers

`AA.weight` `AA.bias`

where `AA` can be:

- `token_embd`: Token embedding layer
- `pos_embd`: Position embedding layer
- `output_norm`: Output normalization layer
- `output`: Output layer

### Attention and feed-forward layer blocks

`blk.N.BB.weight` `blk.N.BB.bias`

where N signifies the block number a layer belongs to, and where `BB` could be:

- `attn_norm`: Attention normalization layer
- `attn_norm_2`: Attention normalization layer
- `attn_qkv`: Attention query-key-value layer
- `attn_q`: Attention query layer
- `attn_k`: Attention key layer
- `attn_v`: Attention value layer
- `attn_output`: Attention output layer

- `ffn_norm`: Feed-forward network normalization layer
- `ffn_up`: Feed-forward network "up" layer
- `ffn_gate`: Feed-forward network "gate" layer
- `ffn_down`: Feed-forward network "down" layer
- `ffn_gate_inp`: Expert-routing layer for the Feed-forward network in MoE models
- `ffn_gate_exp`: Feed-forward network "gate" layer per expert in MoE models
- `ffn_down_exp`: Feed-forward network "down" layer per expert in MoE models
- `ffn_up_exp`: Feed-forward network "up" layer per expert in MoE models

- `ssm_in`: State space model input projections layer
- `ssm_conv1d`: State space model rolling/shift layer
- `ssm_x`: State space model selective parametrization layer
- `ssm_a`: State space model state compression layer
- `ssm_d`: State space model skip connection layer
- `ssm_dt`: State space model time step layer
- `ssm_out`: State space model output projection layer

## Version History

This document is actively updated to describe the current state of the metadata, and these changes are not tracked outside of the commits.（该文档会积极更新以描述元数据的当前状态，并且这些更改不会在提交之外进行跟踪。）

However, the format _itself_ has changed. The following sections describe the changes to the format itself.（但是，格式本身已经发生了变化。以下部分描述了格式本身的变化。）

### v3

Adds big-endian support.

### v2

Most countable values (lengths, etc) were changed from `uint32` to `uint64` to allow for larger models to be supported in the future.

### v1

Initial version.

## Historical State of Affairs

The following information is provided for context, but is not necessary to understand the rest of this document.（下列信息仅供参考，但对于理解本文档的其余部分而言并非必需。）

### Overview

At present, there are three GGML file formats floating around for LLMs:

- **GGML** (unversioned): baseline format, with no versioning or alignment.
- **GGMF** (versioned): the same as GGML, but with versioning. Only one version exists.
- **GGJT**: Aligns the tensors to allow for use with `mmap`, which requires alignment. v1, v2 and v3 are identical, but the latter versions use a different quantization scheme that is incompatible with previous versions.

GGML is primarily used by the examples in `ggml`, while GGJT is used by `llama.cpp` models. Other executors may use any of the three formats, but this is not 'officially' supported.

These formats share the same fundamental structure:

- a magic number with an optional version number
- model-specific hyperparameters, including
  - metadata about the model, such as the number of layers, the number of heads, etc.
  - a `ftype` that describes the type of the majority of the tensors,
    - for GGML files, the quantization version is encoded in the `ftype` divided by 1000
- an embedded vocabulary, which is a list of strings with length prepended. The GGMF/GGJT formats embed a float32 score next to the strings.
- finally, a list of tensors with their length-prepended name, type, and (aligned, in the case of GGJT) tensor data

Notably, this structure does not identify what model architecture the model belongs to, nor does it offer any flexibility for changing the structure of the hyperparameters. This means that the only way to add new hyperparameters is to add them to the end of the list, which is a breaking change for existing models.

### Drawbacks

Unfortunately, over the last few months, there are a few issues that have become apparent with the existing models:

- There's no way to identify which model architecture a given model is for, because that information isn't present
  - Similarly, existing programs cannot intelligently fail upon encountering new architectures
- Adding or removing any new hyperparameters is a breaking change, which is impossible for a reader to detect without using heuristics
- Each model architecture requires its own conversion script to their architecture's variant of GGML
- Maintaining backwards compatibility without breaking the structure of the format requires clever tricks, like packing the quantization version into the ftype, which are not guaranteed to be picked up by readers/writers, and are not consistent between the two formats

### Why not other formats?

There are a few other formats that could be used, but issues include:

- requiring additional dependencies to load or save the model, which is complicated in a C environment
- limited or no support for 4-bit quantization
- existing cultural expectations (e.g. whether or not the model is a directory or a file)
- lack of support for embedded vocabularies
- lack of control over direction of future development

Ultimately, it is likely that GGUF will remain necessary for the foreseeable future, and it is better to have a single format that is well-documented and supported by all executors than to contort an existing format to fit the needs of GGML.（最终，GGUF 在可预见的未来很可能仍是必需的，并且最好拥有一种有据可查且得到所有执行者支持的单一格式，而不是扭曲现有格式来满足 GGML 的需求）
