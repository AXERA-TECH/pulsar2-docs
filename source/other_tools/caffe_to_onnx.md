# caffe2onnx 工具使用说明

本章介绍 AX 版 caffe2onnx 转换工具, 用于将 Caffe 浮点模型转换成 ONNX 浮点模型. 因为平台的限制, 目前 X86 平台跟 ARM 平台
发版提供的命令行接口有一些区别.

:::{note}
下文的模型语义皆为浮点模型.
:::

## 将 Caffe 模型转换成 ONNX 模型

我们提供了命令行工具将 Caffe 模型转换成 ONNX 模型. 在 X86 平台下, 您可以传入一个文件夹以转换其里面所有的 Caffe 模型：

```bash
caffe2onnx_cli --convert --checkpoint_path /path/to/your/model/zoo
```

这将递归地找到指定文件夹里面所有的以 ".caffemodel" 为后缀文件及其对应的 ".prototxt" 文
件, 此为一个 Caffe 模型, 将其转换为 ONNX 模型, 并使用 Caffe 模型的
前缀, 以 ".onnx" 为后缀进行保存.

:::{note}
Caffe 模型对应的 ".prototxt" 和 ".caffemodel"
文件需要在同一个文件夹并共享一个前缀.
:::

## 验证转换出的 ONNX 模型

在 X86 平台下, 您可以使用如下命令行工具对分原始的 Caffe 模型和转换出的 ONNX 模型：

```bash
caffe2onnx --validate --checkpoint_path /path/to/your/model/zoo
```

首先这将递归地找到指定文件夹里所有的以 ".onnx" 为后缀的文件, 然后按照其前缀匹配对应的
".prototxt" 和 ".caffemodel" 文件, 生成一个随机数据集, 分别使用 ONNXRuntime 和
Caffe 推理工具进行推理, 并计算两者的 “相关系数 (Correlation)”、“标准偏差 (Standard Deviation)”、
“余弦距离相似度 (Cosine Similarity)”、“归一化相对误差 (Normalized Relative Error)”、
“最大差异 (Max Difference)” 和 “平均差异 (Mean Difference)”.

:::{note}
Caffe 模型对应的 ".prototxt" 和 ".caffemodel"
文件以及转换出来的 ".onnx" 文件需要在同一个文件夹并共享一个前缀.
:::

:::{note}
此步限制标准 Caffe 算子.
:::

:::{note}
因为 Caffe ARM 平台的兼容性不太好, ARM 平台下暂时不支持这个功能.
:::
