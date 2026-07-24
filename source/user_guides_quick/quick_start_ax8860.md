# Quick Start(AX8860)

**本章节适用于以下平台：**

- AX8860

本章节介绍 `ONNX` 模型转换的基本操作，使用 `pulsar2` 工具将 `ONNX` 模型编译成 `axmodel` 模型。请先参考 {ref}`《开发环境准备》 <dev_env_prepare>` 章节完成开发环境搭建。

`AX8860` 平台采用 Neutron v7 架构，更多硬件架构信息请参考 {ref}`《通元 NPU (Neutron) 介绍》 <soc_introduction>`。

本节示例模型为开源模型 `MobileNetv2`。

## Pulsar2 工具链命令说明

`Pulsar2` 工具链中的功能指令以 `pulsar2` 开头，与用户强相关的命令为 `pulsar2 build`、 `pulsar2 run` 以及 `pulsar2 version`。

- `pulsar2 build` 用于将 `onnx` 模型转换为 `axmodel` 格式模型
- `pulsar2 run` 用于模型转换后的仿真运行
- `pulsar2 version` 可以用于查看当前工具链的版本信息，通常在反馈问题时需要提供此信息

```shell
root@xxx:/data# pulsar2 --help
usage: pulsar2 [-h] {version,build,run} ...

positional arguments:
  {version,build,run}

optional arguments:
  -h, --help           show this help message and exit
```

## 模型编译配置文件说明

`/data/config/` 路径下的 `mobilenet_v2_build_config.json` 展示:

```shell
{
  "model_type": "ONNX",
  "npu_mode": "NPU1",
  "quant": {
    "input_configs": [
      {
        "tensor_name": "input",
        "calibration_dataset": "./dataset/imagenet-32-images.tar",
        "calibration_size": 32,
        "calibration_mean": [103.939, 116.779, 123.68],
        "calibration_std": [58.0, 58.0, 58.0]
      }
    ],
    "calibration_method": "MinMax",
    "precision_analysis": false
  },
  "input_processors": [
    {
      "tensor_name": "input",
      "tensor_format": "BGR",
      "src_format": "BGR",
      "src_dtype": "U8",
      "src_layout": "NHWC",
      "csc_mode": "NoCSC"
    }
  ],
  "compiler": {
    "check": 0
  }
}
```

::::{attention}
`input_processors`、 `output_processors` 及 `quant` 节点下 `input_configs` 中的 `tensor_name` 字段需要根据模型的实际输入/输出节点名称进行设置，也可以设置为 `DEFAULT` 代表当前配置应用于全部输入或者输出。

:::{figure} ../media/tensor_name.png
:align: center
:alt: tensor name
:::
::::

更加详细的内容，请参考 {ref}`《配置文件详细说明》 <config_details>`。

`AX8860` 平台中，配置文件的 `npu_mode` 字段用于指定模型编译时占用的 NPU Core 数量。其映射关系如下：

```{eval-rst}
.. list-table::
    :header-rows: 1
    :align: center

    * - ``npu_mode``
      - NPU Core 数量
    * - ``NPU1``
      - 1 个 NPU Core
    * - ``NPU2``
      - 2 个 NPU Core
    * - ``NPU4``
      - 4 个 NPU Core
```

:::{note}
`AX8860` 支持 `NPU1`、 `NPU2` 和 `NPU4` 编译模式，`NPU4` 表示使用全部 4 个 NPU Core。`npu_mode` 表示 Core 数量，不表示具体的 Core 编号。
:::

(model_compile_ax8860)=

## 编译执行

以 `mobilenetv2-sim.onnx` 为例，执行如下 `pulsar2 build` 命令编译生成 `compiled.axmodel`：

```shell
pulsar2 build --target_hardware AX8860 --input model/mobilenetv2-sim.onnx --output_dir output --config config/mobilenet_v2_build_config.json
```

:::{warning}
在编译模型前，需要确保已经对原始模型使用过 `onnxslim` 工具优化，主要目的是将模型转变成更利于 `Pulsar2` 编译的静态图并获得更好的推理性能。有以下两种方法：

1. 在 `Pulsar2` docker 内部直接执行命令：`onnxslim in.onnx out.onnx`。
2. 使用 `pulsar2 build` 进行模型转换时，增加参数：`--onnx_opt.enable_onnxsim true` （默认值为 false）。

如果想要进一步了解 `onnxslim`，可访问 [官方网站](https://github.com/inisis/OnnxSlim)。
:::

### 模型编译输出文件说明

```shell
root@xxx:/data# tree output/
output/
|-- build_context.json
|-- compiled.axmodel               # 最终板上运行模型，AxModel
|-- compiler                       # 编译器后端中间结果及 debug 信息
|   `-- debug
|       `-- subgraph_npu_0
|           `-- b1
|-- frontend
|   |-- optimized.data
|   `-- optimized.onnx             # 输入模型经过图优化以后的浮点 ONNX 模型
`-- quant                          # 量化工具输出及 debug 信息目录
    |-- dataset
    |   `-- input
    |-- debug
    |   `-- io
    |-- quant_axmodel.data
    |-- quant_axmodel.json         # 量化配置信息
    `-- quant_axmodel.onnx         # 量化后的模型，QuantAxModel
```

其中 `compiled.axmodel` 为最终编译生成的板上可运行的 `.axmodel` 模型文件。

::::{note}
因为 `.axmodel` 基于 **ONNX** 模型存储格式开发，所以将 `.axmodel` 文件后缀修改为 `.axmodel.onnx` 后可支持被网络模型图形化工具 **Netron** 直接打开。

:::{figure} ../media/axmodel-netron.png
:align: center
:alt: axmodel netron
:::
::::

(model_simulator_ax8860)=

## 仿真运行

本章节介绍 `axmodel` 仿真运行的基本操作。使用 `pulsar2 run` 命令可以直接在 `PC` 上运行由 `pulsar2 build` 生成的 `axmodel` 模型，无需上板运行即可快速得到网络模型的运行结果。

### 仿真运行准备

某些模型只能支持特定的输入数据格式，模型的输出数据也是以模型特定的格式输出的。在模型仿真运行前，需要把输入数据转换成模型支持的数据格式，这部分数据操作称为 `前处理`。在模型仿真运行后，需要把输出数据转换成工具可以分析查看的数据格式，这部分数据操作称为 `后处理`。仿真运行时需要的 `前处理` 和 `后处理` 工具已包含在 `pulsar2-run-helper` 文件夹中。

### 仿真运行示例 `mobilenetv2`

将 {ref}`《编译执行》 <model_compile_AX8860>` 章节生成的 `compiled.axmodel` 拷贝到 `pulsar2-run-helper/models` 路径下，并更名为 `mobilenetv2.axmodel`。

```shell
root@xxx:/data# cp output/compiled.axmodel pulsar2-run-helper/models/mobilenetv2.axmodel
```

进入 `pulsar2-run-helper` 目录，使用 `cli_classification.py` 脚本将 `cat.jpg` 处理成 `mobilenetv2.axmodel` 所需要的输入数据格式。

```shell
root@xxx:~/data# cd pulsar2-run-helper
root@xxx:~/data/pulsar2-run-helper# python3 cli_classification.py --pre_processing --image_path sim_images/cat.jpg --axmodel_path models/mobilenetv2.axmodel --intermediate_path sim_inputs/0
```

运行 `pulsar2 run` 命令，将 `input.bin` 作为 `mobilenetv2.axmodel` 的输入数据并执行推理计算，输出 `output.bin` 推理结果。

```shell
root@xxx:~/data/pulsar2-run-helper# pulsar2 run --model models/mobilenetv2.axmodel --input_dir sim_inputs --output_dir sim_outputs --list list.txt
```

使用 `cli_classification.py` 脚本对仿真模型推理输出的 `output.bin` 数据进行后处理，得到最终计算结果。

```shell
root@xxx:/data/pulsar2-run-helper# python3 cli_classification.py --post_processing --axmodel_path models/mobilenetv2.axmodel --intermediate_path sim_outputs/0
```

:::{note}
`AX8860` 板端运行方式与对应 SDK、AXEngine 运行环境及开发板镜像版本相关。完成 `compiled.axmodel` 编译后，可结合目标平台 SDK 文档和 {ref}`《模型部署进阶指南》 <model_deploy_advanced>` 完成板端集成。
:::
