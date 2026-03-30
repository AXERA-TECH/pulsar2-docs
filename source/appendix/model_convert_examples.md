# 模型转换示例

本章节提供多种典型模型的 `pulsar2 build` 转换示例, 包括完整的配置文件、转换命令、真实 log 及模型输入输出说明. 所有示例均基于 `AX650` 平台, 使用 Pulsar2 5.1 版本.

:::{note}
- 本章节中的模型和配置文件均来自 [AXERA-TECH HuggingFace](https://huggingface.co/AXERA-TECH)
- 转换前请确保已使用 `onnxsim` 工具对原始模型进行优化
- 模型的输入输出 tensor 名称需与 ONNX 模型实际定义一致, 可通过 `onnx inspect --io model.onnx` 查看
:::

(convert_yolov5s)

## YOLOv5s (目标检测)

### 模型简介

`YOLOv5s` 是 Ultralytics 发布的实时目标检测模型, 采用 CSPDarknet 骨干网络, 适用于实时检测场景.

- **HuggingFace**: [AXERA-TECH/YOLOv5](https://huggingface.co/AXERA-TECH/YOLOv5)
- **模型来源**: [ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- **AxSamples**: [ax-samples](https://github.com/AXERA-TECH/ax-samples/blob/main/examples/ax650/ax_yolov5s_steps.cc) / [axcl-samples](https://github.com/AXERA-TECH/axcl-samples/blob/main/examples/axcl/ax_yolov5s_steps.cc)

### 配置文件

`yolov5_build.json`:

```json
{
  "model_type": "ONNX",
  "npu_mode": "NPU1",
  "quant": {
    "input_configs": [
      {
        "tensor_name": "images",
        "calibration_dataset": "calib-cocotest2017.tar",
        "calibration_size": 32,
        "calibration_mean": [0, 0, 0],
        "calibration_std": [255.0, 255.0, 255.0]
      }
    ],
    "calibration_method": "MinMax",
    "precision_analysis": false
  },
  "input_processors": [
    {
      "tensor_name": "images",
      "tensor_format": "RGB",
      "src_format": "BGR",
      "src_dtype": "U8",
      "src_layout": "NHWC"
    }
  ],
  "output_processors": [
    {
      "tensor_name": "/model.24/m.0/Conv_output_0",
      "dst_perm": [0, 2, 3, 1]
    },
    {
      "tensor_name": "/model.24/m.1/Conv_output_0",
      "dst_perm": [0, 2, 3, 1]
    },
    {
      "tensor_name": "/model.24/m.2/Conv_output_0",
      "dst_perm": [0, 2, 3, 1]
    }
  ],
  "compiler": {
    "check": 0
  }
}
```

:::{attention}
`output_processors` 中的 `tensor_name` 为 YOLOv5s 模型三个检测头的输出名称. 不同版本的模型可能不同, 请使用 `onnx inspect --io model.onnx` 查看实际的 tensor 名称. `dst_perm` 用于将输出从 `NCHW` 转换为 `NHWC` 布局, 便于后处理.
:::

### 编译执行

```shell
pulsar2 build --target_hardware AX650 --input yolov5s-cut.onnx --output_dir output --config yolov5_build.json
```

#### log 参考信息

```
+----------------------------------+----------------------------+
|            Model Name            |         OnnxModel          |
+----------------------------------+----------------------------+
|            Model Info            | Op Set: 17 / IR Version: 8 |
+----------------------------------+----------------------------+
|            IN: images            | float32: (1, 3, 640, 640)  |
| OUT: /model.24/m.0/Conv_output_0 | float32: (1, 255, 80, 80)  |
| OUT: /model.24/m.1/Conv_output_0 | float32: (1, 255, 40, 40)  |
| OUT: /model.24/m.2/Conv_output_0 | float32: (1, 255, 20, 20)  |
+----------------------------------+----------------------------+
|               Add                |             7              |
|              Concat              |             13             |
|               Conv               |             60             |
|             MaxPool              |             3              |
|               Mul                |             57             |
|              Resize              |             2              |
|             Sigmoid              |             57             |
+----------------------------------+----------------------------+
|            Model Size            |          27.56 MB          |
+----------------------------------+----------------------------+
...
Calibration Progress(Phase 1): 100%|██████████| 32/32 [00:17<00:00,  1.79it/s]
...
--------- Network Snapshot ---------
Num of Op:                    [142]
Num of Quantized Op:          [142]
Num of Variable:              [269]
Num of Quantized Var:         [269]
------- Quantization Snapshot ------
Num of Quant Config:          [432]
BAKED:                        [60]
OVERLAPPED:                   [168]
SLAVE:                        [9]
ACTIVATED:                    [129]
SOI:                          [6]
PASSIVE_BAKED:                [60]
Network Quantization Finished.
...
tiling op...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 147/147 0:00:00
build op serially...   ━━━━━━━━━━━━━━━━━━━━━━━━━━ 649/649 0:00:02
build op...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1139/1139 0:00:00
...
2026-03-23 19:44:00.890 | INFO     | yamain.command.build:compile_ptq_model:1365 - fuse 1 subgraph(s)
```

### 模型输入输出说明

```{eval-rst}
.. list-table::
   :header-rows: 1

   * - 方向
     - Tensor 名称
     - 数据类型
     - Shape
     - 说明
   * - 输入
     - images
     - UINT8
     - (1, 640, 640, 3)
     - BGR 格式图像, NHWC 布局, 需 Letterbox 预处理
   * - 输出
     - /model.24/m.0/Conv_output_0
     - FLOAT32
     - (1, 80, 80, 255)
     - 大尺度特征图 (检测小目标)
   * - 输出
     - /model.24/m.1/Conv_output_0
     - FLOAT32
     - (1, 40, 40, 255)
     - 中尺度特征图 (检测中目标)
   * - 输出
     - /model.24/m.2/Conv_output_0
     - FLOAT32
     - (1, 20, 20, 255)
     - 小尺度特征图 (检测大目标)
```

:::{hint}
板端推理耗时约 `6.32 ms` (AX650). 完整的板端运行示例参考 [AXERA-TECH/YOLOv5](https://huggingface.co/AXERA-TECH/YOLOv5).
:::

(convert_yolo11s)

## YOLO11s (目标检测)

### 模型简介

`YOLO11s` 是 Ultralytics 发布的最新一代 YOLO 检测模型, 采用改进的骨干网络和检测头设计, 精度和速度优于上一代.

- **HuggingFace**: [AXERA-TECH/YOLO11](https://huggingface.co/AXERA-TECH/YOLO11)
- **模型来源**: [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **AxSamples**: [ax-samples](https://github.com/AXERA-TECH/ax-samples/blob/main/examples/ax650/ax_yolo11_steps.cc) / [axcl-samples](https://github.com/AXERA-TECH/axcl-samples/blob/main/examples/axcl/ax_yolo11_steps.cc)

### 配置文件

`yolo11_build.json`:

```json
{
  "model_type": "ONNX",
  "npu_mode": "NPU1",
  "quant": {
    "input_configs": [
      {
        "tensor_name": "images",
        "calibration_dataset": "calib-cocotest2017.tar",
        "calibration_size": 32,
        "calibration_mean": [0, 0, 0],
        "calibration_std": [255.0, 255.0, 255.0]
      }
    ],
    "calibration_method": "MinMax",
    "precision_analysis": false
  },
  "input_processors": [
    {
      "tensor_name": "images",
      "tensor_format": "BGR",
      "src_format": "BGR",
      "src_dtype": "U8",
      "src_layout": "NHWC"
    }
  ],
  "output_processors": [
    {
      "tensor_name": "/model.23/Concat_output_0",
      "dst_perm": [0, 2, 3, 1]
    },
    {
      "tensor_name": "/model.23/Concat_1_output_0",
      "dst_perm": [0, 2, 3, 1]
    },
    {
      "tensor_name": "/model.23/Concat_2_output_0",
      "dst_perm": [0, 2, 3, 1]
    }
  ],
  "compiler": {
    "check": 0
  }
}
```

### 编译执行

```shell
pulsar2 build --target_hardware AX650 --input yolo11s-cut.onnx --output_dir output --config yolo11_build.json
```

#### log 参考信息

```
+----------------------------------+----------------------------+
|            Model Name            |         OnnxModel          |
+----------------------------------+----------------------------+
|            Model Info            | Op Set: 17 / IR Version: 9 |
+----------------------------------+----------------------------+
|            IN: images            | float32: (1, 3, 640, 640)  |
|  OUT: /model.23/Concat_output_0  | float32: (1, 144, 80, 80)  |
| OUT: /model.23/Concat_1_output_0 | float32: (1, 144, 40, 40)  |
| OUT: /model.23/Concat_2_output_0 | float32: (1, 144, 20, 20)  |
+----------------------------------+----------------------------+
|               Add                |             14             |
|              Concat              |             20             |
|               Conv               |             87             |
|              MatMul              |             2              |
|             MaxPool              |             3              |
|               Mul                |             78             |
|             Reshape              |             3              |
|              Resize              |             2              |
|             Sigmoid              |             77             |
|             Softmax              |             1              |
|              Split               |             10             |
|            Transpose             |             2              |
+----------------------------------+----------------------------+
|            Model Size            |          36.03 MB          |
+----------------------------------+----------------------------+
...
Calibration Progress(Phase 1): 100%|██████████| 32/32 [00:25<00:00,  1.25it/s]
...
--------- Network Snapshot ---------
Num of Op:                    [222]
Num of Quantized Op:          [222]
Num of Variable:              [426]
Num of Quantized Var:         [426]
------- Quantization Snapshot ------
Num of Quant Config:          [693]
BAKED:                        [88]
OVERLAPPED:                   [295]
SLAVE:                        [16]
ACTIVATED:                    [190]
SOI:                          [17]
PASSIVE_BAKED:                [87]
Network Quantization Finished.
...
tiling op...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 235/235 0:00:01
build op serially...   ━━━━━━━━━━━━━━━━━━━━━━━ 1033/1033 0:00:05
build op...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1689/1689 0:00:00
...
2026-03-23 19:45:20.303 | INFO     | yamain.command.build:compile_ptq_model:1365 - fuse 1 subgraph(s)
```

### 模型输入输出说明

```{eval-rst}
.. list-table::
   :header-rows: 1

   * - 方向
     - Tensor 名称
     - 数据类型
     - Shape
     - 说明
   * - 输入
     - images
     - UINT8
     - (1, 640, 640, 3)
     - BGR 格式图像, NHWC 布局, 需 Letterbox 预处理
   * - 输出
     - /model.23/Concat_output_0
     - FLOAT32
     - (1, 80, 80, 144)
     - 大尺度特征图 (检测小目标)
   * - 输出
     - /model.23/Concat_1_output_0
     - FLOAT32
     - (1, 40, 40, 144)
     - 中尺度特征图 (检测中目标)
   * - 输出
     - /model.23/Concat_2_output_0
     - FLOAT32
     - (1, 20, 20, 144)
     - 小尺度特征图 (检测大目标)
```

:::{hint}
YOLO11 相比 YOLOv5 采用了注意力机制 (含 `MatMul` 和 `Softmax` 算子), 模型更大但检测精度更高. 板端推理耗时约 `25 ms` (AX650). 完整的板端运行示例参考 [AXERA-TECH/YOLO11](https://huggingface.co/AXERA-TECH/YOLO11).
:::

(convert-depth-anything_v2)

## Depth-Anything-V2 (单目深度估计)

### 模型简介

`Depth-Anything-V2` 是基于 DINOv2 的单目深度估计模型, 输入单张 RGB 图片, 输出逐像素深度图. 本示例使用 ViT-Small 版本.

- **HuggingFace**: [AXERA-TECH/Depth-Anything-V2](https://huggingface.co/AXERA-TECH/Depth-Anything-V2)
- **模型来源**: [depth-anything/Depth-Anything-V2-Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small)
- **ONNX导出参考**: [DepthAnythingV2](https://github.com/AXERA-TECH/DepthAnythingV2.axera/tree/main/model_convert)

### 配置文件

`config.json` (省略部分 `layer_configs` 条目, 完整配置请参考 HuggingFace 仓库):

```
{
  "model_type": "ONNX",
  "npu_mode": "NPU3",
  "quant": {
    "input_configs": [
      {
        "tensor_name": "DEFAULT",
        "calibration_dataset": "calib-cocotest2017.tar",
        "calibration_size": 32,
        "calibration_mean": [123.675, 116.28, 103.53],
        "calibration_std": [58.395, 57.12, 57.375]
      }
    ],
    "calibration_method": "MinMax",
    "precision_analysis": true,
    "precision_analysis_method": "EndToEnd",
    "conv_bias_data_type": "FP32",
    "enable_smooth_quant": true,
    "disable_auto_refine_scale": true,
    "layer_configs":  [
      {
        "layer_name": "op_173:onnx.Mul_1",
        "data_type": "U16"
      },
      {
        "layer_name": "op_173:onnx.Softmax_0",
        "data_type": "U16"
      },
      {
        "layer_name": "op_173:onnx.MatMul_qkv_0",
        "data_type": "U16"
      },
      ...
    ]
  },
  "input_processors": [
    {
      "tensor_name": "DEFAULT",
      "tensor_format": "RGB",
      "src_format": "BGR",
      "src_dtype": "U8",
      "src_layout": "NHWC"
    }
  ],
  "compiler": {
    "check": 0
  }
}
```

:::{attention}
- 该模型使用 `NPU3` 模式 (三核), 充分利用 AX650 的全部 NPU 算力
- 开启了 `enable_smooth_quant` 以降低 Transformer 结构中的 outlier 影响
- `layer_configs` 中配置了大量 Softmax、MatMul 等算子使用 `U16` 精度, 以保证 ViT 模型的量化精度
- `conv_bias_data_type` 设置为 `FP32` 以提升精度
- 完整的 `layer_configs` (约 50 项) 请参考 [HuggingFace 仓库中的 config.json](https://huggingface.co/AXERA-TECH/Depth-Anything-V2/blob/main/config.json)
:::

### 编译执行

```shell
pulsar2 build --target_hardware AX650 --input depth_anything_v2_vits.onnx --output_dir output --config config.json
```

#### log 参考信息

```
+---------------+----------------------------+
|  Model Name   |         OnnxModel          |
+---------------+----------------------------+
|  Model Info   | Op Set: 12 / IR Version: 7 |
+---------------+----------------------------+
|   IN: input   | float32: (1, 3, 518, 518)  |
|  OUT: output  | float32: (1, 1, 518, 518)  |
+---------------+----------------------------+
|      Add      |            148             |
|    Concat     |             1              |
|     Conv      |             31             |
| ConvTranspose |             2              |
|      Div      |             37             |
|      Erf      |             12             |
|    Gather     |             36             |
|    MatMul     |             72             |
|      Mul      |             88             |
|      Pow      |             25             |
|  ReduceMean   |             50             |
|     Relu      |             16             |
|    Reshape    |             29             |
|    Resize     |             5              |
|     Slice     |             4              |
|    Softmax    |             12             |
|     Sqrt      |             25             |
|      Sub      |             25             |
|   Transpose   |             41             |
+---------------+----------------------------+
|  Model Size   |          94.26 MB          |
+---------------+----------------------------+
...
Enable Smooth Quant, this pass is used for outlier activation.
...
Analysing Smooth Quantization Error(Phrase 1): 100%|██████████| 32/32 [00:51<00:00,  1.62s/it]
Get Outlier Progress: 100%|██████████| 32/32 [01:15<00:00,  2.35s/it]
...
Analysing Smooth Quantization Error(Phrase 2): 100%|██████████| 32/32 [00:51<00:00,  1.62s/it]
...
--------- Network Snapshot ---------
Num of Op:                    [792]
Num of Quantized Op:          [792]
Num of Variable:              [1552]
Num of Quantized Var:         [1552]
------- Quantization Snapshot ------
Num of Quant Config:          [2581]
...
Network Quantization Finished.
...
tiling op...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 762/762 0:00:02
build op serially...   ━━━━━━━━━━━━━━━━━━━━━━━ 1178/1178 0:00:11
build op...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1734/1734 0:00:00
add ddr swap...   ━━━━━━━━━━━━━━━━━━━━━━━━━ 15821/15821 0:00:01
...
2026-03-23 19:42:38.553 | INFO     | yamain.command.build:compile_ptq_model:1365 - fuse 1 subgraph(s)
```

:::{note}
该模型转换全流程耗时约 `8 分钟`, 其中 Smooth Quant 分析和逐层精度对分占用了大部分时间. 如果不需要精度分析, 可将 `precision_analysis` 设置为 `false` 以加快转换速度.
:::

### 模型输入输出说明

```{eval-rst}
.. list-table::
   :header-rows: 1

   * - 方向
     - Tensor 名称
     - 数据类型
     - Shape
     - 说明
   * - 输入
     - input
     - UINT8
     - (1, 518, 518, 3)
     - BGR 格式图像 (运行时输入 BGR 自动转换为 RGB), NHWC 布局
   * - 输出
     - output
     - FLOAT32
     - (1, 1, 518, 518)
     - 逐像素深度图, 值越大表示距离越远
```

:::{hint}
板端推理耗时约 `33 ms` (AX650, NPU3 三核模式). 使用 Python 推理示例参考 [AXERA-TECH/Depth-Anything-V2](https://huggingface.co/AXERA-TECH/Depth-Anything-V2), 需安装 [pyaxengine](https://github.com/AXERA-TECH/pyaxengine).
:::

(convert_cnclip)

## CN-CLIP (中文多模态文本编码器)

### 模型简介

`Chinese-CLIP` 是基于 CLIP 框架的中文多模态预训练模型. 本示例为其 **BERT 文本编码器** 部分 (ViT-L/14 配套), 将中文文本编码为嵌入向量, 用于与图像嵌入计算相似度.

- **HuggingFace**: [AXERA-TECH/cnclip](https://huggingface.co/AXERA-TECH/cnclip)
- **模型来源**: [OFA-Sys/Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP)
- **ONNX导出参考**: [cnclip.axera](https://github.com/AXERA-TECH/cnclip.axera?tab=readme-ov-file#%E5%AF%BC%E5%87%BA%E6%A8%A1%E5%9E%8Bpytorch---onnx)
- **AxSamples**: [CLIP-ONNX-AX650-CPP](https://github.com/AXERA-TECH/CLIP-ONNX-AX650-CPP)

### 配置文件

`cnclip_build.json`:

```json
{
  "model_type": "ONNX",
  "npu_mode": "NPU1",
  "quant": {
    "input_configs": [
      {
        "tensor_name": "text",
        "calibration_dataset": "calib_text.tar",
        "calibration_format": "Numpy",
        "calibration_size": 32,
        "calibration_mean": [0],
        "calibration_std": [1]
      }
    ],
    "calibration_method": "MinMax",
    "precision_analysis": false,
    "transformer_opt_level": 1
  },
  "input_processors": [
    {
      "tensor_name": "text",
      "src_dtype": "S32",
      "src_layout": "NCHW"
    }
  ],
  "compiler": {
    "check": 0
  }
}
```

:::{attention}
- 该模型为 **文本编码器**, 输入为分词后的 token ID 序列, 不是图像
- `calibration_format` 设置为 `Numpy`, 校准数据为预分词的 numpy 数组 (shape `(1, 52)`, dtype `int64`)
- `src_dtype` 设置为 `S32` (有符号 32 位整数), 对应 token ID 输入
- `transformer_opt_level` 设置为 1, 启用 Transformer 模型专用量化优化
:::

### 编译执行

```shell
pulsar2 build --target_hardware AX650 --input cnclip_vit_l14_336px_bert_encoder.onnx --output_dir output --config cnclip_build.json
```

#### log 参考信息

```
+---------------------------+----------------------------+
|        Model Name         |         OnnxModel          |
+---------------------------+----------------------------+
|        Model Info         | Op Set: 14 / IR Version: 7 |
+---------------------------+----------------------------+
|         IN: text          |       int64: (1, 52)       |
| OUT: unnorm_text_features |     float32: (1, 768)      |
+---------------------------+----------------------------+
|            Add            |            172             |
|           Cast            |             3              |
|         Constant          |            154             |
|            Div            |             49             |
|            Erf            |             12             |
|          Gather           |             4              |
|          MatMul           |             97             |
|            Mul            |             50             |
|            Pow            |             25             |
|        ReduceMean         |             50             |
|          Reshape          |             48             |
|          Softmax          |             12             |
|           Sqrt            |             25             |
|            Sub            |             26             |
|         Transpose         |             48             |
+---------------------------+----------------------------+
|        Model Size         |         390.12 MB          |
+---------------------------+----------------------------+
...
Transformer optimize level: 1
...
Calibration Progress(Phase 1): 100%|██████████| 32/32 [00:11<00:00,  2.81it/s]
...
--------- Network Snapshot ---------
Num of Op:                    [312]
Num of Quantized Op:          [308]
Num of Variable:              [588]
Num of Quantized Var:         [583]
------- Quantization Snapshot ------
Num of Quant Config:          [949]
BAKED:                        [89]
OVERLAPPED:                   [452]
ACTIVATED:                    [224]
SOI:                          [61]
PASSIVE_BAKED:                [72]
FP32:                         [51]
Network Quantization Finished.
...
tiling op...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 340/340 0:00:02
build op serially...   ━━━━━━━━━━━━━━━━━━━━━━━━━━ 300/300 0:00:04
build op...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 386/386 0:00:00
add ddr swap...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3859/3859 0:00:00
...
2026-03-23 19:47:45.563 | INFO     | yamain.command.build:compile_ptq_model:1365 - fuse 1 subgraph(s)
```

### 模型输入输出说明

```{eval-rst}
.. list-table::
   :header-rows: 1

   * - 方向
     - Tensor 名称
     - 数据类型
     - Shape
     - 说明
   * - 输入
     - text
     - S32
     - (1, 52)
     - 分词后的 token ID 序列, 最大长度 52
   * - 输出
     - unnorm_text_features
     - FLOAT32
     - (1, 768)
     - 未归一化的文本嵌入向量
```

:::{hint}
部署时需配合视觉编码器使用: 视觉编码器提取图像嵌入, 文本编码器提取文本嵌入, 通过计算余弦相似度完成图文匹配. 分词器使用 `cn_vocab.txt` (随模型提供). 板端运行示例参考 [CLIP-ONNX-AX650-CPP](https://github.com/AXERA-TECH/CLIP-ONNX-AX650-CPP).
:::
