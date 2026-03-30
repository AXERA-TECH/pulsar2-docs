# 常见配置示例

本章节提供 `pulsar2 build` 常见场景的配置文件示例, 方便用户快速查阅和复用. 所有示例均基于 `AX650` 平台.

:::{note}
- 配置文件字段的完整定义请参考 {ref}`《配置文件详细说明》 <config_details>`
- `tensor_name` 需与 ONNX 模型中的实际 tensor 名称一致, 可通过 `onnx inspect --io model.onnx` 查看
:::

(rgb_input_config)

## RGB 输入

最常见的图像模型配置. `input_processors` 用于声明 `compiled.axmodel` 运行时输入的数据属性, 工具链根据配置自动在模型中嵌入预处理算子 (如数据类型转换、归一化、布局转换等).

:::{warning}
`tensor_format` 与 `src_format` 的组合 **不支持** RGB ↔ BGR 通道互转. 配置 `src_format` 为 `BGR`、`tensor_format` 为 `RGB` (或反之) 时, 编译后的模型中 **不会** 嵌入通道重排算子. 色彩空间转换仅在 {ref}`YUV 输入 <yuv_input_config>` 场景中支持.
:::

### 配置预处理在 compiled.axmodel 中完成

将预处理 (归一化、Layout变换) 嵌入 `compiled.axmodel`, 配置如下:

```json
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
      "tensor_layout": "NHWC",
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

**关键配置:**

- `src_dtype` 设为 `U8`: 此时编译后的 `compiled.axmodel` 输入为 U8 类型，工具链会自动在模型前端嵌入 `AxDequantizeLinear` 反量化算子, 将 U8 转换为模型所需的 FP32。
- `src_layout` 设为 `NHWC`: 工具链嵌入 `AxTranspose` 算子, 将 NHWC 转换为模型所需的 NCHW
- `calibration_mean` / `calibration_std`: 工具链嵌入 `AxNormalize` 算子, 完成归一化

编译 log 中可以确认预处理算子已嵌入 (关注 `Building native` 之后的输出):

```bash
... | WARNING  | yamain.command.load_model:pre_process:616 - preprocess tensor [input]
... | INFO     | yamain.command.load_model:pre_process:618 - tensor: input, (1, 224, 224, 3), U8
... | INFO     | yamain.command.load_model:pre_process:619 - op: op:pre_dequant_1, AxDequantizeLinear, {'const_inputs': {'x_zeropoint': array(0, dtype=int32), 'x_scale': array(1., dtype=float32)}, 'output_dtype': <class 'numpy.float32'>, 'quant_method': 0}
... | INFO     | yamain.command.load_model:pre_process:618 - tensor: tensor:pre_norm_1, (1, 224, 224, 3), FP32
... | INFO     | yamain.command.load_model:pre_process:619 - op: op:pre_norm_1, AxNormalize, {'dim': 3, 'mean': [103.93900299072266, 116.77899932861328, 123.68000030517578], 'std': [58.0, 58.0, 58.0], 'output_dtype': FP32}
... | INFO     | yamain.command.load_model:pre_process:618 - tensor: tensor:pre_transpose_1, (1, 224, 224, 3), FP32
... | INFO     | yamain.command.load_model:pre_process:619 - op: op:pre_transpose_1, AxTranspose, {'perm': [0, 3, 1, 2]}
... | WARNING  | yamain.command.load_model:post_process:627 - postprocess tensor [output]
```

log 中 `preprocess tensor [input]` 与 `postprocess tensor [output]` 之间列出了三个预处理算子:

- `AxDequantizeLinear`: U8 → FP32 类型转换
- `AxNormalize`: 归一化 (减均值、除标准差)
- `AxTranspose`: NHWC → NCHW 布局转换

### 配置预处理不在 compiled.axmodel 中完成

如果希望自行在 CPU 端完成预处理 (归一化、Layout转换等) 后再送入 NPU 推理, 需要将 `input_processors` 配置为与浮点模型输入完全一致:

```json
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
      "tensor_layout": "NCHW",
      "src_format": "BGR",
      "src_dtype": "FP32",
      "src_layout": "NCHW",
      "mean": [0, 0, 0],
      "std": [1, 1, 1]
    }
  ],
  "compiler": {
    "check": 0
  }
}
```

**关键配置:**

- `src_dtype` 设为 `FP32`: 与模型输入类型一致, 不嵌入类型转换算子
- `src_layout` 设为 `NCHW`: 与模型输入布局一致, 不嵌入布局转换算子
- `mean` 显式设为 `[0, 0, 0]`, `std` 设为 `[1, 1, 1]`: 覆盖 `calibration_mean` / `calibration_std` 的默认值, 不嵌入归一化算子

:::{attention}
必须显式配置 `mean` 和 `std`. 如果不配置, 工具链会默认使用 `calibration_mean` / `calibration_std` 的值, 仍然会在模型中嵌入归一化算子.
:::

编译 log 中可以确认无预处理算子 (关注 `Building native` 之后的输出):

```bash
... | WARNING  | yamain.command.load_model:pre_process:616 - preprocess tensor [input]
... | WARNING  | yamain.command.load_model:post_process:627 - postprocess tensor [output]
```

`preprocess tensor [input]` 与 `postprocess tensor [output]` 之间没有任何 `op:` 行, 表示 `compiled.axmodel` 中不包含预处理算子. 运行时用户需自行完成:

1. 图像解码与 resize
2. BGR 通道归一化 (减均值、除标准差)
3. NHWC → NCHW 布局转换
4. 数据类型转换为 FP32

### 字段说明

```{eval-rst}
.. list-table::
   :header-rows: 1

   * - 字段
     - 说明
   * - ``tensor_format``
     - 模型训练时的通道顺序 (``RGB`` 或 ``BGR``), 用于校准数据读取时的色彩空间转换
   * - ``src_format``
     - 运行时实际输入的通道顺序, 通常为 ``BGR`` (OpenCV 默认)
   * - ``src_dtype``
     - 运行时输入数据类型. 设为 ``U8`` 时嵌入反量化算子; 设为 ``FP32`` 时不嵌入
   * - ``src_layout``
     - 运行时输入布局. 设为 ``NHWC`` 时自动嵌入布局转换; 设为 ``NCHW`` 时不嵌入
   * - ``mean`` / ``std``
     - 归一化参数. 默认使用 ``calibration_mean`` / ``calibration_std``. 显式设为 ``[0,0,0]`` / ``[1,1,1]`` 可禁用归一化嵌入
```

:::{note}
`tensor_format` 与 `src_format` 的组合 **不支持** RGB ↔ BGR 通道互转, 编译后的模型中实际不做通道重排. 色彩空间转换仅在 {ref}`YUV 输入 <yuv_input_config>` 场景中使用.
:::

(yuv_input_config)=

## YUV 输入

摄像头通常输出 NV12/NV21 等 YUV 格式. `Pulsar2` 支持在模型中嵌入 YUV → RGB/BGR 色彩空间转换, 避免运行时额外开销.

### NV12 (YUV420SP)

```json
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
      "src_format": "YUV420SP",
      "src_dtype": "U8",
      "src_layout": "NHWC",
      "csc_mode": "FullRange"
    }
  ],
  "compiler": {
    "check": 0
  }
}
```

### NV21 (YVU420SP)

将 `src_format` 改为 `YVU420SP` 即可:

```json
{
  "input_processors": [
    {
      "tensor_name": "input",
      "tensor_format": "BGR",
      "src_format": "YVU420SP",
      "src_dtype": "U8",
      "src_layout": "NHWC",
      "csc_mode": "FullRange"
    }
  ]
}
```

### YUYV422

```json
{
  "input_processors": [
    {
      "tensor_name": "input",
      "tensor_format": "BGR",
      "src_format": "YUYV422",
      "src_dtype": "U8",
      "src_layout": "NHWC",
      "csc_mode": "LimitedRange"
    }
  ]
}
```

### 参数说明

```{eval-rst}
.. list-table::
   :header-rows: 1

   * - 参数
     - 说明
     - 可选值
   * - ``src_format``
     - 运行时输入的 YUV 格式
     - ``YUV420SP`` (NV12), ``YVU420SP`` (NV21), ``YUYV422``, ``UYVY422``
   * - ``tensor_format``
     - 模型期望的色彩空间
     - ``BGR``, ``RGB``
   * - ``csc_mode``
     - 色彩空间转换模式
     - ``FullRange``, ``LimitedRange``, ``Matrix``
```

**csc_mode 说明:**

- `FullRange`: Full Range YUV 转换系数, 适用于大多数摄像头
- `LimitedRange`: Limited Range (BT.601/BT.709) 系数, 适用于视频流
- `Matrix`: 用户自定义 3×4 转换矩阵, 通过 `csc_mat` 字段配置

**自定义 CSC 矩阵:**

```json
{
  "input_processors": [
    {
      "tensor_name": "input",
      "tensor_format": "BGR",
      "src_format": "YUV420SP",
      "src_dtype": "U8",
      "src_layout": "NHWC",
      "csc_mode": "Matrix",
      "csc_mat": [1.164, 0.0, 1.596, -0.871,
                  1.164, -0.392, -0.813, 0.529,
                  1.164, 2.017, 0.0, -1.082]
    }
  ]
}
```

:::{warning}
- 配置 YUV 输入后, `src_layout` 会自动变更为 `NHWC`
- NV12/NV21 输入时 shape 的高度为原始高度的 1.5 倍 (Y + UV 平面)
- `csc_mat` 中 bias (索引 3, 7, 11) 的数值范围为 (-9, 8), 其余参数范围为 (-524289, 524288)
- 上板测试精度时, 如果 `src_format` 为 YUV 格式, 建议使用 **IVE TDP 做 resize**, 该预处理与 OpenCV bilinear 插值对齐
:::

(static_batch_config)

## 静态 Batch 配置

编译器按照用户指定的 batch 组合进行编译, batch 之间复用权重数据, 模型体积远小于各 batch 模型之和.

**配置文件方式** — 在 `compiler` 中添加 `static_batch_sizes`:

```json
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
    "calibration_method": "MinMax"
  },
  "input_processors": [
    {
      "tensor_name": "input",
      "tensor_format": "BGR",
      "src_format": "BGR",
      "src_dtype": "U8",
      "src_layout": "NHWC"
    }
  ],
  "compiler": {
    "check": 0,
    "static_batch_sizes": [1, 2, 4]
  }
}
```

**命令行方式:**

```shell
pulsar2 build --target_hardware AX650 --input model.onnx --output_dir output --config config.json --compiler.static_batch_sizes 1 2 4
```

:::{hint}
以 mobilenetv2 为例, 原始输入 shape 为 `[1, 224, 224, 3]`, 配置 `static_batch_sizes` 为 `[1, 2, 4]` 后, 编译产物的输入 shape 变为 `[4, 224, 224, 3]`.
:::

:::{attention}
- 静态 batch 与动态 batch 两种模式 **互斥**, 不可同时配置
- 如果模型包含 `Reshape` 算子, 可能需要通过 {ref}`《常量数据修改》 <const_patch>` 功能将 shape 的 batch 维改为 `-1` 或 `0`
:::

(dynamic_batch_config)

## 动态 Batch 配置

编译器自动推导出 NPU 可高效运行且不大于 `max_dynamic_batch_size` 的 batch 组合, 运行时根据实际 batch 大小自动拆分多次推理.

**配置文件方式** — 在 `compiler` 中添加 `max_dynamic_batch_size`:

```json
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
    "calibration_method": "MinMax"
  },
  "input_processors": [
    {
      "tensor_name": "input",
      "tensor_format": "BGR",
      "src_format": "BGR",
      "src_dtype": "U8",
      "src_layout": "NHWC"
    }
  ],
  "compiler": {
    "check": 0,
    "max_dynamic_batch_size": 4
  }
}
```

**命令行方式:**

```shell
pulsar2 build --target_hardware AX650 --input model.onnx --output_dir output --config config.json --compiler.max_dynamic_batch_size 4
```

**推导规则:**

- 编译器从 batch 1 开始, 2 倍递增 (1 → 2 → 4 → ...), 当 batch 超过设定值或理论推理效率下降时停止
- 理论推理效率 = 理论推理耗时 / batch_size

:::{hint}
设置 `max_dynamic_batch_size` 为 4 时, 编译产物可能包含 [1, 2, 4] 三个 batch.

运行时推理框架会自动拆分:

- batch=3 → 内部执行 batch 2 + batch 1 共两次推理
- batch=9 → 内部执行 batch 4 + batch 4 + batch 1 共三次推理
:::

(multi_input_config)

## 多输入配置

当 ONNX 模型包含多个输入时 (如双目视觉、图像+掩码、多传感器融合等), 需要为每个输入分别配置 `input_configs` 和 `input_processors`.

```json
{
  "model_type": "ONNX",
  "npu_mode": "NPU1",
  "quant": {
    "input_configs": [
      {
        "tensor_name": "rgb_image",
        "calibration_dataset": "./dataset/rgb_images.tar",
        "calibration_size": 32,
        "calibration_mean": [103.939, 116.779, 123.68],
        "calibration_std": [58.0, 58.0, 58.0]
      },
      {
        "tensor_name": "depth_map",
        "calibration_dataset": "./dataset/depth_maps.tar",
        "calibration_format": "Numpy",
        "calibration_size": 32,
        "calibration_mean": [0],
        "calibration_std": [1]
      }
    ],
    "calibration_method": "MinMax",
    "precision_analysis": false
  },
  "input_processors": [
    {
      "tensor_name": "rgb_image",
      "tensor_format": "BGR",
      "src_format": "BGR",
      "src_dtype": "U8",
      "src_layout": "NHWC"
    },
    {
      "tensor_name": "depth_map",
      "tensor_format": "GRAY",
      "src_format": "GRAY",
      "src_dtype": "FP32",
      "src_layout": "NCHW"
    }
  ],
  "compiler": {
    "check": 0
  }
}
```

**配置要点:**

- 每个输入需要独立的 `input_configs` 和 `input_processors` 条目
- 不同输入可以使用不同的校准数据集、数据格式和归一化参数
- `calibration_format` 支持 `Image` (默认)、`Numpy`、`Binary`、`NumpyObject` 四种格式
- 如果所有输入使用相同配置, 可将 `tensor_name` 设置为 `DEFAULT`

**校准数据对应关系:**

```{eval-rst}
.. list-table::
   :header-rows: 1

   * - 输入 Tensor
     - 校准数据格式
     - 数据集内容
     - 说明
   * - rgb_image
     - Image (默认)
     - JPEG/PNG 打包为 tar
     - 工具链自动读取图片并归一化
   * - depth_map
     - Numpy
     - .npy 文件打包为 tar
     - 需预处理为与模型输入 shape 一致的 numpy 数组
```

:::{warning}
`tensor_name` 必须与 ONNX 模型中的实际输入名称一致. 仿真运行时, 需为每个输入分别准备 bin 文件, 文件名需与 tensor 名称匹配.
:::

(skip_onnxsim_config)

## 跳过 onnxslim (onnxsim)

`pulsar2 build` 默认会先用开源的 `onnxslim` 工具对 ONNX 模型执行内部图优化. 在某些场景下 (如模型已手动优化、包含自定义算子、图优化导致编译失败等), 可能需要跳过这些优化步骤.

**命令行方式:**

```shell
pulsar2 build --target_hardware AX650 --input model.onnx --output_dir output --config config.json --onnx_opt.disable_onnx_optimization true
```

**配置文件方式** — 在顶层添加 `onnx_opt`:

```json
{
  "model_type": "ONNX",
  "npu_mode": "NPU1",
  "onnx_opt": {
    "disable_onnx_optimization": true
  },
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
    "calibration_method": "MinMax"
  },
  "input_processors": [
    {
      "tensor_name": "input",
      "tensor_format": "BGR",
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
