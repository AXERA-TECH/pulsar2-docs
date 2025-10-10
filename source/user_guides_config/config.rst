.. _config_details:

============================
配置文件详细说明
============================

本节将对 ``pulsar2 build`` 中的 **config** 文件进行详细介绍.

------------------------------------
配置文件概述
------------------------------------

- 工具链支持的全部编译参数定义请参考 :ref:`《proto 配置定义》 <config_define>` ，基础数据结构为 ``BuildConfig``；

- 用户可以根据参数规格编写 ``prototxt / 宽松 json / yaml / toml`` 格式的配置文件，通过命令行参数 ``--config`` 指向配置文件；
  
    - 宽松的 ``json`` 格式：支持包含 ``js-style`` 或者 ``python-style`` 注释的 ``json`` 文件；

- 部分编译参数支持命令行传入，且优先级高于配置文件，通过 ``pulsar2 build -h`` 查看支持的命令行编译参数，比如命令行参数 ``--quant.calibration_method`` 相当于配置了 ``QuantConfig`` 结构体的 ``calibration_method`` 字段.

------------------------------------
完整的 json 配置参考
------------------------------------

.. code-block:: json

    {
      // input model file path. type: string. required: true.
      "input": "/path/to/lenet5.onnx",
      // axmodel output directory. type: string. required: true.
      "output_dir": "/path/to/output_dir",
      // rename output axmodel. type: string. required: false. default: compiled.axmodel.
      "output_name": "compiled.axmodel",
      // temporary data output directory. type: string. required: false. default: same with ${output_dir}.
      "work_dir": "",
      // input model type. type: enum. required: false. default: ONNX. option: ONNX, QuantAxModel, QuantONNX.
      "model_type": "ONNX",
      // target hardware. type: enum. required: false. default: AX650. option: AX650, AX620E, M76H, M57.
      "target_hardware": "AX650",
      // npu mode. while ${target_hardware} is AX650, npu mode can be NPU1 / NPU2 / NPU3. while ${target_hardware} is AX620E, npu mode can be NPU1 / NPU2. type: enum. required: false. default: NPU1.
      "npu_mode": "NPU1",
      // modify model input shape of input model, this feature will take effect before the `input_processors` configuration. format: input1:1x3x224x224;input2:1x1x112x112. type: string. required: false. default: .
      "input_shapes": "input:1x1x28x28",
      "onnx_opt": {
        // disable onnx optimization. type: bool. required: false. default: false.
        "disable_onnx_optimization": false,
        // enable onnx simplify by https://github.com/daquexian/onnx-simplifier. type: bool. required: false. default: false.
        "enable_onnxsim": false,
        // enable model check. type: bool. required: false. default: false.
        "model_check": false,
        // disable transformation check. type: bool. required: false. default: false.
        "disable_transformation_check": false,
        // save tensors data to optimize memory footprint. type: bool. required: false. default: false.
        "save_tensors_data": false
      },
      "quant": {
        "input_configs": [
          {
            // input tensor name in origin model. "DEFAULT" means input config for all input tensors. type: string. required: true.
            "tensor_name": "input",
            // quantize calibration dataset archive file path. type: string. required: true. limitation: tar, tar.gz, zip.
            "calibration_dataset": "/path/to/dataset",
            // quantize calibration data format. type: enum. required: false. default: Image. option: Image, Numpy, Binary, NumpyObject.
            "calibration_format": "Image",
            // quantize calibration data size is min(${calibration_size}, size of ${calibration_dataset}), "-1" means load all dataset. type: int. required: false. default: 32.
            "calibration_size": 32,
            // quantize mean parameter of normlization. type: float array. required: false. default: [].
            "calibration_mean": [127],
            // quantize std parameter of normlization. type: float array. required: false. default: [].
            "calibration_std": [1]
          }
        ],
        "layer_configs": [
          {
            // set layer quantize precision. type: string. required: must choose between `layer_name` and `op_type` and `layer_names` and `op_types`. default: .
            "layer_name": "Conv_0",
            // quantize data type. type: enum. required: false. default: U8. option: U8, S8, U16, S16, FP32.
            "data_type": "U8",
            // quantize data type for Conv. type: enum. required: false. default: U8. option: U8, S8, U16, S16, FP32.
            "output_data_type": "U8",
            // quantize weight type for Conv. type: enum. required: false. default: S8. option: S8, FP32.
            "weight_data_type": "S8"
          },
          {
            // set quantize precision by operator type. type: string. required: must choose between `layer_name` and `op_type` and `layer_names` and `op_types`. default: .
            "op_type": "MaxPool",
            // quantize data type. type: enum. required: false. default: U8. option: U8, S8, U16, S16, FP32.
            "data_type": "U8"
          },
          {
            // set layer quantize precision by layers name. type: enum. required: must choose between `layer_name` and `op_type` and `layer_names` and `op_types`. default: [].
            "layer_names": ["Conv_2"],
            // quantize data type. type: enum. required: false. default: U8. option: U8, S8, U16, S16, FP32.
            "data_type": "U8",
            // quantize data type for Conv. type: enum. required: false. default: U8. option: U8, S8, U16, S16, FP32.
            "output_data_type": "U8",
            // quantize weight type for Conv. type: enum. required: false. default: S8. option: S8, FP32.
            "weight_data_type": "S8"
          },
          {
            // set quantize precision by operator types. type: enum. required: must choose between `layer_name` and `op_type` and `layer_names` and `op_types`. default: [].
            "op_types": ["Gemm"],
            // quantize data type. type: enum. required: false. default: U8. option: U8, S8, U16, S16, FP32.
            "data_type": "U8"
          },
          {
            // start tensor names of subgraph quantization config. type: string array. required: false. default: [].
            "start_tensor_names": ["13"],
            // end tensor names of subgraph quantization config. type: string array. required: false. default: [].
            "end_tensor_names": ["15"],
            // quantize data type. type: enum. required: false. default: U8. option: U8, S8, U16, S16, FP32.
            "data_type": "U16"
          }
        ],
        // quantize calibration method. type: enum. required: false. default: MinMax. option: MinMax, Percentile, MSE, KL.
        "calibration_method": "MinMax",
        // enable quantization precision analysis. type: bool. required: false. default: false.
        "precision_analysis": false,
        // precision analysis method. type: enum. required: false. default: PerLayer. option: PerLayer, EndToEnd.
        "precision_analysis_method": "PerLayer",
        // precision analysis mode. type: enum. required: false. default: Reference. option: Reference, NPUBackend.
        "precision_analysis_mode": "Reference",
        // input sample data dir for precision analysis. type: string. required: false. default: .
        "input_sample_dir": "",
        // enable highest mix precision quantization. type: bool. required: false. default: false.
        "highest_mix_precision": false,
        // conv bias data type. type: enum. required: false. default: S32. option: S32, FP32.
        "conv_bias_data_type": "S32",
        // LayerNormalization scale data type. type: enum. required: false. default: FP32. option: FP32, S32, U32.
        "ln_scale_data_type": "FP32",
        // refine weight threshold, should be a legal float number, like 1e-6. -1 means disable this feature. type: float. required: false. default: 1e-6. limitation: 0 or less than 0.0001.
        "refine_weight_threshold": 1e-6,
        // enalbe smooth quant strategy. type: bool. required: false. default: false.
        "enable_smooth_quant": false,
        // smooth quant threshold. The larger the threshold, the more operators will be involved in performing SmoothQuant. limitation: 0~1.
        "smooth_quant_threshold": 2e-1,
        // smooth quant strength, a well-balanced point to evenly split the quantization difficulty.
        "smooth_quant_strength": 6e-1,
        // tranformer opt level. type: int. required: false. default: 0. limitation: 0~2.
        "transformer_opt_level": 0,
        // quant check level, 0: no check; 1: check node dtype. type: int. required: false. default: 0.
        "check": 0,
        // refine weight scale and input scale, type: bool. required: false. default: false.
        "disable_auto_refine_scale": false,
        // enable easyquant; type bool. required: false. default: false.
        "enable_easy_quant": false,
        // disable quant optimization; type bool. required: false. default: false.
        "disable_quant_optimization": false,
        // enable brecq quantize strategy; type bool. required: false. default: false.
        "enable_brecq": false,
        // enable lsq quantize strategy; type bool. required: false. default: false.
        "enable_lsq": false,
        // enable adaround quantize strategy; type bool. required: false. default: false.
        "enable_adaround": false,
        // finetune epochs when enable finetune algorithm; type int32. required: false. default: 500.
        "finetune_epochs": 500,
        // finetune split block size when enable finetune algorithm; type int32. required: false. default: 4.
        "finetune_block_size": 4,
        // finetune batch size when enable finetune algorithm; type int32. required: false. default: 1.
        "finetune_batch_size": 1,
        // learning rate when enable finetune algorithm; type float. required: false. default: 1e-3.
        "finetune_lr": 1e-3,
        // device for quant calibration. type: string. required: false. default: cpu. option: cpu, cuda:0, cuda:1, ..., cuda:7.
        "device": "cpu"
      },
      "input_processors": [
        {
          // input tensor name in origin model. "DEFAULT" means processor for all input tensors. type: string. required: true.
          "tensor_name": "input",
          // input tensor format in origin model. type: enum. required: false. default: AutoColorSpace. option: AutoColorSpace, BGR, RGB, GRAY.
          "tensor_format": "AutoColorSpace",
          // input tensor layout in origin model. type: enum. required: false. default: NCHW. option: NHWC, NCHW.
          "tensor_layout": "NCHW",
          // input format in runtime. type: enum. required: false. default: AutoColorSpace. option: AutoColorSpace, GRAY, BGR, RGB, YUYV422, UYVY422, YUV420SP, YVU420SP, RAW.
          "src_format": "AutoColorSpace",
          // input layout in runtime; if `src_format` is YUV/YVU, `src_layout` will be changed to NHWC. type: enum. required: false. default: NCHW. option: NHWC, NCHW.
          "src_layout": "NHWC",
          // input data type in runtime. type: enum. required: false. default: FP32. option: U8, S8, U16, S16, U32, S32, FP16, FP32.
          "src_dtype": "U8",
          // extra compiler shapes for this input. src_extra_shapes size of every input should be the same. shape at the same index of every input will be treated as a input group which can inference independently at runtime. type: list of Shape. required: false. default [].
          "src_extra_shapes": [],
          // color space mode. type: enum. required: false. default: NoCSC. option: NoCSC, Matrix, FullRange, LimitedRange.
          "csc_mode": "NoCSC",
          // color space conversion matrix, 12 elements array that represents a 3x4 matrix. type: float array. required: false. default: [].
          "csc_mat": [1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4],
          // mean parameter of normlization in runtime. type: float array. required: false. default: same with ${quant.input_configs.calibration_mean}.
          "mean": [],
          // std parameter of normlization in runtime. type: float array. required: false. default: same with ${quant.input_configs.calibration_std}.
          "std": [],
          // list containing the number of start and end pad values for axis when padding. type: int32 array. required: false. default: [].
          "padding": [],
          // padding mode. type: string. required: false. default: constant.
          "padding_mode": "constant",
          // padding constant value. type: int32. required: false. default: 0.
          "padding_constant_value": 0,
          // list containing the number of start and end pad values for axis when slicing. type: int32 array. required: false. default: [].
          "slicing": []
        }
      ],
      "output_processors": [
        {
          // output tensor name in origin model. "DEFAULT" means processor for all output tensors. type: string. required: true.
          "tensor_name": "output",
          // permute the output tensor. type: int32 array. required: false. default: [].
          "dst_perm": [0, 1],
          // output data type. type: enum. required: false. default: FP32. option: FP32, U8.
          "output_dtype": "FP32"
        }
      ],
      "const_processors": [
        {
          // const tensor name in origin model. type: string. required: true.
          "name": "fc2.bias",
          // const tensor data array. type: list of double. required: false.
          "data": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
          // const tensor data file path, support .bin / .npy / .txt. type: string. required: false.
          "data_path": "replaced_data_file_path"
        }
      ],
      "quant_op_processors": [
        {
          // operator name in origin model. type: string. required: true.
          "op_name": "MaxPool_3",
          // operator attributes to be patched. type: dict. default: {}. required: true.
          "attrs": {
            "ceil_mode": 0
          }
        },
        {
          "op_name": "Flatten_4", // AxReshape
          "attrs": {
            "shape": [0, 800]
          }
        }
      ],
      "compiler": {
        // static batch sizes. type: int array. required: false. default: [].
        "static_batch_sizes": [],
        // max dynamic batch. type: int, required: false. default: 0.
        "max_dynamic_batch_size": 0,
        // ddr bandwidth limit in GB, 0 means no limit. type: int. required: false. default: 0.
        "ddr_bw_limit": 0,
        // disable ir fix, only work in multi-batch compilation. type: bool. required: false. default: false.
        "disable_ir_fix": false,
        // compiler check level, 0: no check; 1: assert all close; 2: assert all equal; 3: check cosine simularity. type: int. required: false. default: 0.
        "check": 0,
        // dump npu perf information for profiling. type: bool. required: false. default: false.
        "npu_perf": false,
        // compiler check mode, CheckOutput: only check model output; CheckPerLayer: check model intermediate tensor and output. type: enum. required: false. default: CheckOutput. option: CheckOutput, CheckPerLayer.
        "check_mode": "CheckOutput",
        // relative tolerance when check level is 1. type: float. required: false. default: 1e-5.
        "check_rtol": 1e-5,
        // absolute tolerance when check level is 1. type: float. required: false. default: 0.
        "check_atol": 0,
        // cosine simularity threshold when check level is 3. type: float. required: false. default: 0.999.
        "check_cosine_simularity": 0.999,
        // tensor black list for per layer check, support regex. type: list of string. required: false. default: [].
        "check_tensor_black_list": [],
        // input sample data dir for compiler check. type: string. required: false. default: .
        "input_sample_dir": "",
        // enable slice mode scheduler. type: bool. required: false. default: false.
        "enable_slice_mode": false,
        // enable tile mode scheduler. type: bool. required: false. default: false.
        "enable_tile_mode": false,
        // enable data soft compression. type: bool. required: false. default: false.
        "enable_data_soft_compression": false
      }
    }


.. _config_define:

------------------------------------
量化参数说明
------------------------------------

- ``input_configs`` 中的 ``tensor_name`` 需要根据模型的实际输入/输出节点名称进行设置。
- ``input_configs`` 中的 ``tensor_name`` 可以设置为 ``DEFAULT`` 代表量化配置应用于全部输入。
- 模型输入的色彩空间由预处理 ``input_processors`` 配置中的 ``tensor_format`` 参数来表达。
- 工具链读取量化校准集时，会根据 ``input_processors`` 中的 ``tensor_format`` 参数自动转换校准集数据的色彩空间。
- ``layer_configs`` 中的 ``layer_name`` 及 ``op_type`` 选项不可以同时配置。
- ``transformer_opt_level`` 设置 ``Transformer`` 模型的优化选项。

.. _quant_precision_analysis_config_define:

------------------------------------
量化精度分析参数说明
------------------------------------

- 精度分析计算方法，``precision_analysis_mode`` 字段。

    - ``Reference`` 可以运行编译器支持的全部模型（支持包含 CPU 及 NPU 子图的模型），但是计算结果相比于最终上板结果会有少量误差（基本上差距在正负 1 内，且无系统性误差）。
    - ``NPUBackend`` 可以运行仅包含 NPU 子图的模型，但是计算结果与上板结果比特对齐。

- 精度分析方法，``precision_analysis_method`` 字段。

    - ``PerLayer`` 意味着每一层都采用浮点模型对应的层输入，计算每一层的输出与浮点模型输出的相似度。
    - ``EndToEnd`` 代表首层采用浮点模型输入，然后进行完整模型的仿真，计算最终输出结果与浮点模型输出的相似度。


.. _processing_arg_details:

------------------------------------
预处理、后处理参数说明
------------------------------------

- ``input_processors`` / ``output_processors`` 配置说明

    - ``tensor_name`` 需要根据模型的实际输入/输出节点名称进行设置。
    - ``tensor_name`` 可以设置为 ``DEFAULT`` 代表配置应用于全部输入或者输出。
    - 前缀为 ``tensor_`` 的参数代表原始模型中的输入输出属性。
    - 前缀为 ``src_`` 的参数代表着运行时实际的输入输出属性。
    - 工具链会根据用户的配置自动添加算子，以完成运行时输入输出与原始模型输入输出之间的转换。

        - 例如：当 ``tensor_layout`` 为 ``NCHW``，且 ``src_layout`` 为 ``NHWC`` 时，工具链会在原始模型输入之前自动添加一个 ``perm`` 属性为 [0, 3, 1, 2] 的 ``Transpose`` 算子。

- 色彩空间转换预处理

    - 当 ``csc_mode`` 为 ``LimitedRange`` 或者 ``FullRange`` 且 ``src_format`` 为 ``YUV 色彩空间`` 时，工具链会根据内置的模板参数，在原始的输入前添加一个色彩空间转换算子，此时 ``csc_mat`` 配置无效；
    - 当 ``csc_mode`` 为 ``Matrix`` 且 ``src_format`` 为 ``YUV 色彩空间`` 时，工具链会根据用户配置的 ``csc_mat`` 矩阵，在原始的输入前添加一个色彩空间转换算子，以实现在运行时将输入的 ``YUV`` 数据转换为模型计算所需的 ``BGR`` 或者 ``RGB`` 数据；
    - 当 ``csc_mode`` 为 ``Matrix`` 时，计算流程为，先将 ``YUV / YVU 色彩空间`` 输入统一转换为 ``YUV444`` 格式，然后再乘以 ``csc_mat`` 系数矩阵。
    - 当 ``csc_mode`` 为 ``Matrix`` 时，``bias`` (csc_mat[3] / csc_mat[7] / csc_mat[11]) 数值范围为 (-9, 8)。其余参数 (csc_mat[0-2] / csc_mat[4-6] / csc_mat[8-10]) 数值范围为 (-524289, 524288)。

- 归一化预处理

    - ``input_processors`` 中的 ``mean`` / ``std`` 参数，默认为用户在量化配置中 ``calibration_mean`` / ``calibration_std`` 参数所配置的值。
    - 如果用户希望在运行时采用不同的归一化参数，那么可以显示的配置 中的 ``mean`` / ``std`` 参数以覆盖默认值。

- 数据预处理中的填充 (Pad) 和切片 (Slice) 操作

    配置示例:

    .. code-block:: shell

        {
          ...
          "input_processors": [
            {
              "slicing": [0, 0, 0, 0, 0, 1, 0, 1]
            }
          ],
          ...
        }

    - ``padding`` 此字段表示在数据预处理对特定轴进行填充时，每个轴的开始和结束部分应填充的长度。以 32 位整型数组的形式表示，如果未设置，则使用默认值，即空列表，表示不进行填充。
    - ``padding_mode`` 这个字段指定了填充的模式。它是一个字符串类型，可能的值决定了填充值的生成方式。默认值为 "constant"，表示使用常数值进行填充。目前仅支持 "constant" 模式填充。
    - ``padding_constant_value`` 此字段指定了在填充模式为 "constant" 时使用的常数值。它是一个 32 位整型。表示用于填充的固定值。默认值值为 0。
    - ``slicing`` 此字段表示在数据预处理对特定轴进行切片时，每个轴的开始和结束部分应切片的长度。以 32 位整型数组的形式表示的，如果未设置，则使用默认值，即空列表，表示不进行切片。

------------------------------------
proto 配置定义
------------------------------------

.. code-block:: shell

    syntax = "proto3";
    
    package common;
    
    enum ColorSpace {
      AutoColorSpace = 0;
      GRAY = 1;
      BGR = 2;
      RGB = 3;
      RGBA = 4;
      YUV420SP = 6;   // Semi-Planner, NV12
      YVU420SP = 7;   // Semi-Planner, NV21
      YUYV422 = 8;     // Planner, YUYV
      UYVY422 = 9;     // Planner, UYVY
      RAW = 10;       // Planner, BayerBGGR
    }
    
    enum Layout {
      DefaultLayout = 0;
      NHWC = 1;
      NCHW = 2;
    }
    
    message Shape {
      repeated int32 shape = 1;
    }
    
    enum DataType {
      DefaultDataType = 0;
      U8 = 1;
      S8 = 2;
      U16 = 3;
      S16 = 4;
      U32 = 5;
      S32 = 6;
      U64 = 7;
      S64 = 8;
      FP16 = 9;
      FP32 = 10;
      BF16 = 11;
    }
    
    enum NPUMode {
      NPU1 = 0;
      NPU2 = 1;
      NPU3 = 2;
    }
    
    enum HardwareType {
      AX650 = 0;
      AX620E = 1;
      M76H = 2;
      M57 = 5;
    }

.. code-block:: shell

    syntax = "proto3";
    
    import "path/to/common.proto";
    import "google/protobuf/struct.proto";
    
    package pulsar2.build;

    enum ModelType {
      ONNX = 0;
      QuantAxModel = 1;
      QuantONNX = 3;
      OptimizedQuantAxModel = 4;
    }
    
    enum QuantMethod {
      MinMax = 0;
      Percentile = 1;
      MSE = 2;
      KL = 3;
    }
    
    enum PrecisionAnalysisMethod {
      PerLayer = 0;
      EndToEnd = 1;
    }
    
    enum PrecisionAnalysisMode {
      Reference = 0;
      NPUBackend = 1;
    }
    
    enum CheckMode {
      CheckOutput = 0;
      CheckPerLayer = 1;
    }
    
    enum DataFormat {
      Image = 0;
      Numpy = 1;
      Binary = 2;
      NumpyObject = 3;
    }
    
    enum CSCMode {
      NoCSC = 0;
      Matrix = 1;
      FullRange = 2;
      LimitedRange = 3;
    }
    
    enum ScheduleStrategy {
      Tile = 0;
      Slice = 1;
    }
    
    enum MatchMode {
      Op = 0;
      Tensor = 1;
    }
    
    message Node {
      repeated string types = 1;
      repeated string inputs = 2;
      repeated string outputs = 3;
      .google.protobuf.Struct properties = 4;
    }
    
    message Graph {
      repeated Node nodes = 1;
      ScheduleStrategy type = 2;
      MatchMode match_mode = 3;
    }
    
    message InputQuantConfig {
      // input tensor name in origin model. "DEFAULT" means input config for all input tensors. type: string. required: true.
      string tensor_name = 1;
      // quantize calibration dataset archive file path. type: string. required: true. limitation: tar, tar.gz, zip.
      string calibration_dataset = 2;
      // quantize calibration data format. type: enum. required: false. default: Image. option: Image, Numpy, Binary, NumpyObject.
      DataFormat calibration_format = 3;
      // quantize calibration data size is min(${calibration_size}, size of ${calibration_dataset}), "-1" means load all dataset. type: int. required: false. default: 32.
      int32 calibration_size = 4;
      // quantize mean parameter of normlization. type: float array. required: false. default: [].
      repeated float calibration_mean = 5;
      // quantize std parameter of normlization. type: float array. required: false. default: [].
      repeated float calibration_std = 6;
    }
    
    message LayerConfig {
      // set layer quantize precision. type: string. required: must choose between `layer_name` and `op_type` and `layer_names` and `op_types`. default: .
      string layer_name = 1;
    
      // set quantize precision by operator type. type: string. required: must choose between `layer_name` and `op_type` and `layer_names` and `op_types`. default: .
      string op_type = 2;
    
      // start tensor names of subgraph quantization config. type: string array. required: false. default: [].
      repeated string start_tensor_names = 3;
      // end tensor names of subgraph quantization config. type: string array. required: false. default: [].
      repeated string end_tensor_names = 4;
    
      // quantize data type. type: enum. required: false. default: U8. option: U8, S8, U16, S16, FP32.
      common.DataType data_type = 5;
    
      // quantize weight type for Conv. type: enum. required: false. default: S8. option: S8, FP32.
      common.DataType weight_data_type = 6;
    
      // set layer quantize precision by layers name. type: enum. required: must choose between `layer_name` and `op_type` and `layer_names` and `op_types`. default: [].
      repeated string layer_names = 7;
    
      // set quantize precision by operator types. type: enum. required: must choose between `layer_name` and `op_type` and `layer_names` and `op_types`. default: [].
      repeated string op_types = 8;
    
      // quantize data type for Conv. type: enum. required: false. default: U8. option: U8, S8, U16, S16, FP32.
      common.DataType output_data_type = 10;
    }
    
    message OnnxOptimizeOption {
      // disable onnx optimization. type: bool. required: false. default: false.
      bool disable_onnx_optimization = 1;
      // enable onnx simplify by https://github.com/daquexian/onnx-simplifier. type: bool. required: false. default: false.
      bool enable_onnxsim = 2;
      // enable model check. type: bool. required: false. default: false.
      bool model_check = 3;
      // disable transformation check. type: bool. required: false. default: false.
      bool disable_transformation_check = 4;
      // save tensors data to optimize memory footprint. type: bool. required: false. default: false.
      bool save_tensors_data = 5;
    }
    
    message QuantConfig {
      repeated InputQuantConfig input_configs = 1;
      repeated LayerConfig layer_configs = 2;
    
      // quantize calibration method. type: enum. required: false. default: MinMax. option: MinMax, Percentile, MSE, KL.
      QuantMethod calibration_method = 3;
      // enable quantization precision analysis. type: bool. required: false. default: false.
      bool precision_analysis = 4;
      // precision analysis method. type: enum. required: false. default: PerLayer. option: PerLayer, EndToEnd.
      PrecisionAnalysisMethod precision_analysis_method = 5;
      // precision analysis mode. type: enum. required: false. default: Reference. option: Reference, NPUBackend.
      PrecisionAnalysisMode precision_analysis_mode = 6;
      // enable highest mix precision quantization. type: bool. required: false. default: false.
      bool highest_mix_precision = 7;
      // conv bias data type. type: enum. required: false. default: S32. option: S32, FP32.
      common.DataType conv_bias_data_type = 8;
      // refine weight threshold, should be a legal float number, like 1e-6. -1 means disable this feature. type: float. required: false. default: 1e-6. limitation: 0 or less than 0.0001.
      float refine_weight_threshold = 9;
      // enalbe smooth quant strategy. type: bool. required: false. default: false.
      bool enable_smooth_quant = 10;
      // smooth quant threshold. The larger the threshold, the more operators will be involved in performing SmoothQuant. limitation: 0~1.
      float smooth_quant_threshold = 20;
      // smooth quant strength, a well-balanced point to evenly split the quantization difficulty.
      float smooth_quant_strength = 30;
      // tranformer opt level. type: int. required: false. default: 0. limitation: 0~2.
      int32 transformer_opt_level = 40;
      // input sample data dir for precision analysis. type: string. required: false. default: .
      string input_sample_dir = 50;
      // LayerNormalization scale data type. type: enum. required: false. default: FP32. option: FP32, S32, U32.
      common.DataType ln_scale_data_type = 60;
      // quant check level, 0: no check; 1: check node dtype. type: int. required: false. default: 0.
      int32 check = 70;
      // refine weight scale and input scale, type: bool. required: false. default: false.
      bool disable_auto_refine_scale = 80;
      // enable easyquant; type bool. required: false. default: false.
      bool enable_easy_quant = 90;
      // disable quant optimization; type bool. required: false. default: false.
      bool disable_quant_optimization = 100;
      // enable brecq quantize strategy; type bool. required: false. default: false.
      bool enable_brecq = 110;
      // enable lsq quantize strategy; type bool. required: false. default: false.
      bool enable_lsq = 120;
      // enable adaround quantize strategy; type bool. required: false. default: false.
      bool enable_adaround = 130;
      // finetune epochs when enable finetune algorithm; type int32. required: false. default: 500.
      int32 finetune_epochs = 140;
      // finetune split block size when enable finetune algorithm; type int32. required: false. default: 4.
      int32 finetune_block_size = 150;
      // finetune batch size when enable finetune algorithm; type int32. required: false. default: 1.
      int32 finetune_batch_size = 160;
      // learning rate when enable finetune algorithm; type float. required: false. default: 1e-3.
      float finetune_lr = 170;
      // device for quant calibration. type: string. required: false. default: cpu. option: cpu, cuda:0, cuda:1, ..., cuda:7.
      string device = 180;
    }
    
    message InputProcessor {
      // input tensor name in origin model. "DEFAULT" means processor for all input tensors. type: string. required: true.
      string tensor_name = 1;
    
      // input tensor format in origin model. type: enum. required: false. default: AutoColorSpace. option: AutoColorSpace, BGR, RGB, GRAY.
      common.ColorSpace tensor_format = 2;
      // input tensor layout in origin model. type: enum. required: false. default: NCHW. option: NHWC, NCHW.
      common.Layout tensor_layout = 3;
    
      // input format in runtime. type: enum. required: false. default: AutoColorSpace. option: AutoColorSpace, GRAY, BGR, RGB, YUYV422, UYVY422, YUV420SP, YVU420SP, RAW.
      common.ColorSpace src_format = 4;
      // input layout in runtime; if `src_format` is YUV/YVU, `src_layout` will be changed to NHWC. type: enum. required: false. default: NCHW. option: NHWC, NCHW.
      common.Layout src_layout = 5;
      // input data type in runtime. type: enum. required: false. default: FP32. option: U8, S8, U16, S16, U32, S32, FP16, FP32.
      common.DataType src_dtype = 6;
    
      // extra compiler shapes for this input. src_extra_shapes size of every input should be the same. shape at the same index of every input will be treated as a input group which can inference independently at runtime. type: list of Shape. required: false. default [].
      repeated common.Shape src_extra_shapes = 11;
    
      // color space mode. type: enum. required: false. default: NoCSC. option: NoCSC, Matrix, FullRange, LimitedRange.
      CSCMode csc_mode = 7;
      // color space conversion matrix, 12 elements array that represents a 3x4 matrix. type: float array. required: false. default: [].
      repeated float csc_mat = 8;
      // mean parameter of normlization in runtime. type: float array. required: false. default: same with ${quant.input_configs.calibration_mean}.
      repeated float mean = 9;
      // std parameter of normlization in runtime. type: float array. required: false. default: same with ${quant.input_configs.calibration_std}.
      repeated float std = 10;
      // list containing the number of start and end pad values for axis when padding. type: int32 array. required: false. default: [].
      repeated int32 padding = 20;
      // padding mode. type: string. required: false. default: constant.
      string padding_mode = 21;
      // padding constant value. type: int32. required: false. default: 0.
      int32 padding_constant_value = 22;
      // list containing the number of start and end pad values for axis when slicing. type: int32 array. required: false. default: [].
      repeated int32 slicing = 30;
    }
    
    message OutputProcessor {
      // output tensor name in origin model. "DEFAULT" means processor for all output tensors. type: string. required: true.
      string tensor_name = 1;
    
      common.Layout tensor_layout = 2;
    
      // permute the output tensor. type: int32 array. required: false. default: [].
      repeated int32 dst_perm = 3;
    
      // output data type. type: enum. required: false. default: FP32. option: FP32, U8.
      common.DataType output_dtype = 4;
    }
    
    message OpProcessor {
      // operator name in origin model. type: string. required: true.
      string op_name = 1;
    
      // operator attributes to be patched. type: dict. default: {}. required: true.
      .google.protobuf.Struct attrs = 2;
    }
    
    message ConstProcessor {
      // const tensor name in origin model. type: string. required: true.
      string name = 1;
    
      // const tensor data array. type: list of double. required: false.
      repeated double data = 2;
    
      // const tensor data file path, support .bin / .npy / .txt. type: string. required: false.
      string data_path = 3;
    }
    
    message CompilerConfig {
      // static batch sizes. type: int array. required: false. default: [].
      repeated int32 static_batch_sizes = 1;
      // max dynamic batch. type: int, required: false. default: 0.
      optional int32 max_dynamic_batch_size = 2;
      // ddr bandwidth limit in GB, 0 means no limit. type: int. required: false. default: 0.
      optional float ddr_bw_limit = 12;
      // disable ir fix, only work in multi-batch compilation. type: bool. required: false. default: false.
      optional bool disable_ir_fix = 3;
      // compiler check level, 0: no check; 1: assert all close; 2: assert all equal; 3: check cosine simularity. type: int. required: false. default: 0.
      optional int32 check = 5;
      // dump npu perf information for profiling. type: bool. required: false. default: false.
      optional bool npu_perf = 6;
      // compiler check mode, CheckOutput: only check model output; CheckPerLayer: check model intermediate tensor and output. type: enum. required: false. default: CheckOutput. option: CheckOutput, CheckPerLayer.
      optional CheckMode check_mode = 7;
      // relative tolerance when check level is 1. type: float. required: false. default: 1e-5.
      optional float check_rtol = 8;
      // absolute tolerance when check level is 1. type: float. required: false. default: 0.
      optional float check_atol = 9;
      // cosine simularity threshold when check level is 3. type: float. required: false. default: 0.999.
      optional float check_cosine_simularity = 10;
      // tensor black list for per layer check, support regex. type: list of string. required: false. default: [].
      repeated string check_tensor_black_list = 11;
      // enable slice mode scheduler. type: bool. required: false. default: false.
      optional bool enable_slice_mode = 13;
      // enable tile mode scheduler. type: bool. required: false. default: false.
      optional bool enable_tile_mode = 52;
      // enable data soft compression. type: bool. required: false. default: false.
      optional bool enable_data_soft_compression = 14;
      // input sample data dir for compiler check. type: string. required: false. default: .
      optional string input_sample_dir = 30;
      repeated Graph compiler_group_patterns = 15;
    
      // sub compiler configs. type: CompilerConfig. required: false. default: [].
      repeated CompilerConfig sub_configs = 60;
      // start tensor names of subgraph compiler config, only can be configured in sub compiler config. type: string array. required: false. default: [].
      repeated string start_tensor_names = 50;
      // end tensor names of subgraph compiler config, only can be configured in sub compiler config. type: string array. required: false. default: [].
      repeated string end_tensor_names = 51;
    }
    
    message BuildConfig {
      // input model file path. type: string. required: true.
      string input = 1;
      // axmodel output directory. type: string. required: true.
      string output_dir = 2;
      // rename output axmodel. type: string. required: false. default: compiled.axmodel.
      string output_name = 3;
      // temporary data output directory. type: string. required: false. default: same with ${output_dir}.
      string work_dir = 4;
    
      // input model type. type: enum. required: false. default: ONNX. option: ONNX, QuantAxModel, QuantONNX.
      ModelType model_type = 5;
    
      // target hardware. type: enum. required: false. default: AX650. option: AX650, AX620E, M76H, M57.
      common.HardwareType target_hardware = 6;
      // npu mode. while ${target_hardware} is AX650, npu mode can be NPU1 / NPU2 / NPU3. while ${target_hardware} is AX620E, npu mode can be NPU1 / NPU2. type: enum. required: false. default: NPU1.
      common.NPUMode npu_mode = 7;
    
      // modify model input shape of input model, this feature will take effect before the `input_processors` configuration. format: input1:1x3x224x224;input2:1x1x112x112. type: string. required: false. default: .
      string input_shapes = 8;
    
      OnnxOptimizeOption onnx_opt = 10;
    
      QuantConfig quant = 20;
    
      repeated InputProcessor input_processors = 31;
      repeated OutputProcessor output_processors = 32;
      repeated ConstProcessor const_processors = 33;
      repeated OpProcessor op_processors = 34;
      repeated OpProcessor quant_op_processors = 35;
    
      CompilerConfig compiler = 40;
    }
