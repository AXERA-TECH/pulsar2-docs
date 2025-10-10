======================
Quick Start(M57)
======================

**本章节适用于以下平台：**

- M57

本章节介绍 ``ONNX`` 模型转换的基本操作, 使用 ``pulsar2`` 工具将 ``ONNX``  模型编译成 ``axmodel`` 模型. 请先参考 :ref:`《开发环境准备》 <dev_env_prepare>` 章节完成开发环境搭建. 
本节示例模型为开源模型 ``MobileNetv2``.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pulsar2 工具链命令说明
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Pulsar2`` 工具链中的功能指令以 ``pulsar2`` 开头, 与用户强相关的命令为 ``pulsar2 build`` , ``pulsar2 run`` 以及 ``pulsar2 version``. 

* ``pulsar2 build`` 用于将 ``onnx`` 模型转换为 ``axmodel`` 格式模型
* ``pulsar2 run`` 用于模型转换后的仿真运行
* ``pulsar2 version`` 可以用于查看当前工具链的版本信息, 通常在反馈问题时需要提供此信息

.. code-block:: shell

    root@xxx:/data# pulsar2 --help
    usage: pulsar2 [-h] {version,build,run} ...
    
    positional arguments:
      {version,build,run}
    
    optional arguments:
      -h, --help           show this help message and exit

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
模型编译配置文件说明
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``/data/config/`` 路径下的 ``mobilenet_v2_build_config.json`` 展示:

.. code-block:: json

    {
      "model_type": "ONNX",
      "npu_mode": "NPU1",
      "quant": {
        "input_configs": [
          {
            "tensor_name": "input",
            "calibration_dataset": "./dataset/imagenet-32-images.tar", 
            "calibration_size": 32,
            // 校验数据集归一化的各通道均值, 通道顺序与 tensor_format 一致
            "calibration_mean": [103.939, 116.779, 123.68], 
            // 校验数据集归一化的各通道标准差
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
          // 运行时输入格式
          "src_format": "BGR",
          // 运行时数据类型
          "src_dtype": "U8",
          // 运行时数据布局格式
          "src_layout": "NHWC",
          // 颜色空间转换
          "csc_mode": "NoCSC"
        }
      ],
      "compiler": {
        "check": 0
      }
    }

.. attention::

    ``input_processors``, ``output_processors`` 及 ``quant`` 节点下 ``input_configs`` 中的 ``tensor_name`` 字段需要根据模型的实际输入/输出节点名称进行设置，也可以设置为 ``DEFAULT`` 代表当前配置应用于全部输入或者输出。

    .. figure:: ../media/tensor_name.png
        :alt: pipeline
        :align: center

更加详细的内容，请参考 :ref:`《配置文件详细说明》 <config_details>`.

.. _model_compile_M57:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
编译执行
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

以 ``mobilenetv2-sim.onnx`` 为例, 执行如下 ``pulsar2 build`` 命令编译生成 ``compiled.axmodel``:

.. code-block:: shell

    pulsar2 build --target_hardware M57 --input model/mobilenetv2-sim.onnx --output_dir output --config config/mobilenet_v2_build_config.json

.. warning::

    在编译模型前，需要确保已经对原始模型使用过 ``onnxsim`` 工具优化，主要目的是将模型转变成更利于 ``Pulsar2`` 编译的静态图及获得更好的推理性能。有以下两种方法：

    1. 在 ``Pulsar2`` docker 内部直接执行命令：``onnxsim in.onnx out.onnx``。
    2. 使用 ``pulsar2 build`` 进行模型转换时，增加参数：``--onnx_opt.enable_onnxsim true`` （默认值为 false)。

    如果想要进一步了解 ``onnxsim`` ，可访问 `官方网站 <https://github.com/daquexian/onnx-simplifier>`_ 。

^^^^^^^^^^^^^^^^^^^^^
log 参考信息
^^^^^^^^^^^^^^^^^^^^^

.. code-block::

    $ pulsar2 build --target_hardware M57 --input model/mobilenetv2-sim.onnx --output_dir output --config config/mobilenet_v2_build_config.json
    2025-03-14 14:45:26.362 | WARNING  | yamain.command.build:fill_default:265 - apply default output processor configuration to ['output']
    2025-03-14 14:45:26.362 | WARNING  | yamain.command.build:fill_default:340 - ignore input csc config because of src_format is AutoColorSpace or src_format and tensor_format are the same
    2025-03-14 14:45:26.363 | INFO     | yamain.common.util:extract_archive:217 - extract [dataset/imagenet-32-images.tar] to [output/quant/dataset/input]...
    32 File(s) Loaded.
    Building onnx ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
    2025-03-14 14:45:27.529 | INFO     | yamain.command.build:quant:748 - save optimized onnx to [output/frontend/optimized.onnx]
                                                                                Quant Config Table                                                                             
    ┏━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
    ┃ Input ┃ Shape            ┃ Dataset Directory          ┃ Data Format ┃ Tensor Format ┃ Mean                                                         ┃ Std                ┃
    ┡━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
    │ input │ [1, 3, 224, 224] │ output/quant/dataset/input │ Image       │ BGR           │ [103.93900299072266, 116.77899932861328, 123.68000030517578] │ [58.0, 58.0, 58.0] │
    └───────┴──────────────────┴────────────────────────────┴─────────────┴───────────────┴──────────────────────────────────────────────────────────────┴────────────────────┘
    Transformer optimize level: 0
    32 File(s) Loaded.
    Stastic Inf tensor: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  9.41it/s]
    [14:45:28] AX Set Float Op Table Pass Running ...         
    [14:45:29] AX Set MixPrecision Pass Running ...           
    [14:45:29] AX Set LN Quant dtype Quant Pass Running ...   
    [14:45:29] AX Reset Mul Config Pass Running ...           
    [14:45:29] AX Refine Operation Config Pass Running ...    
    [14:45:29] AX Tanh Operation Format Pass Running ...      
    [14:45:29] AX Confused Op Refine Pass Running ...         
    [14:45:29] AX Quantization Fusion Pass Running ...        
    [14:45:29] AX Quantization Simplify Pass Running ...      
    [14:45:29] AX Parameter Quantization Pass Running ...     
    [14:45:29] AX Runtime Calibration Pass Running ...        
    Calibration Progress(Phase 1): 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:03<00:00,  8.86it/s]
    [14:45:33] AX Quantization Alignment Pass Running ...     
    [14:45:33] AX Refine Int Parameter Pass Running ...       
    [14:45:33] AX Refine Scale Pass Running ...               
    [14:45:33] AX Passive Parameter Quantization Running ...  
    [14:45:33] AX Parameter Baking Pass Running ...           
    --------- Network Snapshot ---------
    Num of Op:                    [100]
    Num of Quantized Op:          [100]
    Num of Variable:              [278]
    Num of Quantized Var:         [278]
    ------- Quantization Snapshot ------
    Num of Quant Config:          [387]
    BAKED:                        [53]
    OVERLAPPED:                   [145]
    ACTIVATED:                    [65]
    SOI:                          [1]
    PASSIVE_BAKED:                [53]
    FP32:                         [70]
    Network Quantization Finished.
    Do quant optimization
    quant.axmodel export success: 
            /opt/pulsar2/quick_start_example/output/quant/quant_axmodel.onnx
            /opt/pulsar2/quick_start_example/output/quant/quant_axmodel.data
    ===>export io data to folder: output/quant/debug/io
    Building native ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
    2025-03-14 14:45:35.003 | INFO     | yamain.command.build:compile_ptq_model:1029 - group 0 compiler transformation
    2025-03-14 14:45:35.005 | WARNING  | yamain.command.load_model:pre_process:616 - preprocess tensor [input]
    2025-03-14 14:45:35.005 | INFO     | yamain.command.load_model:pre_process:617 - tensor: input, (1, 224, 224, 3), U8
    2025-03-14 14:45:35.005 | INFO     | yamain.command.load_model:pre_process:617 - op: op:pre_dequant_1, AxDequantizeLinear, {'const_inputs': {'x_zeropoint': array(0, dtype=int32), 'x_scale': array(1., dtype=float32)}, 'output_dtype': <class 'numpy.float32'>, 'quant_method': 0}
    2025-03-14 14:45:35.005 | INFO     | yamain.command.load_model:pre_process:617 - tensor: tensor:pre_norm_1, (1, 224, 224, 3), FP32
    2025-03-14 14:45:35.005 | INFO     | yamain.command.load_model:pre_process:617 - op: op:pre_norm_1, AxNormalize, {'dim': 3, 'mean': [103.93900299072266, 116.77899932861328, 123.68000030517578], 'std': [58.0, 58.0, 58.0], 'output_dtype': FP32}
    2025-03-14 14:45:35.006 | INFO     | yamain.command.load_model:pre_process:617 - tensor: tensor:pre_transpose_1, (1, 224, 224, 3), FP32
    2025-03-14 14:45:35.006 | INFO     | yamain.command.load_model:pre_process:617 - op: op:pre_transpose_1, AxTranspose, {'perm': [0, 3, 1, 2]}
    2025-03-14 14:45:35.006 | WARNING  | yamain.command.load_model:post_process:638 - postprocess tensor [output]
    2025-03-14 14:45:35.006 | INFO     | yamain.command.load_model:ir_compiler_transformation:824 - use random data as gt input: input, uint8, (1, 224, 224, 3)
    2025-03-14 14:45:35.209 | INFO     | yamain.command.build:compile_ptq_model:1052 - group 0 QuantAxModel macs: 300,774,272
    2025-03-14 14:45:35.220 | INFO     | yamain.command.build:compile_ptq_model:1182 - subgraph [0], group: 0, type: GraphType.NPU
    2025-03-14 14:45:35.221 | INFO     | yamain.command.npu_backend_compiler:compile:174 - compile npu subgraph [0]
    tiling op...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 68/68 0:00:00
    new_ddr_tensor = []
    build op serially...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 149/149 0:00:00
    build op...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 268/268 0:00:00
    add ddr swap...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 344/344 0:00:00
    calc input dependencies...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 576/576 0:00:00
    calc output dependencies...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 576/576 0:00:00
    assign eu heuristic   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 576/576 0:00:00
    assign eu onepass   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 576/576 0:00:00
    assign eu greedy   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 576/576 0:00:00
    2025-03-14 14:45:36.965 | INFO     | yasched.test_onepass:results2model:2682 - clear job deps
    2025-03-14 14:45:36.966 | INFO     | yasched.test_onepass:results2model:2683 - max_cycle = 684,124
    build jobs   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 576/576 0:00:00
    2025-03-14 14:45:37.179 | INFO     | yamain.command.npu_backend_compiler:compile:235 - assemble model [0] [subgraph_npu_0] b1
    2025-03-14 14:45:37.931 | INFO     | yamain.command.build:compile_ptq_model:1221 - fuse 1 subgraph(s)

.. note::

    该示例所运行的主机配置为:

        - Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz
        - Memory 32G

    全流程耗时大约 ``12s`` , 不同配置的主机转换时间略有差异.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
模型编译输出文件说明
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell  

    root@xxx:/data# tree output/
    output/
    |-- build_context.json
    |-- compiled.axmodel               # 最终板上运行模型，AxModel
    |-- compiler                       # 编译器后端中间结果及 debug 信息
    |   `-- debug                      # 前端图优化中间结果及 debug 信息
    |       `-- subgraph_npu_0
    |           `-- b1
    |-- frontend
    |   |-- optimized.data
    |   `-- optimized.onnx             # 输入模型经过图优化以后的浮点 ONNX 模型
    `-- quant                          # 量化工具输出及 debug 信息目录
        |-- dataset
        |   `-- input
        |       |-- ILSVRC2012_val_00000001.JPEG
        |       |-- ......
        |       `-- ILSVRC2012_val_00000032.JPEG
        |-- debug
        |   `-- io
        |       |-- float
        |       |   |-- input.npy
        |       |   `-- output.npy
        |       `-- quant
        |           |-- input.npy
        |           `-- output.npy
        |-- quant_axmodel.data
        |-- quant_axmodel.json         # 量化配置信息
        `-- quant_axmodel.onnx         # 量化后的模型，QuantAxModel

其中 ``compiled.axmodel`` 为最终编译生成的板上可运行的 ``.axmodel`` 模型文件

.. note::

    因为 ``.axmodel`` 基于 **ONNX** 模型存储格式开发，所以将 ``.axmodel`` 文件后缀修改为 ``.axmodel.onnx`` 后可支持被网络模型图形化工具 **Netron** 直接打开。

    .. figure:: ../media/axmodel-netron.png
        :alt: pipeline
        :align: center

----------------------
模型信息查询
----------------------

可以通过 ``onnx inspect --io ${axmodel/onnx_path}`` 来查看编译后 ``axmodel`` 模型的输入输出信息，还有其他 ``-m -n -t`` 参数可以查看模型里的 ``meta / node / tensor`` 信息。

.. code-block:: shell

    root@xxx:/data# onnx inspect -m -n -t output/compiled.axmodel
    Failed to check model output/compiled.axmodel, statistic could be inaccurate!
    Meta information
    --------------------------------------------------------------------------------
      IR Version: 10
      Opset Import: [domain: ""
    version: 18
    ]
      Producer name: Pulsar2
      Producer version: 
      Domain: 
      Doc string: Pulsar2 Version:  3.4
    Pulsar2 Commit: 3dfd5692
      meta.{} = {} extra_data CgsKBWlucHV0EAEYAgoICgZvdXRwdXQSATEaQwoOc3ViZ3JhcGhfbnB1XzBSMQoVc3ViZ3JhcGhfbnB1XzBfYjFfbmV1EAEaFgoGcGFyYW1zGgxucHVfMF9wYXJhbXMiACgE
    Node information
    --------------------------------------------------------------------------------
      Node type "neu mode" has: 1
    --------------------------------------------------------------------------------
      Node "subgraph_npu_0": type "neu mode", inputs "['input']", outputs "['output']"
    Tensor information
    --------------------------------------------------------------------------------
      ValueInfo "input": type UINT8, shape [1, 224, 224, 3],
      ValueInfo "npu_0_params": type UINT8, shape [4324276],
      ValueInfo "subgraph_npu_0_b1_neu": type UINT8, shape [122368],
      ValueInfo "output": type FLOAT, shape [1, 1000],
      Initializer "npu_0_params": type UINT8, shape [4324276],
      Initializer "subgraph_npu_0_b1_neu": type UINT8, shape [122368],

.. _model_simulator_M57:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
仿真运行
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

本章节介绍 ``axmodel`` 仿真运行的基本操作, 使用 ``pulsar2 run`` 命令可以直接在 ``PC`` 上直接运行由 ``pulsar2 build`` 生成的 ``axmodel`` 模型，无需上板运行即可快速得到网络模型的运行结果。

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
仿真运行准备
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

某些模型只能支持特定的输入数据格式，模型的输出数据也是以模组特定的格式输出的。在模型仿真运行前，需要把输入数据转换成模型支持的数据格式，这部分数据操作称为 ``前处理`` 。在模型仿真运行后，需要把输出数据转换成工具可以分析查看的数据格式，这部分数据操作称为 ``后处理`` 。仿真运行时需要的 ``前处理`` 和 ``后处理`` 工具已包含在 ``pulsar2-run-helper`` 文件夹中。

``pulsar2-run-helper`` 文件夹内容如下所示：

.. code-block:: shell

    root@xxx:/data# ll pulsar2-run-helper/
    drwxr-xr-x 2 root root 4.0K Dec  2 12:23 models/
    drwxr-xr-x 5 root root 4.0K Dec  2 12:23 pulsar2_run_helper/
    drwxr-xr-x 2 root root 4.0K Dec  2 12:23 sim_images/
    drwxr-xr-x 2 root root 4.0K Dec  2 12:23 sim_inputs/
    drwxr-xr-x 2 root root 4.0K Dec  2 12:23 sim_outputs/
    -rw-r--r-- 1 root root 3.0K Dec  2 12:23 cli_classification.py
    -rw-r--r-- 1 root root 4.6K Dec  2 12:23 cli_detection.py
    -rw-r--r-- 1 root root    2 Dec  2 12:23 list.txt
    -rw-r--r-- 1 root root   29 Dec  2 12:23 requirements.txt
    -rw-r--r-- 1 root root  308 Dec  2 12:23 setup.cfg

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
仿真运行 示例 ``mobilenetv2``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

将 :ref:`《编译执行》 <model_compile_M57>` 章节生成的 ``compiled.axmodel`` 拷贝 ``pulsar2-run-helper/models`` 路径下，并更名为 ``mobilenetv2.axmodel``

.. code-block:: shell

    root@xxx:/data# cp output/compiled.axmodel pulsar2-run-helper/models/mobilenetv2.axmodel

----------------------
输入数据准备
----------------------

进入 ``pulsar2-run-helper`` 目录，使用 ``cli_classification.py`` 脚本将 ``cat.jpg`` 处理成 ``mobilenetv2.axmodel`` 所需要的输入数据格式。

.. code-block:: shell

    root@xxx:~/data# cd pulsar2-run-helper
    root@xxx:~/data/pulsar2-run-helper# python3 cli_classification.py --pre_processing --image_path sim_images/cat.jpg --axmodel_path models/mobilenetv2.axmodel --intermediate_path sim_inputs/0
    [I] Write [input] to 'sim_inputs/0/input.bin' successfully.

----------------------
仿真模型推理
----------------------

运行 ``pulsar2 run`` 命令，将 ``input.bin`` 作为 ``mobilenetv2.axmodel`` 的输入数据并执行推理计算，输出 ``output.bin`` 推理结果。

.. code-block:: shell

    root@xxx:~/data/pulsar2-run-helper# pulsar2 run --model models/mobilenetv2.axmodel --input_dir sim_inputs --output_dir sim_outputs --list list.txt
    Building native ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
    2025-03-14 15:00:34.804 | INFO     | yamain.command.run:run:90 - >>> [0] start
    2025-03-14 15:00:34.805 | INFO     | frontend.npu_subgraph_op:pyrun:89 - running npu subgraph: subgraph_npu_0, version: 1, target batch: 0
    2025-03-14 15:00:43.900 | INFO     | yamain.command.run:write_output:55 - write [output] to [sim_outputs/0/output.bin] successfully, size: 4000

----------------------
输出数据处理
----------------------

使用 ``cli_classification.py`` 脚本对仿真模型推理输出的 ``output.bin`` 数据进行后处理，得到最终计算结果。

.. code-block:: shell

    root@xxx:/data/pulsar2-run-helper# python3 cli_classification.py --post_processing --axmodel_path models/mobilenetv2.axmodel --intermediate_path sim_outputs/0
    [I] The following are the predicted score index pair.
    [I] 8.8490, 283
    [I] 8.7169, 285
    [I] 8.4528, 282
    [I] 8.4528, 281
    [I] 7.6603, 463

.. _onboard_running_M57:
