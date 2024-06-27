======================
Quick Start(AX620E)
======================

**本章节适用于以下平台：**

- AX630C
- AX620Q

本章节介绍 ``ONNX`` 模型转换的基本操作, 使用 ``pulsar2`` 工具将 ``ONNX``  模型编译成 ``axmodel`` 模型. 请先参考 :ref:`《开发环境准备》 <dev_env_prepare>` 章节完成开发环境搭建. 
本节示例模型为开源模型 ``MobileNetv2``.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
命令说明
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
配置文件说明
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``/data/config/`` 路径下的 ``mobilenet_v2_build_config.json`` 展示:

.. code-block:: shell

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

.. attention::

    ``input_processors``, ``output_processors`` 及 ``quant`` 节点下 ``input_configs`` 中的 ``tensor_name`` 字段需要根据模型的实际输入/输出节点名称进行设置，也可以设置为 ``DEFAULT`` 代表当前配置应用于全部输入或者输出。

    .. figure:: ../media/tensor_name.png
        :alt: pipeline
        :align: center

更加详细的内容，请参考 :ref:`《配置文件详细说明》 <config_details>`.

.. _model_compile_20e:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
编译执行
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

以 ``mobilenetv2-sim.onnx`` 为例, 执行如下 ``pulsar2 build`` 命令编译生成 ``compiled.axmodel``:

.. code-block:: shell

    pulsar2 build --target_hardware AX620E --input model/mobilenetv2-sim.onnx --output_dir output --config config/mobilenet_v2_build_config.json

.. warning::

    在编译模型前，需要确保已经对原始模型使用过 ``onnxsim`` 工具优化，主要目的是将模型转变成更利于 ``Pulsar2`` 编译的静态图及获得更好的推理性能。有以下两种方法：

    1. 在 ``Pulsar2`` docker 内部直接执行命令：``onnxsim in.onnx out.onnx``。
    2. 使用 ``pulsar2 build`` 进行模型转换时，增加参数：``--onnx_opt.enable_onnxsim true`` （默认值为 false）。

    如果想要进一步了解 ``onnxsim`` ，可访问 `官方网站 <https://github.com/daquexian/onnx-simplifier>`_ 。

^^^^^^^^^^^^^^^^^^^^^
log 参考信息
^^^^^^^^^^^^^^^^^^^^^

.. code-block::

    $ pulsar2 build --target_hardware AX620E --input model/mobilenetv2-sim.onnx --output_dir output --config config/mobilenet_v2_build_config.json
    2023-07-29 14:23:01.757 | WARNING  | yamain.command.build:fill_default:313 - ignore input csc config because of src_format is AutoColorSpace or src_format and tensor_format are the same
    Building onnx ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
    2023-07-29 14:23:07.806 | INFO     | yamain.command.build:build:424 - save optimized onnx to [output/frontend/optimized.onnx]
    patool: Extracting ./dataset/imagenet-32-images.tar ...
    patool: running /usr/bin/tar --extract --file ./dataset/imagenet-32-images.tar --directory output/quant/dataset/input
    patool: ... ./dataset/imagenet-32-images.tar extracted to `output/quant/dataset/input'.
                                                                            Quant Config Table
    ┏━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
    ┃ Input ┃ Shape            ┃ Dataset Directory ┃ Data Format ┃ Tensor Format ┃ Mean                                                         ┃ Std                ┃
    ┡━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
    │ input │ [1, 3, 224, 224] │ input             │ Image       │ BGR           │ [103.93900299072266, 116.77899932861328, 123.68000030517578] │ [58.0, 58.0, 58.0] │
    └───────┴──────────────────┴───────────────────┴─────────────┴───────────────┴──────────────────────────────────────────────────────────────┴────────────────────┘
    Transformer optimize level: 0
    32 File(s) Loaded.
    [14:23:09] AX LSTM Operation Format Pass Running ...      Finished.
    [14:23:09] AX Set MixPrecision Pass Running ...           Finished.
    [14:23:09] AX Refine Operation Config Pass Running ...    Finished.
    [14:23:09] AX Reset Mul Config Pass Running ...           Finished.
    [14:23:09] AX Tanh Operation Format Pass Running ...      Finished.
    [14:23:09] AX Confused Op Refine Pass Running ...         Finished.
    [14:23:09] AX Quantization Fusion Pass Running ...        Finished.
    [14:23:09] AX Quantization Simplify Pass Running ...      Finished.
    [14:23:09] AX Parameter Quantization Pass Running ...     Finished.
    Calibration Progress(Phase 1): 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:01<00:00, 18.07it/s]
    Finished.
    [14:23:11] AX Passive Parameter Quantization Running ...  Finished.
    [14:23:11] AX Parameter Baking Pass Running ...           Finished.
    [14:23:11] AX Refine Int Parameter Pass Running ...       Finished.
    [14:23:11] AX Refine Weight Parameter Pass Running ...    Finished.
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
    [Warning]File output/quant/quant_axmodel.onnx has already exist, quant exporter will overwrite it.
    [Warning]File output/quant/quant_axmodel.json has already exist, quant exporter will overwrite it.
    quant.axmodel export success: output/quant/quant_axmodel.onnx
    Building native ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
    2023-07-29 14:23:18.332 | WARNING  | yamain.command.load_model:pre_process:454 - preprocess tensor [input]
    2023-07-29 14:23:18.332 | INFO     | yamain.command.load_model:pre_process:456 - tensor: input, (1, 224, 224, 3), U8
    2023-07-29 14:23:18.332 | INFO     | yamain.command.load_model:pre_process:459 - op: op:pre_dequant_1, AxDequantizeLinear, {'const_inputs': {'x_zeropoint': 0, 'x_scale': 1}, 'output_dtype': <class 'numpy.float32'>, 'quant_method': 0}
    2023-07-29 14:23:18.332 | INFO     | yamain.command.load_model:pre_process:456 - tensor: tensor:pre_norm_1, (1, 224, 224, 3), FP32
    2023-07-29 14:23:18.332 | INFO     | yamain.command.load_model:pre_process:459 - op: op:pre_norm_1, AxNormalize, {'dim': 3, 'mean': [103.93900299072266, 116.77899932861328, 123.68000030517578], 'std': [58.0, 58.0, 58.0]}
    2023-07-29 14:23:18.332 | INFO     | yamain.command.load_model:pre_process:456 - tensor: tensor:pre_transpose_1, (1, 224, 224, 3), FP32
    2023-07-29 14:23:18.332 | INFO     | yamain.command.load_model:pre_process:459 - op: op:pre_transpose_1, AxTranspose, {'perm': [0, 3, 1, 2]}
    tiling op...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 174/174 0:00:00
    new_ddr_tensor = []
    build op...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 440/440 0:00:00
    add ddr swap...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1606/1606 0:00:00
    calc input dependencies...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2279/2279 0:00:00
    calc output dependencies...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2279/2279 0:00:00
    assign eu heuristic   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2279/2279 0:00:00
    assign eu onepass   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2279/2279 0:00:00
    assign eu greedy   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2279/2279 0:00:00
    2023-07-29 14:23:21.762 | INFO     | yasched.test_onepass:results2model:1882 - max_cycle = 782,940
    2023-07-29 14:23:22.159 | INFO     | yamain.command.build:compile_npu_subgraph:1004 - QuantAxModel macs: 280,262,480
    2023-07-29 14:23:25.209 | INFO     | backend.ax620e.linker:link_with_dispatcher:1586 - DispatcherQueueType.IO: Generate 69 EU chunks, 7 Dispatcher Chunk
    2023-07-29 14:23:25.209 | INFO     | backend.ax620e.linker:link_with_dispatcher:1586 - DispatcherQueueType.Compute: Generate 161 EU chunks, 23 Dispatcher Chunk
    2023-07-29 14:23:25.209 | INFO     | backend.ax620e.linker:link_with_dispatcher:1587 - EU mcode size: 147 KiB
    2023-07-29 14:23:25.209 | INFO     | backend.ax620e.linker:link_with_dispatcher:1588 - Dispatcher mcode size: 21 KiB
    2023-07-29 14:23:25.209 | INFO     | backend.ax620e.linker:link_with_dispatcher:1589 - Total mcode size: 168 KiB
    2023-07-29 14:23:26.928 | INFO     | yamain.command.build:compile_ptq_model:940 - fuse 1 subgraph(s)

.. note::

    该示例所运行的主机配置为:

        - Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz
        - Memory 32G

    全流程耗时大约 ``11s`` , 不同配置的主机转换时间略有差异.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
输出文件说明
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell  

    root@xxx:/data# tree output/
    output/
    ├── build_context.json
    ├── compiled.axmodel            # 最终板上运行模型，AxModel
    ├── compiler                    # 编译器后端中间结果及 debug 信息
    ├── frontend                    # 前端图优化中间结果及 debug 信息
    │   └── optimized.onnx          # 输入模型经过图优化以后的浮点 ONNX 模型
    └── quant                       # 量化工具输出及 debug 信息目录
        ├── dataset                 # 解压后的校准集数据目录
        │   └── input
        │       ├── ILSVRC2012_val_00000001.JPEG
        │       ├── ......
        │       └── ILSVRC2012_val_00000032.JPEG
        ├── debug
        ├── quant_axmodel.json      # 量化配置信息
        └── quant_axmodel.onnx      # 量化后的模型，QuantAxModel

其中 ``compiled.axmodel`` 为最终编译生成的板上可运行的 ``.axmodel`` 模型文件

.. note::

    因为 ``.axmodel`` 基于 **ONNX** 模型存储格式开发，所以将 ``.axmodel`` 文件后缀修改为 ``.axmodel.onnx`` 后可支持被网络模型图形化工具 **Netron** 直接打开。

    .. figure:: ../media/axmodel-netron.png
        :alt: pipeline
        :align: center

----------------------
信息查询
----------------------

可以通过 ``onnx inspect --io ${axmodel/onnx_path}`` 来查看 ``axmodel`` 模型的输入输出信息，还有其他 ``-m -n -t`` 参数可以查看模型里的 ``meta / node / tensor`` 信息。

.. code-block:: shell

    root@xxx:/data# onnx inspect -m -n -t output/compiled.axmodel
    Failed to check model output/compiled.axmodel, statistic could be inaccurate!
    Inpect of model output/compiled.axmodel
    ================================================================================
      Graph name: 8
      Graph inputs: 1
      Graph outputs: 1
      Nodes in total: 1
      ValueInfo in total: 2
      Initializers in total: 2
      Sparse Initializers in total: 0
      Quantization in total: 0

    Meta information:
    --------------------------------------------------------------------------------
      IR Version: 7
      Opset Import: [version: 13
    ]
      Producer name: Pulsar2
      Producer version:
      Domain:
      Doc string: Pulsar2 Version:  1.8-beta1
    Pulsar2 Commit: 6a7e59de
      meta.{} = {} extra_data CgsKBWlucHV0EAEYAgoICgZvdXRwdXQSATEaMgoFbnB1XzBSKQoNbnB1XzBfYjFfZGF0YRABGhYKBnBhcmFtcxoMbnB1XzBfcGFyYW1zIgAoAQ==

    Node information:
    --------------------------------------------------------------------------------
      Node type "neu mode" has: 1
    --------------------------------------------------------------------------------
      Node "npu_0": type "neu mode", inputs "['input']", outputs "['output']"

    Tensor information:
    --------------------------------------------------------------------------------
      ValueInfo "input": type UINT8, shape [1, 224, 224, 3],
      ValueInfo "output": type FLOAT, shape [1, 1000],
      Initializer "npu_0_params": type UINT8, shape [3740416],
      Initializer "npu_0_b1_data": type UINT8, shape [173256],

.. _model_simulator_20e:

----------------------
仿真运行
----------------------

本章节介绍 ``axmodel`` 仿真运行的基本操作, 使用 ``pulsar2 run`` 命令可以直接在 ``PC`` 上直接运行由 ``pulsar2 build`` 生成的 ``axmodel`` 模型，无需上板运行即可快速得到网络模型的运行结果。

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
仿真运行准备
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

仿真运行时需要的 ``前处理`` 和 ``后处理`` 工具已包含在 ``pulsar2-run-helper`` 文件夹中。

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

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
仿真运行 ``mobilenetv2``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

将 :ref:`《编译执行》 <model_compile_20e>` 章节生成的 ``compiled.axmodel`` 拷贝 ``pulsar2-run-helper/models`` 路径下，并更名为 ``mobilenetv2.axmodel``

.. code-block:: shell

    root@xxx:/data# cp output/compiled.axmodel pulsar2-run-helper/models/mobilenetv2.axmodel

^^^^^^^^^^^^^^^^^^^^^
输入数据准备
^^^^^^^^^^^^^^^^^^^^^

进入 ``pulsar2-run-helper`` 目录，使用 ``cli_classification.py`` 脚本将 ``cat.jpg`` 处理成 ``mobilenetv2.axmodel`` 所需要的输入数据格式。

.. code-block:: shell

    root@xxx:~/data# cd pulsar2-run-helper
    root@xxx:~/data/pulsar2-run-helper# python3 cli_classification.py --pre_processing --image_path sim_images/cat.jpg --axmodel_path models/mobilenetv2.axmodel --intermediate_path sim_inputs/0
    [I] Write [input] to 'sim_inputs/0/input.bin' successfully.

^^^^^^^^^^^^^^^^^^^^^
仿真模型推理
^^^^^^^^^^^^^^^^^^^^^

运行 ``pulsar2 run`` 命令，将 ``input.bin`` 作为 ``mobilenetv2.axmodel`` 的输入数据并执行推理计算，输出 ``output.bin`` 推理结果。

.. code-block:: shell

    root@xxx:~/data/pulsar2-run-helper# pulsar2 run --model models/mobilenetv2.axmodel --input_dir sim_inputs --output_dir sim_outputs --list list.txt
    Building native ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
    >>> [0] start
    write [output] to [sim_outputs/0/output.bin] successfully
    >>> [0] finish

^^^^^^^^^^^^^^^^^^^^^
输出数据处理
^^^^^^^^^^^^^^^^^^^^^

使用 ``cli_classification.py`` 脚本对仿真模型推理输出的 ``output.bin`` 数据进行后处理，得到最终计算结果。

.. code-block:: shell

    root@xxx:/data/pulsar2-run-helper# python3 cli_classification.py --post_processing --axmodel_path models/mobilenetv2.axmodel --intermediate_path sim_outputs/0
    [I] The following are the predicted score index pair.
    [I] 9.1132, 285
    [I] 8.8490, 281
    [I] 8.7169, 282
    [I] 8.0566, 283
    [I] 6.8679, 463

.. _onboard_running_20e:

----------------------
开发板运行
----------------------

本章节介绍如何在 ``AX630C`` ``AX620Q`` 开发板上运行通过 :ref:`《编译执行》 <model_compile_20e>` 章节获取 ``compiled.axmodel`` 模型. 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
开发板获取
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- 通过企业途径向 AXera 签署 NDA 后获取 **AX630C DEMO Board**.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
使用 ax_run_model 工具快速测试模型推理速度
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

为了方便用户测评模型，在开发板上预制了 :ref:`ax_run_model <ax_run_model>` 工具，此工具有若干参数，可以很方便地测试模型速度和精度。

将 ``mobilenetv2.axmodel`` 拷贝到开发板上，执行以下命令即可快速测试模型推理性能（首先推理 3 次进行预热，以排除资源初始化导致的统计误差，然后推理 10 次，统计平均推理速度）。

.. code-block:: shell

    /root # ax_run_model -m /opt/data/npu/models/mobilenetv2.axmodel -w 3 -r 10
      Run AxModel:
            model: /opt/data/npu/models/mobilenetv2.axmodel
             type: Half
             vnpu: Disable
         affinity: 0b01
           warmup: 3
           repeat: 10
            batch: { auto: 0 }
      pulsar2 ver: 1.8-beta1 6a7e59de
       engine ver: 2.6.3sp
         tool ver: 2.3.3sp
         cmm size: 4414192 Bytes
      ------------------------------------------------------
      min =   1.093 ms   max =   1.098 ms   avg =   1.096 ms
      ------------------------------------------------------

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
使用 sample_npu_classification 示例测试单张图片推理结果
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. hint::

    该运行示例已经预装在开发板的文件系统中，其源文件位于 SDK 的 ``msp/sample/npu`` 路径下文件夹下。将 ``mobilennetv2.axmodel`` 拷贝到开发板上，使用 ``sample_npu_classification`` 进行测试。

``sample_npu_classification`` 输入参数说明: 

.. code-block:: shell

    /root # sample_npu_classification --help
    usage: sample_npu_classification --model=string --image=string [options] ...
    options:
      -m, --model     joint file(a.k.a. joint model) (string)
      -i, --image     image file (string)
      -g, --size      input_h, input_w (string [=224,224])
      -r, --repeat    repeat count (int [=1])
      -?, --help      print this message

通过执行 ``sample_npu_classification`` 程序实现分类模型板上运行, 运行结果如下:

.. code-block:: shell

    /root # sample_npu_classification -m mobilenetv2.axmodel -i /opt/data/npu/images/cat.jpg -r 100
    --------------------------------------
    model file : mobilenetv2.axmodel
    image file : /opt/data/npu/images/cat.jpg
    img_h, img_w : 224 224
    --------------------------------------
    Engine creating handle is done.
    Engine creating context is done.
    Engine get io info is done.
    Engine alloc io is done.
    Engine push input is done.
    --------------------------------------
    topk cost time:0.10 ms
    9.1132, 285
    8.8490, 281
    8.7169, 282
    8.0566, 283
    6.8679, 463
    --------------------------------------
    Repeat 100 times, avg time 1.09 ms, max_time 1.10 ms, min_time 1.09 ms
    --------------------------------------

- 从这里可知，同一个 ``mobilenetv2.axmodel`` 模型在开发板上运行的结果与 :ref:`《仿真运行》 <model_simulator_20e>` 的结果一致；
- 板上可执行程序 ``ax_classification`` 相关源码及编译生成详情请参考 :ref:`《模型部署进阶指南》 <model_deploy_advanced>`。 
