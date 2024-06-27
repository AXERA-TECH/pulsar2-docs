===================
仿真运行进阶指南
===================

-------------------------------
概述
-------------------------------

``pulsar2 run`` 用于在 ``x86`` 平台上对 ``axmodel`` 模型进行 **x86仿真推理计算**，提前获取编译后的模型运行结果.

.. figure:: ../media/pulsar2-run-pipeline.png
    :alt: pipeline
    :align: center

.. _pulsar_run:

-------------------------------
仿真运行详解
-------------------------------

~~~~~~~~~~~~~~~~~~~~~
pulsar2 run 
~~~~~~~~~~~~~~~~~~~~~

本节介绍 ``pulsar2 run`` 命令完整使用方法.

``pulsar2 run -h`` 可显示详细命令行参数:

.. code-block:: python
    :name: input_conf_items
    :linenos:

    usage: main.py run [-h] [--config] [--model] [--input_dir] [--output_dir]
                       [--list] [--random_input] [--batch_size]
                       [--enable_perlayer_output] [--dump_with_stride]
                       [--group_index] [--mode] [--target_hardware]
    optional arguments:
      -h, --help            show this help message and exit
      --config              config file path, supported formats: json / yaml /
                            toml / prototxt. type: string. required: false.
                            default:.
      --model               run model path, support ONNX, QuantAxModel and
                            CompiledAxmodel. type: string. required: true.
      --input_dir           model input data in this directory. type: string.
                            required: true. default:.
      --output_dir          model output data directory. type: string. required:
                            true. default:.
      --list                list file path. type: string. required: true.
                            default:.
      --random_input        random input data. type: bool. required: false.
                            default: false.
      --batch_size          batch size to be used in dynamic inference mode, only
                            work for CompiledAxModel. type: int. required: false.
                            defalult: 0.
      --enable_perlayer_output 
                            enable dump perlayer output. type: bool. required:
                            false. default: false.
      --dump_with_stride 
      --group_index 
      --mode                run mode, only work for QuantAxModel. type: enum.
                            required: false. default: Reference. option:
                            Reference, NPUBackend.
      --target_hardware     target hardware, only work for QuantAxModel. type:
                            enum. required: false. default: AX650. option: AX650,
                            AX620E, M76H.

.. data:: pulsar2 run 参数解释
  
    --model

        - 数据类型：string
        - 是否必选：是
        - 描述：推理仿真的模型路径，模型支持 ``ONNX``, ``QuantAXModel`` 或者 ``AXModel`` 格式

    --input_dir

        - 数据类型：string
        - 是否必选：是
        - 描述：模型仿真输入数据文件所在的目录。

    --output_dir
    
        - 数据类型：string
        - 是否必选：是
        - 描述：模型仿真输出数据文件所在的目录。

    --list
    
        - 数据类型：string
        - 是否必选：否
        - 默认值：""
        - 描述：若未指定，则直接从 ``input_dir`` 中读取仿真输入数据，仿真结果直接写到 ``output_dir`` 中。若指定了 list 文件路径，则文件中的每一行代表一次仿真，会在 ``input_dir`` / ``output_dir`` 下寻找以行内容命名的子目录，分别用于读取仿真输入和写出仿真结果。例如：当 ``list`` 指定的文件中有一行内容为 0，仿真输入数据文件在 ``input_dir/0`` 目录下，仿真结果在 ``output_dir/0`` 目录下。

    --random_input
    
        - 数据类型：bool
        - 是否必选：否
        - 默认值：false
        - 描述：是否在 ``input_dir`` 中生成随机输入用于后续的仿真。

    .. attention::
    
        仿真输入输出数据文件的命名方法。
    
        .. code-block:: python
            :linenos:
        
            import re
        
            # 假设变量 name 代表模型输入名称
            escaped_name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
            file_name = escaped_name + ".bin"

    --batch_size
    
        - 数据类型：int
        - 是否必选：否
        - 默认值：0
        - 描述：多 batch 仿真大小，仅支持 ``CompiledAxmodel``。
            - 当输入模型是非多 batch 编译出的模型时，循环运行 batch_size 次。
            - 当输入模型是多 batch 编译出的模型时，会根据模型中包含的 batch 组合以及 batch_size 自动计算出仿真过程。

    --enable_perlayer_output
    
        - 数据类型：bool
        - 是否必选：否
        - 默认值：false
        - 描述：仿真时，将中间层的输出保存到输出目录。

    --mode
    
        - 数据类型：enum
        - 是否必选：否
        - 默认值：Reference
        - 描述：AX 算子的运行模式，仅支持 ``QuantAxModel``。可选：Reference / NPUBackend。

    --target_hardware
    
        - 数据类型：enum
        - 是否必选：否
        - 默认值：AX650
        - 描述：运行 AX 算子的目标后端实现，仅支持 ``QuantAxModel``。当 ``mode`` 为 ``NPUBackend`` 时生效。

~~~~~~~~~~~~~~~~~~~~~
pulsar2-run-helper
~~~~~~~~~~~~~~~~~~~~~

为了使用 ``pulsar2 run`` 模拟真实的上板运行结果，我们提供了 ``pulsar2-run-helper`` 工具实现网络模型运行依赖的 **输入**、 **输出** 数据处理，指导用户实现以下功能。

* 将 ``jpg``、 ``png`` 等格式的图片预处理成 ``pulsar2 run`` 命令参数 ``input_dir`` 所指定的格式；
* 解析 ``pulsar2 run`` 运行完成后输出到 ``output_dir`` 中的输出数据，实现 **分类**、 **检测** 任务的后处理操作；
* 所以工具内容均由 **python** 脚本实现，便于算法工程师快速上手。

``pulsar2-run-helper`` 获取方式及环境搭建请先参考 :ref:`《仿真运行》 <model_simulator>` 章节。

**pulsar2-run-helper** 目录说明如下：

.. code-block:: shell

    root@xxx:/data/pulsar2-run-helper# tree -L 2
    .
    ├── cli_classification.py     # 分类任务的数据处理参考脚本 
    ├── cli_detection.py          # 检测任务的数据处理参考脚本
    ├── models
    │   ├── mobilenetv2.axmodel   # 由 pulsar2 build 生成的 axmodel
    │   └── yolov5s.axmodel
    ├── pulsar2_run_helper
    │   ├── __init__.py
    │   ├── pipeline
    │   ├── postprocessing.py
    │   ├── preprocessing.py
    │   ├── utils
    │   └── yolort
    ├── pyproject.toml
    ├── README.md
    ├── requirements.txt
    ├── setup.cfg
    ├── sim_images                # 仿真运行的图片
    │   ├── cat.jpg
    │   └── dog.jpg
    ├── sim_inputs                # 输入数据
    ├── sim_inputs
    │   ├── 0
    │   │   └── input.bin
    │   └── input.bin
    └── sim_outputs
        ├── 0
        │   └── output.bin
        └── output.bin

**cli_classification** 参数说明

.. code-block:: shell

    root@xxx:/data# python3 pulsar2-run-helper/cli_classification.py -h
    usage: CLI tools for pre-processing and post-processing. [-h] [--image_path IMAGE_PATH] --axmodel_path AXMODEL_PATH --intermediate_path INTERMEDIATE_PATH
                                                            [--output_path OUTPUT_PATH] [--crop_size CROP_SIZE] [--pre_processing] [--post_processing]

    optional arguments:
      -h, --help            show this help message and exit
      --image_path IMAGE_PATH
                            The path of image file.
      --axmodel_path AXMODEL_PATH
                            The path of compiled axmodel.
      --intermediate_path INTERMEDIATE_PATH
                            The path of intermediate data bin.
      --output_path OUTPUT_PATH
                            The path of output files.
      --crop_size CROP_SIZE
                            Image size for croping (default: 224).
      --pre_processing      Do pre processing.
      --post_processing     Do post processing.

**cli_detection** 参数说明

.. code-block:: shell

    root@xxx:/data/pulsar2-run-helper# python3 cli_detection.py --help
    usage: CLI tools for pre-processing and post-processing. [-h] [--image_path IMAGE_PATH] --axmodel_path AXMODEL_PATH --intermediate_path INTERMEDIATE_PATH [--output_path OUTPUT_PATH]
                                                            [--letterbox_size LETTERBOX_SIZE] [--num_classes NUM_CLASSES] [--score_thresh SCORE_THRESH] [--nms_thresh NMS_THRESH]
                                                            [--pre_processing] [--post_processing]

    optional arguments:
      -h, --help            show this help message and exit
      --image_path IMAGE_PATH
                            The path of image file.
      --axmodel_path AXMODEL_PATH
                            The path of compiled axmodel.
      --intermediate_path INTERMEDIATE_PATH
                            The path of intermediate data bin.
      --output_path OUTPUT_PATH
                            The path of output files.
      --letterbox_size LETTERBOX_SIZE
                            Image size for croping (default: 640).
      --num_classes NUM_CLASSES
                            Number of classes (default: 80).
      --score_thresh SCORE_THRESH
                            Threshold of score (default: 0.45).
      --nms_thresh NMS_THRESH
                            Threshold of NMS (default: 0.45).
      --pre_processing      Do pre processing.
      --post_processing     Do post processing.

--------------------
仿真运行示例
--------------------

以下示例中使用到的 ``mobilenetv2.axmodel`` 和 ``yolov5s.axmodel`` 获取方式：

* 参考 :ref:`《模型编译》 <model_simulator>` 章节自行编译生成；
* 从 :ref:`《开发板运行》 <onboard_running>` 章节中提及到的 ``demo_onboard.zip`` 中获取预编译好的版本。

~~~~~~~~~~~~~~~~~~~~~
MobileNetv2
~~~~~~~~~~~~~~~~~~~~~

^^^^^^^^^^^^^^^^^^^^^
输入数据准备
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

    root@xxx:/data/pulsar2-run-helper# python3 cli_classification.py --pre_processing --image_path sim_images/cat.jpg --axmodel_path models/mobilenetv2.axmodel --intermediate_path sim_inputs/0
    [I] Write [input] to 'sim_inputs/0/input.bin' successfully.

^^^^^^^^^^^^^^^^^^^^^
仿真模型推理
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

    root@xxx:/data/pulsar2-run-helper# pulsar2 run --model models/mobilenetv2.axmodel --input_dir sim_inputs --output_dir sim_outputs --list list.txt
    Building native ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
    >>> [0] start
    write [output] to [sim_outputs/0/output.bin] successfully
    >>> [0] finish

^^^^^^^^^^^^^^^^^^^^^
输出数据处理
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

    root@xxx:/data/pulsar2-run-helper# python3 cli_classification.py --post_processing --axmodel_path models/mobilenetv2.axmodel --intermediate_path sim_outputs/0
    [I] The following are the predicted score index pair.
    [I] 9.5094, 285
    [I] 9.3773, 283
    [I] 9.2452, 281
    [I] 8.5849, 282
    [I] 7.6603, 463

~~~~~~~~~~~~~~~~~~~~~
YOLOv5s
~~~~~~~~~~~~~~~~~~~~~

^^^^^^^^^^^^^^^^^^^^^
输入数据准备
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

    root@xxx:/data/pulsar2-run-helper# python3 cli_detection.py --pre_processing --image_path sim_images/dog.jpg --axmodel_path models/yolov5s.axmodel --intermediate_path sim_inputs/0
    [I] Write [images] to 'sim_inputs/0/images.bin' successfully.

^^^^^^^^^^^^^^^^^^^^^
仿真模型推理
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

    root@xxx:/data/pulsar2-run-helper# pulsar2 run --model models/yolov5s.axmodel --input_dir sim_inputs/ --output_dir sim_outputs/ --list list.txt
    Building native ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
    >>> [0] start
    write [326] to [sim_outputs/0/326.bin] successfully
    write [370] to [sim_outputs/0/370.bin] successfully
    write [414] to [sim_outputs/0/414.bin] successfully
    >>> [0] finish

^^^^^^^^^^^^^^^^^^^^^
输出数据处理
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

    root@xxx:/data/pulsar2-run-helper# python3 cli_detection.py --post_processing --image_path sim_images/dog.jpg --axmodel_path models/yolov5s.axmodel --intermediate_path sim_outputs/0
    [I] Number of detected objects: 4
    [I] 16: 92.62%, [182, 291, 411, 721]
    [I]  2: 72.18%, [626, 101, 919, 231]
    [I]  1: 59.62%, [212, 158, 760, 558]
    [I]  7: 46.22%, [628, 101, 916, 232]
