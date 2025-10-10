Polygraphy 工具介绍
====================

Polygraphy 是一个用于深度学习模型推理和验证的 Python 工具库，由 NVIDIA 开发者维护。它主要用于模型转换、性能分析和正确性验证，帮助开发者优化和调试。其 ``surgeon extract`` 功能可用于便捷的切分 QuantAxmodel 以定位出现问题的最小子图。

此工具可以灵活应用于基于 ONNX 容器格式表达的模型上，比如：ONNX、QuantAxModel、OptimizedAxModel。

常用参数
--------

.. code-block:: text

   --inputs INPUT_META [INPUT_META ...]
       子图的输入元数据（名称、形状、数据类型）
       使用 auto 可让 extract 自动推断这些值。
       格式：
       --inputs <名称>:<形状>:<数据类型>
       example:
       --inputs input0:[1,3,224,224]:float32 input1:auto:auto
       如果省略此参数，则默认使用当前模型的输入配置。

   --outputs OUTPUT_META [OUTPUT_META ...]
       子图的输出元数据（名称和数据类型）。
       使用 auto 可让 extract 自动推断这些值。
       格式：
       --outputs <名称>:<数据类型>
       example:
       --outputs output0:float32 output1:auto
       如果省略此参数，则默认使用当前模型的输出配置。

   -o SAVE_ONNX, --output SAVE_ONNX
       截取后的子图 onnx 保存路径

快速开始
--------

.. code-block:: bash

   polygraphy surgeon extract output/quant/quant_axmodel.onnx \
       --inputs your_inputs1:auto:auto your_inputs2:auto:auto \
       --outputs your_outputs1:auto:auto your_outputs2:auto:auto \
       -o output_quant_axmodel.onnx
