=======================
caffe2onnx 工具使用说明
=======================

本章介绍 AX 版 caffe2onnx 转换工具，用于将 Caffe 浮点模型转换成 ONNX 浮点模型。

.. note::

   下文的模型语义皆为浮点模型。

-----------------------------
将 Caffe 模型转换成 ONNX 模型
-----------------------------

我们提供了三种方式将 Caffe 模型转换成 ONNX 模型。

1. 您可以将 Caffe 文件传入进来以转换您指定的某一个 Caffe 模型：

   .. code:: bash

      python3 /opt/pulsar2/tools/convert_caffe_to_onnx.py
            --prototxt_path /path/to/your/model.prototxt
            --caffemodel_path /path/to/your/model.caffemodel
            --onnx_path /path/to/your/model.onnx  # optional
            --opset_version OPSET_VERSION  # default to ONNX opset 13

   一个 ".caffemodel" 跟其匹配的 ".prototxt" 文件一起组成一个 Caffe 模型，
   您需要同时指定 ``--caffemodel_path`` 和 ``--prototxt_path`` 这两个参数以确定一个
   Caffe 模型。 ``--onnx_path`` 跟 ``--opset_version`` 参数是可选的，
   其中 ``--opset_version`` 的缺省值为 13.

   .. note::

      如果您不指定 ``--onnx_path`` 命令行参数，生成的 ONNX 模型会
      使用 ".caffemodel" 模型文件（由 ``--caffemodel_path`` 指定）
      的前缀，并存放到和 ".caffemodel" 文件同一级的目录下。

2. 或者您也可以传入一个文件夹以转换其里面所有的 Caffe 模型：

   .. code:: bash

      python3 /opt/pulsar2/tools/convert_caffe_to_onnx.py
            --checkpoint_path /path/to/your/model/zoo
            --opset_version OPSET_VERSION  # default to ONNX opset 13

   这将递归地找到指定文件夹里面所有的以 ".caffemodel" 为后缀文件及其对应的
   ".prototxt" 文件，此为一个 Caffe 模型，将其转换为 ONNX 模型，
   并使用 Caffe 模型的前缀，以 ".onnx" 为后缀进行保存。

   .. note::

      Caffe 模型对应的 ".prototxt" 和 ".caffemodel"
      文件需要在同一个文件夹并共享一个前缀。

3. caffe2onnx 命令行工具

   新版工具链提供了 caffe2onnx 的命令行工具，也可以使用如下方式来转换模型。

   .. code:: bash

      caffe2onnx --convert --checkpoint_path /path/to/your/model/zoo

----------------------
验证转换出的 ONNX 模型
----------------------

您可以使用如下命令对分原始的 Caffe 模型和转换出的 ONNX 模型：

.. code:: bash

   python3 /opt/pulsar2/tools/validate_caffe_onnx.py
         --checkpoint_path /path/to/your/model/zoo

首先这将递归地找到指定文件夹里所有的以 ".onnx" 为后缀的文件，然后按照其前缀匹配对应的
".prototxt" 和 ".caffemodel" 文件，生成一个随机数据集，分别使用 ONNX Runtime 和
Caffe 推理工具进行推理，并计算两者的“相关系数 (Correlation)”、“标准偏差 (Standard Deviation)”、
“余弦距离相似度 (Cosine Similarity)”、“归一化相对误差 (Normalized Relative Error)”、
“最大差异 (Max Difference)” 和 “平均差异 (Mean Difference)”。

.. note::

   Caffe 模型对应的 ".prototxt" 和 ".caffemodel"
   文件以及转换出来的 ".onnx" 文件需要在同一个文件夹并共享一个前缀。

.. note::

   此步需要安装 caffe。

.. note::

   新版工具链提供了 caffe2onnx 的命令行工具，也可以使用如下方式来验证转换后的模型。

.. code:: bash

   caffe2onnx --validate --checkpoint_path /path/to/your/model/zoo
