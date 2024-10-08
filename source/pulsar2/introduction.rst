========================================
Pulsar2 工具链概述
========================================

----------------------------
简介
----------------------------

**Pulsar2** 由 `爱芯元智 <https://www.axera-tech.com/>`_ **自主研发** 的 ``all-in-one`` 新一代神经网络编译器, 
即 **转换**、 **量化**、 **编译**、 **异构** 四合一, 实现深度学习神经网络模型 **快速**、 **高效** 的部署需求. 
针对新一代 `AX6、M7` 系列芯片（AX630C、AX620Q、AX650A、AX650N、M76H）特性进行了深度定制优化, 充分发挥片上异构计算单元(CPU+NPU)算力, 提升神经网络模型的产品部署效率.

**特别说明：**

- 工具链文档中的提示说明
   - **Note**: 注释内容，对某些专业词做进一步解释说明
   - **Hint**: 提示内容，提醒用户确认相关信息
   - **Attention**: 注意内容，提醒用户对工具配置的相关注意事项
   - **Warning**: 告警内容，提醒用户注意工具链的正确使用方法。如果客户没有按Warning提示内容进行使用，有可能会出现错误结果。
- 工具链文档中的命令兼容车载芯片，例如 ``Pulsar2`` 支持 ``M76H``
- 工具链文档中的 **示例命令**、 **示例输出** 均基于 ``AX650`` 进行展示
- 具体芯片的算力配置，以芯片SPEC为准

``Pulsar2`` 工具链核心功能是将 ``.onnx`` 模型编译成芯片能解析并运行的 ``.axmodel`` 模型.

**部署流程**

.. figure:: ../media/deploy-pipeline.png
    :alt: pipeline
    :align: center

.. _soc_introduction:

----------------------------
虚拟 NPU 介绍
----------------------------

.. figure:: ../media/vNPU-ax650.png
    :alt: pipeline
    :align: center

**AX650 和 M76H NPU** 主要由 **3** 个 Conv 卷积计算核，3 组向量 Vector 计算核组成。这些 Conv 和 Vector 计算核按照 1:1 的比例分配，划分为 **3 组 vNPU**。

- 在运行时，通过 **AXEngine API** 可以设置 NPU 的工作模式，灵活的对 vNPU 进行分组，可以设置为 1 + 1 + 1 的三个对称 vNPU 模式，或者 2 + 1 的大小 vNPU 模式，也可以设置为 3 的大算力单 vNPU 模式。

- 在转换模型时，可以根据需求灵活的指定模型推理所需的 vNPU 数量（详细信息请查看 ``pulsar2 build`` 的 ``--npu_mode 参数``）。当模型部署到芯片平台上加载时，AXEngine 可以根据当前设置的 NPU 工作模式将模型分配到对应算力的 vNPU 之上运行。

.. figure:: ../media/vNPU-ax620e.png
    :alt: pipeline
    :align: center

**AX630C、AX620Q** 采用双核 NPU 设计，根据 AI-ISP 是否启用划分成两种工况分配给用户不同算力。

- 在转换模型时，需根据实际业务中 AI-ISP 的工况显式配置用户模型的 NPU 工作模式（详细信息请查看 ``pulsar2 build`` 的 ``--npu_mode 参数``）。
- AX630C、AX620Q 中的 NPU 模块均采用爱芯元智 **通元4.0** NPU 引擎，后续章节使用 ``AX620E`` 简化目标硬件平台指定。

----------------------------
后续章节内容引导
----------------------------

* **Section3**: 本章介绍使用NPU工具链进行模型转换和部署的软硬件环境准备和安装。如何在不同系统环境下安装 ``Docker`` 并启动容器
* **Section4**：本章介绍NPU工具链在爱芯AX650（包括AX650A，AX650N，M76H）芯片平台上的基本应用流程
* **Section5**：本章介绍NPU工具链在爱芯AX620E（包括AX620Q，AX630C）芯片平台上的基本应用流程
* **Section6**：本章为模型转换的进阶说明，即详细介绍如何利用 ``Pulsar2 Docker`` 工具链将 ``onnx`` 模型转换为 ``axmodel`` 模型
* **Section7**: 本章为模型仿真的进阶说明，即详细介绍如何使用 ``axmodel`` 模型在 ``x86`` 平台上仿真运行并衡量推理结果与 ``onnx`` 推理结果之间的差异度(内部称之为 ``对分``)
* **Section8**: 本章为模型上板运行的进阶说明，即详细介绍如何上板运行 ``axmodel`` 得到模型在爱芯SOC硬件上的推理结果
* **Section9**: 本章对模型转换编译过程使用的配置文件进行详细说明
* **Section10**: Caffe AI训练平台导出的模型不是NPU工具链支持的 ``onnx`` 格式，需要一个工具把Caffe模型转换成 ``onnx`` 模型。本章介绍这个模型转换工具的使用方法。
* **Section11**: 本章为模型板上速度和精度测试工具的使用说明
* **Section12**: 本章为NPU工具链功能安全符合性的声明
* **附录**：文档附录部分包括算子支持列表、精度调优建议

.. note::

    所谓 ``对分``, 即对比工具链编译前后的同一个模型不同版本 (文件类型) 推理结果之间的误差。
