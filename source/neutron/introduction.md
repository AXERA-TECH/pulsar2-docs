# 通元 NPU (Neutron) 介绍

(soc_introduction)=

通元 NPU（Neutron）是爱芯元智芯片平台中的神经网络计算引擎。不同 Neutron 版本在系统架构、计算核心数量、虚拟 NPU（vNPU）划分方式、离线编译 `npu_mode` 以及 AXEngine 运行时工作模式上存在差异。

vNPU 是对底层 NPU 计算资源的逻辑划分。模型编译时通过 `pulsar2 build` 的 `--npu_mode` 参数指定模型需要占用的计算资源规模；模型加载到芯片平台运行时，AXEngine 会根据当前 NPU 工作模式将模型调度到匹配的 NPU Core 或 vNPU 资源上执行。

本节先介绍 Neutron 架构中的基本名词和 EU 功能分工，再通过二级目录分别介绍各 Neutron 版本的 NPU 计算资源组织方式，最后通过芯片平台信息表提供具体芯片平台、可选 `npu_mode`、硬件资源规模以及 AXEngine 运行时工作模式之间对应关系的跳转入口。

## 硬件名词介绍

`NPU Core`

: NPU 的基本执行单元。一个模型在编译后会被拆分为一系列可执行任务，这些任务会被映射到一个或多个 NPU Core 上运行。每个 NPU Core 通常包含 EU、本地 OCM 和同步管理能力。

`EU`

: Execution Unit，执行具体计算或数据搬运的硬件单元。不同 EU 面向不同任务类型，例如卷积计算、张量/向量计算、数据搬运和数据整理等。

`Local OCM`

: On-Chip Memory，NPU Core 内的高速存储。相比 DDR，OCM 访问延迟更低、带宽更高，通常用于保存当前任务需要频繁访问的输入块、权重块、中间结果和临时数据。

`Shared OCM`

: 可被多个 NPU Core 或 vNPU 共享的片上存储空间，用于跨任务交换中间数据，减少通过 DDR 中转带来的额外搬运和访存开销。

`SyncManager`

: 同步管理模块，用于描述和管理 NPU Core、EU 以及子任务之间的执行依赖，确保并行执行时数据读写顺序正确。

`npu_mode`

: `pulsar2 build` 的离线编译参数，用于指定模型编译时面向的 NPU 资源规模。例如 `NPU1`、 `NPU2`、 `NPU3` 和 `NPU4` 表示不同规模的编译目标，但不表示具体的 NPU Core 编号或运行时 vNPU 编号。

`AXEngine 运行时工作模式`

: AXEngine 初始化阶段通过 `AX_ENGINE_Init` 的 `AX_ENGINE_NPU_ATTR_T.eHardMode` 配置的硬件资源划分方式。该配置决定运行时暴露给模型的 vNPU 或 NPU 资源形态，必须与待加载模型的 `npu_mode` 资源需求匹配。

## EU 类型介绍

EU 是 Neutron 中承担具体执行工作的硬件资源，按职责可分为计算类 EU 和数据搬运类 EU。编译器会根据算子类型、张量形状、数据布局、OCM 压力和目标硬件能力，将模型中的任务自动映射到合适的 EU 上，用户通常不需要手动选择 EU 类型。

`CONV`

: 卷积计算类 EU，面向深度学习中的高复用密集计算。典型任务包括卷积、反卷积、卷积后处理，以及可映射到卷积数据流的部分矩阵类计算。

`TENG`

: Tensor Engine，张量/向量计算类 EU，适合多通道并行的数据处理和通用张量计算。典型任务包括逐元素计算、激活、查表、归约、坐标/地址计算、部分数据重复与格式辅助处理等。

`SDMA`

: 数据搬运类 EU，负责在 DDR、OCM 以及片上存储区域之间移动数据。典型任务包括模型输入加载、权重加载、中间结果读写、输出写回，以及通道裁剪、通道填充、数据格式编码/解码等搬运相关任务。

`DAU`

: 计算类 EU，用于处理部分数据重排、筛选和排序类任务。典型任务包括 Padding、Transpose、TopK/Sort 等数据整理任务。

:::{note}
不同 Neutron 版本和具体芯片平台支持的 EU 数量、子功能和可用工作模式可能存在差异，实际能力以芯片 SPEC 和工具链支持情况为准。
:::

## 离线编译与运行时工作模式

`npu_mode` 和 AXEngine 运行时工作模式是两个层面的配置，需要配合使用：

- `npu_mode` 在离线编译阶段生效，决定编译器为模型生成的资源规模、任务切分方式和调度约束。编译完成后的 `axmodel` 会携带这类资源需求信息。
- AXEngine 运行时工作模式在程序初始化 NPU 时生效，决定底层硬件被划分为一个完整 NPU 资源、多个物理/逻辑 vNPU 资源，或多个 NPU Core 的调度资源。
- 运行时工作模式不能把一个已经按小资源编译的模型自动变成大资源模型，也不能让一个按大资源编译的模型在资源不足的运行时模式下执行。部署时应先确定模型的 `npu_mode`，再选择能提供匹配资源的 AXEngine 工作模式。

## Neutron 版本架构

各 Neutron 版本的详细资源组织方式见以下二级目录：

```{toctree}
:maxdepth: 1

v3/introduction
v4/introduction
v6/introduction
v7/introduction
```

(neutron_platform_npu_mode)=

### 芯片平台信息及可选 npu_mode

不同芯片平台对应的 Neutron 版本和 `pulsar2 build` 可选 `npu_mode` 参数如下。`npu_mode` 用于描述模型编译时使用的 NPU 资源规模，不表示具体的 NPU Core 或 vNPU 编号。

各平台 AXEngine 运行时工作模式、离线 `npu_mode` 与硬件计算资源的对应关系，请跳转到对应 Neutron 架构版本文档查看。

```{eval-rst}
.. list-table::
    :header-rows: 1
    :align: center
    :widths: 18 18 24 40

    * - 芯片平台
      - Neutron 版本
      - 可选 ``npu_mode``
      - 详细说明
    * - ``AX650``
      - :ref:`v3 <neutron_v3_architecture>`
      - ``NPU1`` / ``NPU2`` / ``NPU3``
      - :ref:`Neutron v3 工作模式与 npu_mode <neutron_v3_platform_npu_mode>`
    * - ``M76H``
      - :ref:`v3 <neutron_v3_architecture>`
      - ``NPU1`` / ``NPU2`` / ``NPU3``
      - :ref:`Neutron v3 工作模式与 npu_mode <neutron_v3_platform_npu_mode>`
    * - ``AX620E``
      - :ref:`v4 <neutron_v4_architecture>`
      - ``NPU1`` / ``NPU2``
      - :ref:`Neutron v4 工作模式与 npu_mode <neutron_v4_platform_npu_mode>`
    * - ``AX630C``
      - :ref:`v4 <neutron_v4_architecture>`
      - ``NPU1`` / ``NPU2``
      - :ref:`Neutron v4 工作模式与 npu_mode <neutron_v4_platform_npu_mode>`
    * - ``AX620Q``
      - :ref:`v4 <neutron_v4_architecture>`
      - ``NPU1`` / ``NPU2``
      - :ref:`Neutron v4 工作模式与 npu_mode <neutron_v4_platform_npu_mode>`
    * - ``AX615``
      - :ref:`v6 <neutron_v6_architecture>`
      - ``NPU1`` / ``NPU2``
      - :ref:`Neutron v6 工作模式与 npu_mode <neutron_v6_platform_npu_mode>`
    * - ``M57``
      - :ref:`v6 <neutron_v6_architecture>`
      - ``NPU1``
      - :ref:`Neutron v6 工作模式与 npu_mode <neutron_v6_platform_npu_mode>`
    * - ``AX637``
      - :ref:`v6 <neutron_v6_architecture>`
      - ``NPU1``
      - :ref:`Neutron v6 工作模式与 npu_mode <neutron_v6_platform_npu_mode>`
    * - ``AX8860``
      - :ref:`v7 <neutron_v7_architecture>`
      - ``NPU1`` / ``NPU2`` / ``NPU4``
      - :ref:`Neutron v7 工作模式与 npu_mode <neutron_v7_platform_npu_mode>`
```
