# Neutron v4 架构

(neutron_v4_architecture)=

:::{figure} ../../media/vNPU-ax620e.png
:align: center
:alt: Neutron v4 vNPU
:::

**Neutron v4** 由 **1** 个 NPU Core 构成，Core 内包含 **1** 个 `CONV`、 **1** 个 `TENG` 和 **1** 个 `SDMA`。与 Neutron v3 通过物理基础资源组合形成 vNPU 的方式不同，Neutron v4 通过硬件时分复用技术，在同一组物理 EU 上虚拟出 **2** 个 vNPU。

- `CONV` 和 `TENG` 属于计算类 EU，负责模型推理中的卷积、张量/向量计算等计算任务；`SDMA` 属于数据搬运类 EU，负责 DDR 与 OCM 之间的数据读写以及片上数据搬运。
- 硬件时分复用机制会在两个逻辑 vNPU 之间调度计算类 EU 和数据搬运类 EU，使不同推理任务的计算阶段与通信阶段交错运行。例如，一个 vNPU 进行 `CONV` / `TENG` 计算时，另一个 vNPU 可以通过 `SDMA` 准备输入、加载权重或写回结果，从而提高物理 EU 的整体利用率。启用时分复用后，两个逻辑 vNPU 会共享片上资源，单个 vNPU 可用的 OCM 会少于关闭时分复用时的完整 NPU 资源。

(neutron_v4_platform_npu_mode)=

## 工作模式与 npu_mode

### AXEngine 运行时工作模式

```{eval-rst}
.. list-table::
    :header-rows: 1
    :align: center
    :widths: 30 70

    * - AXEngine 工作模式
      - 计算资源说明
    * - ``AX_ENGINE_VIRTUAL_NPU_ENABLE``
      - 启用逻辑 vNPU 划分，在同一组物理 EU 上虚拟出 2 个 vNPU；两个逻辑 vNPU 共享物理 EU 和 OCM，单个 vNPU 可用 OCM 会减少。
    * - ``AX_ENGINE_VIRTUAL_NPU_DISABLE``
      - 关闭逻辑 vNPU 划分，不启用硬件时分复用；运行时暴露 1 个完整 NPU 资源，模型使用全部 NPU 资源推理。
```

### 离线 npu_mode 与硬件资源

```{eval-rst}
.. list-table::
    :header-rows: 1
    :align: center
    :widths: 18 18 64

    * - 芯片平台
      - ``npu_mode``
      - 硬件计算资源对应关系
    * - ``AX620E``
      - ``NPU1``
      - 时分复用模式下的 1 个逻辑 vNPU 编译资源，可用 OCM 少于完整 NPU 资源模式。
    * - ``AX620E``
      - ``NPU2``
      - 关闭时分复用时的完整 NPU 编译资源，使用全部 NPU 资源推理，可用 OCM 多于时分复用模式下的单个逻辑 vNPU。
    * - ``AX630C``
      - ``NPU1``
      - 时分复用模式下的 1 个逻辑 vNPU 编译资源，可用 OCM 少于完整 NPU 资源模式。
    * - ``AX630C``
      - ``NPU2``
      - 关闭时分复用时的完整 NPU 编译资源，使用全部 NPU 资源推理，可用 OCM 多于时分复用模式下的单个逻辑 vNPU。
    * - ``AX620Q``
      - ``NPU1``
      - 时分复用模式下的 1 个逻辑 vNPU 编译资源，可用 OCM 少于完整 NPU 资源模式。
    * - ``AX620Q``
      - ``NPU2``
      - 关闭时分复用时的完整 NPU 编译资源，使用全部 NPU 资源推理，可用 OCM 多于时分复用模式下的单个逻辑 vNPU。
```
