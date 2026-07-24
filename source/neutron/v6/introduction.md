# Neutron v6 架构

(neutron_v6_architecture)=

:::{figure} ../../media/vNPU-ax615.png
:align: center
:alt: Neutron v6 vNPU
:::

**Neutron v6** 基于 **Neutron v4** 架构对功耗及面积进行迭代优化，整体延续 **1** 个 NPU Core、 **1** 个 `CONV`、 **1** 个 `TENG`、 **1** 个 `SDMA` 以及可选的双 vNPU 时分复用资源组织方式。

- 与 Neutron v4 类似，Neutron v6 仍将 EU 分为计算类和数据搬运类两类。启用时分复用后，两个逻辑 vNPU 可以让不同推理任务的计算阶段与通信阶段交错运行，同时两个逻辑 vNPU 会共享片上资源，单个 vNPU 可用的 OCM 会少于关闭时分复用时的完整 NPU 资源。
- 部分 Neutron v6 平台仅开放单 NPU 资源模式，实际可选项请参考下方工作模式与 `npu_mode` 说明。

(neutron_v6_platform_npu_mode)=

## 工作模式与 npu_mode

### AXEngine 运行时工作模式

```{eval-rst}
.. list-table::
    :header-rows: 1
    :align: center
    :widths: 18 30 52

    * - 芯片平台
      - AXEngine 工作模式
      - 计算资源说明
    * - ``AX615``
      - ``AX_ENGINE_VIRTUAL_NPU_ENABLE``
      - 启用逻辑 vNPU 划分，在同一 NPU Core 上通过硬件时分复用调度双 vNPU 资源；两个逻辑 vNPU 共享片上资源，单个 vNPU 可用 OCM 会减少。
    * - ``AX615``
      - ``AX_ENGINE_VIRTUAL_NPU_DISABLE``
      - 关闭逻辑 vNPU 划分，不启用硬件时分复用；运行时暴露 1 个完整 NPU 资源，模型使用全部 NPU 资源推理。
    * - ``M57``
      - ``AX_ENGINE_VIRTUAL_NPU_DISABLE``
      - 仅支持单 NPU 资源模式。
    * - ``AX637``
      - ``AX_ENGINE_VIRTUAL_NPU_DISABLE``
      - 仅支持单 NPU 资源模式。
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
    * - ``AX615``
      - ``NPU1``
      - 时分复用模式下的 1 个逻辑 vNPU 编译资源，可用 OCM 少于完整 NPU 资源模式。
    * - ``AX615``
      - ``NPU2``
      - 关闭时分复用时的完整 NPU 编译资源，使用全部 NPU 资源推理，可用 OCM 多于时分复用模式下的单个逻辑 vNPU。
    * - ``M57``
      - ``NPU1``
      - 1 个 NPU Core，包含 1 个 ``CONV``、1 个 ``TENG``、1 个 ``DAU``、1 个 ``SDMA``。
    * - ``AX637``
      - ``NPU1``
      - 1 个 NPU Core，包含 1 个 ``CONV``、1 个 ``TENG``、1 个 ``DAU``、1 个 ``SDMA``。
```
