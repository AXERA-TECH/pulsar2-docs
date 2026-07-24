# Neutron v3 架构

(neutron_v3_architecture)=

:::{figure} ../../media/vNPU-ax650.png
:align: center
:alt: Neutron v3 vNPU
:::

**Neutron v3** 由 **1** 个 NPU Core 构成，Core 内部包含 **6** 个 `CONV`、 **3** 个 `TENG` 和 **3** 个 `SDMA`。这些物理 EU 按 **3** 组基础资源组织，每组基础资源包含 **2** 个 `CONV`、 **1** 个 `TENG` 和 **1** 个 `SDMA`。

运行时可通过 `AX_ENGINE_Init` 的 `AX_ENGINE_NPU_ATTR_T.eHardMode` 设置虚拟 NPU 模式，决定这 3 组基础资源如何划分为 vNPU。使用 `Pulsar2` 编译器转换模型时可根据资源规模指定 `NPU1`、 `NPU2` 或 `NPU3`。

(neutron_v3_platform_npu_mode)=

## 工作模式与 npu_mode

### AXEngine 运行时工作模式

```{eval-rst}
.. list-table::
    :header-rows: 1
    :align: center
    :widths: 30 70

    * - AXEngine 工作模式
      - 计算资源说明
    * - ``AX_ENGINE_VIRTUAL_NPU_STD``
      - 1 + 1 + 1 划分方式，将 3 组基础资源划分为 3 个独立 vNPU；每个 vNPU 包含 2 个 ``CONV``、1 个 ``TENG``、1 个 ``SDMA``。
    * - ``AX_ENGINE_VIRTUAL_NPU_BIG_LITTLE`` / ``AX_ENGINE_VIRTUAL_NPU_LITTLE_BIG``
      - 2 + 1 或 1 + 2 划分方式；大 vNPU 占用 2 组基础资源，小 vNPU 占用 1 组基础资源。
    * - ``AX_ENGINE_VIRTUAL_NPU_DISABLE``
      - 3 的划分方式，不再拆分多个 vNPU，将全部 3 组基础资源作为一个完整 NPU 资源使用。
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
    * - ``AX650``
      - ``NPU1``
      - 1 组 Neutron v3 基础资源，包含 2 个 ``CONV``、1 个 ``TENG``、1 个 ``SDMA``。
    * - ``AX650``
      - ``NPU2``
      - 2 组 Neutron v3 基础资源，包含 4 个 ``CONV``、2 个 ``TENG``、2 个 ``SDMA``。
    * - ``AX650``
      - ``NPU3``
      - 3 组 Neutron v3 基础资源，包含 6 个 ``CONV``、3 个 ``TENG``、3 个 ``SDMA``。
    * - ``M76H``
      - ``NPU1``
      - 1 组 Neutron v3 基础资源，包含 2 个 ``CONV``、1 个 ``TENG``、1 个 ``SDMA``。
    * - ``M76H``
      - ``NPU2``
      - 2 组 Neutron v3 基础资源，包含 4 个 ``CONV``、2 个 ``TENG``、2 个 ``SDMA``。
    * - ``M76H``
      - ``NPU3``
      - 3 组 Neutron v3 基础资源，包含 6 个 ``CONV``、3 个 ``TENG``、3 个 ``SDMA``。
```
