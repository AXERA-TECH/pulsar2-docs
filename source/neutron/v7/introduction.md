# Neutron v7 架构

(neutron_v7_architecture)=

**Neutron v7** 架构上进一步扩展为多 NPU Core 协同的 NPU 系统。与前述版本侧重单 Core 内 vNPU 划分不同，Neutron v7 更强调多 Core 并行、片上存储共享、任务依赖同步以及计算/搬运协同调度。

模型会经过编译器拆分为多个可执行任务，这些任务会被映射到一个或多个 NPU Core 上运行。每个 NPU Core 内部包含计算类 EU、数据搬运类 EU、Local OCM 和 SyncManager；多个 Core 之间可以通过 Shared OCM 和 SyncManager 进行数据交换与任务协同。

- **多 Core 并行**：模型子图或任务可以被调度到多个 NPU Core 上执行，提升并行度和吞吐能力。
- **片上存储复用**：Local OCM 用于 Core 内数据复用，Shared OCM 用于跨 Core 数据交换，从而减少对 DDR 的访问压力。

在 `AX8860` 平台上，`pulsar2 build` 的 `--npu_mode` 用于指定模型编译时占用的 NPU Core 数量。`NPU1`、 `NPU2` 和 `NPU4` 分别对应 1、2 和 4 个 NPU Core，不表示具体的 Core 编号。

(neutron_v7_platform_npu_mode)=

## 工作模式与 npu_mode

### AXEngine 运行时工作模式

```{eval-rst}
.. list-table::
    :header-rows: 1
    :align: center
    :widths: 22 78

    * - 芯片平台
      - 计算资源说明
    * - ``AX8860``
      - 待更新。
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
    * - ``AX8860``
      - ``NPU1``
      - 1 个 NPU Core；每个 Core 包含 4 个 ``CONV``、4 个 ``TENG``、2 个 ``SDMA``、4 个 ``DAU``。
    * - ``AX8860``
      - ``NPU2``
      - 2 个 NPU Core；每个 Core 包含 4 个 ``CONV``、4 个 ``TENG``、2 个 ``SDMA``、4 个 ``DAU``，支持跨 Core ``Shared OCM``。
    * - ``AX8860``
      - ``NPU4``
      - 4 个 NPU Core；每个 Core 包含 4 个 ``CONV``、4 个 ``TENG``、2 个 ``SDMA``、4 个 ``DAU``，支持跨 Core ``Shared OCM``。
```
