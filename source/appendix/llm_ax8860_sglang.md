# AX8860 大模型部署（SGLang）

`AX8860` 使用 `SGLang-axera` 部署大模型。官方已提供预置 NPU kernel 与推理适配，模型准备好后即可运行。

## 前置条件

- `AX8860P` 开发板已烧写 Debian rootfs，可正常联网
- SGLang 所需的 NPU kernel 已预先提供
- 已获取 Pulsar2 8.0 release 中对应的 aarch64 安装包（见下表）

## 需要准备的 release 包

| 文件 | 用途 |
| --- | --- |
| `sglang_axera-0.5.18-py3-none-any.whl` | SGLang 的 AX 适配 |
| `sgl_kernel_axera-0.1.0-py3-none-any.whl` | SGLang 自定义 kernel |
| `axnn-8.0-cp312-cp312-linux_aarch64.whl` | 编译器工具链（板端 aarch64） |
| `torch_axera-0.1.0-cp312-cp312-linux_aarch64.whl` | PyTorch 的 NPU 后端 |
| `pyaxcl-0.5.1-py3-none-any.whl` | AXCL Python 绑定 |

:::{note}
板端必须使用 `linux_aarch64` 版本的 wheel，不要使用 x86_64 版本。
:::

## 安装 AXCL 与 pyaxcl

:::{note}
AXCL 本体（驱动 / 运行库）的安装步骤待补充；若 Debian rootfs 已预装，可跳过。
:::

`pyaxcl` 与其余原生扩展一起安装：

```shell
uv pip install --no-deps pyaxcl-0.5.1-py3-none-any.whl
```

## 板端环境准备

### 安装 Miniforge3 并创建 Python 3.12 环境

```shell
# 安装 Miniforge3 后执行
conda create -y -n sglang python=3.12
conda activate sglang
pip install uv
```

:::{hint}
后续命令使用 `uv pip install`；没有 `uv` 时，替换为等价的 `pip install` 即可。
:::

### 安装软件栈

```shell
# 1) torch CPU 版
uv pip install torch==2.13.0+cpu torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cpu

# 2) sglang_axera
uv pip install sglang_axera-0.5.18-py3-none-any.whl
uv pip install transformers==5.12.1 pillow

# 3) 原生扩展 + kernel + AXCL 绑定
uv pip install --no-deps \
  axnn-8.0-cp312-cp312-linux_aarch64.whl \
  torch_axera-0.1.0-cp312-cp312-linux_aarch64.whl \
  sgl_kernel_axera-0.1.0-py3-none-any.whl \
  pyaxcl-0.5.1-py3-none-any.whl
```

:::{note}
第 3 步使用 `--no-deps`，避免 pip 重新解析依赖并覆盖前面固定版本的 `torch` / `transformers`。
:::

### 配置 swap（27B 模型必需）

```shell
fallocate -l 16G /home/arena/data/swapfile
chmod 600 /home/arena/data/swapfile
mkswap /home/arena/data/swapfile
swapon /home/arena/data/swapfile

# 开机自动挂载
echo "/home/arena/data/swapfile none swap sw 0 0" >> /etc/fstab

free -m   # Swap 行应看到约 16383
```

:::{hint}
`/home/arena/data/swapfile` 请替换为开发板上实际可用的磁盘路径。
:::

## 拷贝模型到开发板

:::{note}
本节内容正在补充中。
:::

<!-- TODO: 27B / 35B-A3B 模型文件清单、scp/rsync 示例、板端存放目录 -->

## 运行与 Benchmark

:::{note}
本节内容正在补充中。
:::

<!-- TODO: SGLang 启动命令、TTFT / decode 指标获取与评测方法 -->

## 启动 OpenAI 兼容服务

:::{note}
本节内容正在补充中。
:::

<!-- TODO: server 启动命令、端口、curl 调用示例 -->

## 与 llama.cpp 的差异

| 项目 | SGLang-axera | llama.cpp-axera |
| --- | --- | --- |
| 定位 | 服务化、OpenAI 兼容 API | 轻量单机、离线运行 |
| 适用场景 | 多用户、业务联调 | 单机验证、边缘设备 |

## 常见问题

<!-- TODO -->
