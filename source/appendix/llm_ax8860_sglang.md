# 大模型部署（AX8860，SGLang）

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
pip install --no-deps pyaxcl-0.5.1-py3-none-any.whl
```

## 板端环境准备

### 安装 Miniforge3 并创建 Python 3.12 环境

```shell
# 下载并安装 Miniforge3（板端为 aarch64）
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh -b -p $HOME/miniforge3

# 初始化当前 shell
source $HOME/miniforge3/etc/profile.d/conda.sh

# 创建并激活 Python 3.12 环境
conda create -y -n sglang python=3.12
conda activate sglang
```

### 安装软件栈

```shell
# 1) torch CPU 版
pip install torch==2.13.0+cpu torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cpu

# 2) sglang_axera
pip install sglang_axera-0.5.18-py3-none-any.whl
pip install transformers==5.12.1 pillow

# 3) 原生扩展 + kernel + AXCL 绑定
pip install --no-deps \
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

<!-- TODO: 27B 模型文件清单、scp/rsync 示例、板端存放目录 -->

## 启动服务

::: {note}
以下示例中的模型路径 `/home/ai01/sglang-axera/models/Qwen3.8-27B-NVFP4` 请替换为开发板上的实际路径。
:::

```shell
RUN_MODE=BRUN \
SGLANG_PLATFORM=axera \
SGLANG_AXERA_INFERENCE_MODE=dynamic \
SGLANG_AXERA_KERNEL_CORE_NUM=4 \
python -m sglang.launch_server \
    --model-path /home/ai01/sglang-axera/models/Qwen3.8-27B-NVFP4 \
    --device axera \
    --attention-backend axera \
    --kv-cache-dtype bfloat16 \
    --context-length 2044 \
    --max-total-tokens 4096 \
    --max-running-requests 1 \
    --chunked-prefill-size 256 \
    --page-size 256 \
    --host 127.0.0.1 \
    --port 30001 \
    --sampling-defaults openai \
    --disable-radix-cache \
    --disable-cuda-graph \
    --watchdog-timeout 7200
```

关键环境变量：

| 变量 | 说明 |
| --- | --- |
| `RUN_MODE=BRUN` | 板端运行模式 |
| `SGLANG_PLATFORM=axera` | 指定 AX 平台 |
| `SGLANG_AXERA_INFERENCE_MODE=dynamic` | 动态推理模式 |
| `SGLANG_AXERA_KERNEL_CORE_NUM=4` | 使用 4 个 NPU Core |

服务启动后即可通过 OpenAI 兼容接口调用：

```shell
curl http://127.0.0.1:30001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-27B-NVFP4","messages":[{"role":"user","content":"你好"}]}'
```

:::{hint}
`model` 字段使用模型目录名（即 `--model-path` 的最后一级），如示例中的 `Qwen3.8-27B-NVFP4`。
:::

## Benchmark 评测

服务启动后，另开一个终端执行：

```shell
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30001 \
  --num-prompts 1 \
  --model /home/ai01/sglang-axera/models/Qwen3.8-27B-NVFP4 \
  --dataset-name random --random-range-ratio 1 \
  --random-input-len 2048 --random-output-len 2048 \
  --max-concurrency 1
```

评测结果重点看以下指标：

| 指标 | 说明 |
| --- | --- |
| `Mean TTFT` | 首 token 延迟，反映 prefill 性能 |
| `Mean TPOT` / `Mean ITL` | 相邻输出 token 的间隔，反映 decode 性能 |
| `Output token throughput` | 输出 token 吞吐（tokens/s） |
| `Request throughput` | 每秒完成的请求数 |

## 与 llama.cpp 的差异

| 项目 | SGLang-axera | llama.cpp-axera |
| --- | --- | --- |
| 定位 | 服务化、OpenAI 兼容 API | 轻量单机、离线运行 |
| 适用场景 | 多用户、业务联调 | 单机验证、边缘设备 |

## 常见问题

<!-- TODO -->
