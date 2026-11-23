# 大模型部署（AX8860，SGLang）

`AX8860` 使用 `SGLang-axera` 部署大模型。官方已提供预置 NPU kernel 与推理适配，模型准备好后即可运行。

## 前置条件

- `AX8860P` 开发板已烧写 Debian rootfs（Debian 13 自带 `/usr/bin/python3.13`，可直接作为运行环境）
- SGLang 所需的 NPU kernel 已预先提供
- **AXCL 驱动 / 运行库 ≥ 0.7.0**（pyaxcl 0.7.0 起在运行时强制校验：板端 `libaxcl_rt.so` 为 0.6.0 会直接拒绝加载并报 `Incompatible AXCL 0.6.0; required >=0.7.0`）
- 已获取 Pulsar2 9.0 release 中对应的 aarch64 安装包（见下表）

## 需要准备的 release 包

| 文件 | 用途 |
| --- | --- |
| `sglang_axera-0.5.18-py3-none-any.whl` | SGLang 的 AX 适配 |
| `sgl_kernel_axera-0.1.0-py3-none-any.whl` | SGLang 自定义 kernel |
| `ax_kernels-0.1.0-py3-none-any.whl` | sgl_kernel_axera 的运行时依赖（纯 Python） |
| `axnn-9.0-cp313-cp313-linux_aarch64.whl` | 编译器工具链（板端 aarch64，Python 3.13） |
| `torch_axera-0.1.0-cp313-cp313-linux_aarch64.whl` | PyTorch 的 NPU 后端 |
| `pyaxcl-0.7.0-py3-none-any.whl` | AXCL Python 绑定 |

板端联网时，`pip install sglang_axera` 会自动从 PyPI 解析全部 Python 依赖（含 `triton`、`apache_tvm_ffi`、`llguidance`、`transformers` 等），无需逐一手装。其中 `torch` 请务必按下文步骤单独安装 `+cpu` 版本（PyPI 默认版会引入 CUDA 依赖）。

:::{note}
板端必须使用 `linux_aarch64` 版本的 wheel，不要使用 x86_64 版本。
:::

## 安装 AXCL 与 pyaxcl

:::{note}
AXCL 本体（驱动 / 运行库）的安装步骤待补充；若 Debian rootfs 已预装，可跳过。
:::

`pyaxcl` 与其余原生扩展一起安装：

```shell
pip install --no-deps pyaxcl-0.7.0-py3-none-any.whl
```

## 板端环境准备

### 创建 Python 3.13 venv

Pulsar2 9.0 起软件栈要求 Python ≥ 3.13（wheel 的 `Requires-Python` 为 `>=3.13,<3.14`）。Debian 13 rootfs 自带 3.13，直接建 venv 即可（无需 conda）：

```shell
/usr/bin/python3.13 -m venv $HOME/sglang-env313
source $HOME/sglang-env313/bin/activate
```

:::{hint}
若 rootfs 分区空间不足（`df -h $HOME` 确认），可将 venv 路径换到 `/data` 等大盘分区。
:::

### 安装软件栈

```shell
# 1) torch CPU 版（注意 +cpu 后缀与 --index-url，见上文备注）
pip install torch==2.13.0+cpu torchvision==0.28.0+cpu torchaudio==2.11.0+cpu \
  --index-url https://download.pytorch.org/whl/cpu

# 2) sglang_axera（自动解析 triton / apache_tvm_ffi / llguidance / transformers 等依赖）
pip install sglang_axera-0.5.18-py3-none-any.whl

# 3) 原生扩展 + kernel + AXCL 绑定
pip install --no-deps \
  axnn-9.0-cp313-cp313-linux_aarch64.whl \
  torch_axera-0.1.0-cp313-cp313-linux_aarch64.whl \
  sgl_kernel_axera-0.1.0-py3-none-any.whl \
  ax_kernels-0.1.0-py3-none-any.whl \
  pyaxcl-0.7.0-py3-none-any.whl
```

安装后做一次 import 冒烟：

```shell
LD_LIBRARY_PATH=/usr/local/axhelix/lib python -c \
  "import torch, axnn, torch_axera, sgl_kernel_axera, triton, tvm_ffi; import axnn.dsl as ax; ax.init(chip_num=1); print('smoke OK')"
```

:::{note}
- 第 3 步使用 `--no-deps`，避免 pip 重新解析依赖并覆盖前面固定版本的 `torch` / `transformers`。
- `torch_axera` 是 CPython 扩展（`_C.cpython-313-aarch64-linux-gnu.so`），换 Python 小版本需重编 wheel；`axnn` 的 `.so` 均为 ctypes 加载、无 CPython 绑定。

:::

### 离线部署环境

板端无法访问 PyPI 时，改用官方离线环境包（双架构一体包，x86/aarch64 全程离线安装）。

**1. 在可联网的主机上下载环境包**（[HF 仓库 `AXERA-TECH/Pulsar2` 的 9.0 版本](https://huggingface.co/AXERA-TECH/Pulsar2/9.0)）：

```shell
python3 -m pip install -U huggingface_hub
hf download AXERA-TECH/Pulsar2 --revision 9.0 \
  --include "axera-axnn-env*.tar.gz" --local-dir .
```

**2. 传到开发板**（文件较大，多流 scp 更快；传完务必 `stat` 校验尺寸一致，断流的文件会静默截断）：

```shell
scp axera-axnn-env-*.tar.gz root@<板ip>:/data/
```

**3. 板端解压安装**：

```shell
cd /data
tar xzf axera-axnn-env-*.tar.gz
cd axera-axnn-env
sh install.sh                 # 默认安装到解压目录本身
# 或指定安装目录：DATA=/opt/axera-env sh install.sh
```

`install.sh` 按 `uname -m` 自动选择架构内容：aarch64 板端使用包内自带的独立 CPython、uv 与离线 wheelhouse 安装全部依赖（包括 `triton`、`apache_tvm_ffi`、`llguidance` 等），不访问在线索引；`x86_64` 主机亦可使用同一包做编译侧环境。安装结果包含：

| 内容 | 说明 |
| --- | --- |
| `sglang-env/` | 就绪的 Python 环境（包内独立 CPython，不依赖系统 `python3.13`） |
| `start.sh` | 一键启动（smoke / server / chat / stop） |
| `scripts/` | 各模型运行脚本与示例 |
| `sglang/cache/` | BRUN kernel cache 位置（首次运行编译写入，同配置复用） |

:::{note}
- 环境包只含运行环境，**不含模型权重**，需按下一节另行拷贝。
- 与上一节"创建 venv + 联网安装"是两条等价路径，不要混用。
:::

## 拷贝模型到开发板
:::{note} 官方适配仓库 https://huggingface.co/collections/AXERA-TECH/ax8860-sglang ，内容正在逐渐补齐 :::
### Qwen3-0.6B-FP4

可先下载较小的 NVFP4 模型验证环境。Hugging Face 仓库为 [`NVFP4/Qwen3-0.6B-FP4`](https://huggingface.co/NVFP4/Qwen3-0.6B-FP4)，完整模型目录约 550 MB。

在开发板或可联网的主机上下载：

```shell
python3 -m pip install -U huggingface_hub
hf download NVFP4/Qwen3-0.6B-FP4 \
  --local-dir ./Qwen3-0.6B-FP4
```

若在主机下载，传到开发板：

```shell
ssh root@<板ip> 'mkdir -p /home/axera/sglang-axera/models'
rsync -avP ./Qwen3-0.6B-FP4/ \
  root@<板ip>:/home/axera/sglang-axera/models/Qwen3-0.6B-FP4/
```

## 启动服务

::: {note}
以下示例中的模型路径 `/home/axera/sglang-axera/models/Qwen3-0.6B-FP4` 请替换为开发板上的实际路径。
:::

```shell
RUN_MODE=BRUN \
SGLANG_PLATFORM=axera \
SGLANG_AXERA_INFERENCE_MODE=dynamic \
SGLANG_AXERA_KERNEL_CORE_NUM=4 \
python -m sglang.launch_server \
    --model-path /home/axera/sglang-axera/models/Qwen3-0.6B-FP4 \
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
  -d '{"model":"Qwen3-0.6B-FP4","messages":[{"role":"user","content":"你好"}]}'
```

:::{hint}
`model` 字段使用模型目录名（即 `--model-path` 的最后一级），如示例中的 `Qwen3-0.6B-FP4`。
:::

:::{note}
- **当板端运行模型内存不足时**，才需要配置 swap：

  ```shell
  fallocate -l 16G /data/swapfile
  chmod 600 /data/swapfile
  mkswap /data/swapfile
  swapon /data/swapfile
  # 开机自动挂载
  echo "/data/swapfile none swap sw 0 0" >> /etc/fstab
  free -m   # Swap 行应看到约 16383
  ```

  其中 `/data/swapfile` 请替换为开发板上实际可用的磁盘路径。
:::

## Benchmark 评测

服务启动后，另开一个终端执行：

```shell
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30001 \
  --num-prompts 5 \
  --model /home/axera/sglang-axera/models/Qwen3-0.6B-FP4 \
  --dataset-name random --random-range-ratio 1 \
  --random-input-len 128 --random-output-len 128 \
  --max-concurrency 1
```

上面的参数适合对 `Qwen3-0.6B-FP4` 做 smoke benchmark；测试较长上下文或持续吞吐时，再按目标场景调整输入和输出长度。

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

- **设备初始化或 NPU 调用失败**：确认 AXCL 运行库已注册（`ldconfig -p | grep axcl_rt`）且 `slave_daemon` 正在运行（`ps aux | grep '[s]lave_daemon'`）。
- **`torch_axera._C` 报 `undefined symbol: axclrtPeekAtLastError`**：板端 AXCL 运行库与当前 `torch_axera` wheel 不匹配或固件过旧，应检查并升级对应 AXCL release。
- **`RuntimeError: Incompatible AXCL 0.6.0; required >=0.7.0`**：pyaxcl 0.7.0 的硬校验。Pulsar2 9.0 栈要求板端 AXCL 驱动/固件 ≥ 0.7.0。
- **`ModuleNotFoundError: No module named 'triton'` / `'tvm_ffi'`**：`triton`、`apache_tvm_ffi` 会随 `sglang_axera` 自动安装；出现该错误通常是第 2 步被 `--no-deps` 跳过或依赖解析被中断，重新执行 `pip install triton==3.7.0 apache_tvm_ffi==0.1.15.post0` 即可。
- **`ModuleNotFoundError: No module named 'sgl_kernel'`**：检查 `SGLANG_USE_CPU_ENGINE` 是否被环境残留置 `1`。
- **`import torch` 报 `libcublasLt.so not found`**：装了 PyPI 版 torch（声明 CUDA 依赖）。换装 `download.pytorch.org/whl/cpu` 的 `+cpu` 版本。
- **`BRUN cache miss, compiling model`**：首次运行或修改输入 shape、context、page size 等参数时会触发编译；等待完成后，同一配置可复用缓存。
