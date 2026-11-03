# AX8860 大模型部署（llama.cpp）

`AX8860` 使用 `llama.cpp` + `ggml-axera` 在 4 核 NPU 上运行大模型。官方已提供预置 NPU kernel
与板端程序，模型下载后即可运行。

本文以 `Qwen3.8-27B-AX8860-llama.cpp` 为例说明，模型可从以下任意渠道获取：

- [Hugging Face](https://huggingface.co/AXERA-TECH/Qwen3.8-27B-AX8860-llama.cpp)
- [ModelScope](https://modelscope.cn/models/AXERA-TECH/Qwen3.8-27B-AX8860-llama.cpp)

## 适用范围

- 平台：`AX8860`
- SDK：V0.5.0 及以上（可通过 `cat /proc/ax_proc/version` 查看）

## 模型文件

| 文件 | 说明 |
| --- | --- |
| `qwen3.8-27b-mxfp4.gguf` | 模型权重（GGUF MXFP4，约 14.8 GB） |
| `mmproj-qwen3.8-27b-bf16.gguf` | 视觉编码器权重（BF16，约 931 MB） |
| `bin/` | llama.cpp 板端程序（静态链接，仅依赖系统 `/opt/lib` 与 libc） |
| `kernels/` | NPU kernel：decode、prefill、ViT、输出头各一个，以及 `manifest.json` |
| `run.sh` | 运行脚本 |

:::{note}
kernel 中不含权重，加载时会从 GGUF 转换到 NPU。
:::

## 运行

文本补全：

```shell
./run.sh "The capital of France is" 64
```

图像描述：

```shell
./run.sh -i bus.jpg "Describe this image." 256
```

启动 OpenAI 兼容服务：

```shell
./run.sh -s 8080
```

```shell
curl http://<board>:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好"}]}'
```

直接调用程序时需要设置：

- `LD_LIBRARY_PATH=/opt/lib`
- `GGML_AXERA_KERNEL_DIR=<repo>/kernels`
- `-ngl 99`
- 上下文 `-c` 最大 `16384`，服务端需 `-np 1`
- 设置 `GGML_AXERA_PROF=1` 可在退出时打印各阶段耗时

## 性能参考

测试条件：AX8860、SDK V0.5.1、NPU 4 核。

| 项目 | 结果 |
| --- | --- |
| 4k prompt（3908 token）prefill | 3.18 s（1231 tokens/s） |
| decode（4k 上下文） | 72.8 ms/token（13.7 tokens/s） |
| 图像编码（3072 patch = 768 图像 token） | 0.28 s |

## 已知限制

- 运行时独占 NPU 4 核，不能同时运行其他 NPU 程序
- 仅支持单序列，服务端必须使用 `-np 1`
- 最大上下文长度为 `16384`

## 与 SGLang-axera 的差异

| 项目 | SGLang-axera | llama.cpp-axera |
| --- | --- | --- |
| 定位 | 服务化、OpenAI 兼容 API | 轻量单机、离线运行 |
| 适用场景 | 多用户、业务联调 | 单机验证、边缘设备 |

## 相关链接

- 模型与程序：
  - [Hugging Face](https://huggingface.co/AXERA-TECH/Qwen3.8-27B-AX8860-llama.cpp)
  - [ModelScope](https://modelscope.cn/models/AXERA-TECH/Qwen3.8-27B-AX8860-llama.cpp)
- llama.cpp 源码：[ZHEQIUSHUI/llama.cpp](https://github.com/ZHEQIUSHUI/llama.cpp)
- NPU kernel 源码：[ZHEQIUSHUI/ax-dsl-kernels](https://github.com/ZHEQIUSHUI/ax-dsl-kernels)
