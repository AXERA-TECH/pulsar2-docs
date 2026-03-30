# 大模型编译

**本章节适用于平台**

- AX650A/AX650N/AX8850
  \- SDK ≥ v3.6.2
- AX630C
  \- SDK ≥ v3.0.0

**已验证模型**

- Qwen3、Qwen2.5
- DeepSeek-R1-Distill
- MiniCPM4
- InternVL2_5、InternVL3
- ChatGLM3
- OpenBuddy
- SmolLM2
- Llama3.2
- Gemma2
- Phi2、Phi3
- TinyLlama

本章节介绍如何将 Huggingface 上的模型转换的基本操作, 使用 `pulsar2` 工具将从 Huggingface 下载的项目中 `*.safetensor` 或 `pytorch_model.bin` 模型编译成 `axmodel` 模型. 请先参考 {ref}`《开发环境准备》 <dev_env_prepare>` 章节完成开发环境搭建.
本节示例模型为 `Qwen3-0.6B`.

**版本约束**

本文档基于 Pulsar2 5.2 版本进行编写。

**LLM ModelZoo**

不定期更新业内关注度较高的大语言模型适配，包括预编译模型和上板运行示例。

- [Huggingface](https://huggingface.co/AXERA-TECH)

**关联项目 AX-LLM**

该项目用于探索业界常用 LLM(Large Language Model) 在已有芯片平台上落地的可行性和相关能力边界，方便社区开发者进行快速评估和二次开发自己的 LLM 应用。

- [AX-LLM](https://github.com/AXERA-TECH/ax-llm)

## 命令说明

`Pulsar2` 工具链中使用 `pulsar2 llm_build` 命令来完成 LLM 模型的转换.

```shell
root@xxx:/data# pulsar2 llm_build --help
usage: pulsar2 llm_build [-h] [--input_path INPUT_PATH] [--output_path OUTPUT_PATH] [--prefill_len PREFILL_LEN] [--parallel PARALLEL]
                         [--model_config MODEL_CONFIG] [--model_type MODEL_TYPE] [--kv_cache_len KV_CACHE_LEN] [--post_topk POST_TOPK]
                         [--post_weight_type {bf16,s8,fp8_e5m2,fp8_e4m3}] [-t {fp16,bf16,fp32}] [-w {fp16,bf16,fp32,s8,s4,fp8_e5m2,fp8_e4m3}] [-c CHECK_LEVEL]
                         [--chip {AX620E,AX650,LAMBERT}] [--prompt PROMPT] [--image_size IMAGE_SIZE] [--last_kv_cache_len LAST_KV_CACHE_LEN]
                         [--tensor_parallel_size TENSOR_PARALLEL_SIZE] [--ret_postnorm] [--ld_param_opt] [--npu_mode {NPU1,NPU2,NPU3}]

options:
  -h, --help            show this help message and exit
  --input_path INPUT_PATH
                        path of model or npy path (default: )
  --output_path OUTPUT_PATH
                        path of dumpped ax_model (default: .)
  --prefill_len PREFILL_LEN
                        token length of prefill (default: 0)
  --parallel PARALLEL   build parallel (default: 1)
  --model_config MODEL_CONFIG
                        config file (default: )
  --model_type MODEL_TYPE
                        config file (default: )
  --kv_cache_len KV_CACHE_LEN
                        length of kv_cache (default: 127)
  --post_topk POST_TOPK
                        post model output indices and prob (default: 0)
  --post_weight_type {bf16,s8,fp8_e5m2,fp8_e4m3}
                        post weight type (default: s8)
  -t {fp16,bf16,fp32}, --hidden_state_type {fp16,bf16,fp32}
                        hidden_state dtype (default: bf16)
  -w {fp16,bf16,fp32,s8,s4,fp8_e5m2,fp8_e4m3}, --weight_type {fp16,bf16,fp32,s8,s4,fp8_e5m2,fp8_e4m3}
                        weight dtype (default: s8)
  -c CHECK_LEVEL, --check_level CHECK_LEVEL
                        check level 0:run 1:layer_check 2: cal 1+1 (default: 0)
  --chip {AX620E,AX650,LAMBERT}
                        chip (default: AX650)
  --prompt PROMPT       prompt for check_level==2 (default: 1+1=)
  --image_size IMAGE_SIZE
                        vlm vision_part input_size (default: 224)
  --last_kv_cache_len LAST_KV_CACHE_LEN
                        last kv cache len (default: None)
  --tensor_parallel_size TENSOR_PARALLEL_SIZE
                        tensor parallel size (default: 0)
  --ret_postnorm        weather to return post_norm value in post layer (default: False)
  --ld_param_opt        ld_param_opt (default: False)
  --npu_mode {NPU1,NPU2,NPU3}
```

## 下载 ax-llm-build 项目

如需将原始 Huggingface 模型自行编译为 `axmodel` 文件，可使用 `ax-llm-build` 项目提供的辅助脚本完成下载、embed 处理等操作。
若直接使用 [AXERA-TECH](https://huggingface.co/AXERA-TECH) 已提供的预编译模型目录在板端运行，则本步骤可跳过。

```shell
git clone https://github.com/AXERA-TECH/ax-llm-build.git
```

## 下载 Qwen3-0.6B

```shell
cd ax-llm-build
pip install -U huggingface_hub
hf download Qwen/Qwen3-0.6B --local-dir Qwen/Qwen3-0.6B
```

## 编译执行

```shell
pulsar2 llm_build --input_path Qwen/Qwen3-0.6B/  --output_path Qwen/Qwen3-0.6B-ax650 --hidden_state_type bf16 --kv_cache_len 1023 --prefill_len 128 --last_kv_cache_len 128 --last_kv_cache_len 256 --last_kv_cache_len 384 --last_kv_cache_len 512  --chip AX650 -c 1 --parallel 8
```

### log 参考信息

```
pulsar2 llm_build --input_path Qwen/Qwen3-0.6B/  --output_path Qwen/Qwen3-0.6B-ax650 --hidden_state_type bf16 --kv_cache_len 1023 --prefill_len 128 --last_kv_cache_len 128 --last_kv_cache_len 256 --last_kv_cache_len 384 --last_kv_cache_len 512  --chip AX650 -c 1 --parallel 8
Config(
    model_name='Qwen3-0.6B',
    model_type='qwen3',
    num_hidden_layers=28,
    num_attention_heads=16,
    num_key_value_heads=8,
    hidden_size=1024,
    head_dim=128,
    intermediate_size=3072,
    vocab_size=151936,
    rope_theta=1000000,
    max_position_embeddings=40960,
    rope_partial_factor=1.0,
    rope_local_base_freq=None,
    rms_norm_eps=1e-06,
    norm_type='rms_norm',
    hidden_act='silu',
    hidden_act_param=0.03,
    scale_depth=1.4,
    scale_emb=1,
    dim_model_base=256,
    origin_model_type='',
    quant=False,
    quant_sym=False,
    quant_bits=4,
    quant_group_size=128,
    rs_factor=32,
    rs_high_freq_factor=4.0,
    rs_low_freq_factor=1.0,
    rs_original_max_position_embeddings=8192,
    rs_rope_type='',
    rs_alpha=None,
    rs_beta_fast=None,
    rs_beta_slow=None,
    rs_mscale=None,
    rs_mscale_all_dim=None,
    rs_mrope_section=[16, 24, 24],
    interleaved_mrope=False,
    use_qk_norm=False,
    qk_norm_after_rope=False,
    layer_types=[],
    kv_cache_len=1023
)
2026-03-23 21:05:42.252 | SUCCESS  | yamain.command.llm_build:llm_build:258 - prepare llm model done!
building llm decode layers   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 28/28     0:02:50
building llm post layer   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1     0:01:23
2026-03-23 21:09:56.131 | SUCCESS  | yamain.command.llm_build:llm_build:368 - build llm model done!
2026-03-23 21:10:01.591 | INFO     | yamain.command.llm_build:llm_build:519 - decode layer0_gt layer0_got cos_sim is: 1.0
2026-03-23 21:10:12.356 | INFO     | yamain.command.llm_build:llm_build:553 - prefill layer0_gt layer0_got cos_sim is: 1.0
2026-03-23 21:10:12.357 | SUCCESS  | yamain.command.llm_build:llm_build:578 - check llm model done!
```

:::{note}
该示例所运行的主机配置为:

> - Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz
> - Memory 32G

全流程耗时大约 `5min` , 不同配置的主机转换时间略有差异.
:::

### embed 提取和优化

```shell
chmod +x ./tools/fp32_to_bf16
chmod +x ./tools/embed_process.sh
./tools/embed_process.sh Qwen/Qwen3-0.6B/ Qwen/Qwen3-0.6B-ax650/
```

### 输出文件说明

```shell
root@xxx:/data/ax-llm-build# tree Qwen/Qwen3-0.6B-ax650/
Qwen/Qwen3-0.6B-ax650/
├── model.embed_tokens.weight.bfloat16.bin
├── model.embed_tokens.weight.float32.bin   # 临时文件，可删掉
├── model.embed_tokens.weight.npy           # 临时文件，可删掉
├── qwen3_p128_l0_together.axmodel
├── qwen3_p128_l10_together.axmodel
├── qwen3_p128_l11_together.axmodel
├── qwen3_p128_l12_together.axmodel
├── qwen3_p128_l13_together.axmodel
├── qwen3_p128_l14_together.axmodel
├── qwen3_p128_l15_together.axmodel
├── qwen3_p128_l16_together.axmodel
├── qwen3_p128_l17_together.axmodel
├── qwen3_p128_l18_together.axmodel
├── qwen3_p128_l19_together.axmodel
├── qwen3_p128_l1_together.axmodel
├── qwen3_p128_l20_together.axmodel
├── qwen3_p128_l21_together.axmodel
├── qwen3_p128_l22_together.axmodel
├── qwen3_p128_l23_together.axmodel
├── qwen3_p128_l24_together.axmodel
├── qwen3_p128_l25_together.axmodel
├── qwen3_p128_l26_together.axmodel
├── qwen3_p128_l27_together.axmodel
├── qwen3_p128_l2_together.axmodel
├── qwen3_p128_l3_together.axmodel
├── qwen3_p128_l4_together.axmodel
├── qwen3_p128_l5_together.axmodel
├── qwen3_p128_l6_together.axmodel
├── qwen3_p128_l7_together.axmodel
├── qwen3_p128_l8_together.axmodel
├── qwen3_p128_l9_together.axmodel
└── qwen3_post.axmodel

0 directories, 32 files
```

其中 `model.embed_tokens.weight.bfloat16.bin`, `qwen3_p128_l0_together.axmodel ~ qwen3_p128_l27_together.axmodel`, `qwen3_post.axmodel` 文件是上板运行所需要

## 开发板运行

本章节介绍如何在 `AX650` 开发板上运行 LLM 模型.

### 安装 axllm

推荐使用 `ax-llm` 项目提供的安装脚本在板端直接完成安装:

```shell
curl -fsSL https://raw.githubusercontent.com/AXERA-TECH/ax-llm/axllm/install.sh | bash
```

安装完成后, 可直接执行如下命令确认 `axllm` 已成功安装:

```shell
root@ax650:~/llm-test# axllm --help
Usage:
  axllm run <model_path> [options]    Run interactive chat mode
  axllm serve <model_path> [options]  Run HTTP API server mode

Arguments:
  model_path    Path to model directory containing config.json and model files

Serve options:
  --port <port> Server port (default: 8080)

Model directory structure:
  model_path/
    ├── config.json          # Model configuration
    ├── tokenizer.txt        # Tokenizer model
    ├── *.axmodel            # AXera model files
    └── post_config.json     # Post-processing config (optional)
```

如需了解手动编译方式, 可参考 [AX-LLM](https://github.com/AXERA-TECH/ax-llm) 项目说明。

### 使用 ax-llm 运行大模型

运行该实例相关文件可从 Huggingface 上直接下载，当前 ModelZoo 已将 `tokenizer` 相关文件一并放入对应模型仓库中，例如：

> - [Qwen3-0.6B](https://huggingface.co/AXERA-TECH/Qwen3-0.6B)

因此当前版本不再需要单独运行 tokenizer 解析器，只需下载对应 Huggingface 模型目录后，直接将该目录作为参数传给 `axllm run` 或 `axllm serve` 即可自动加载运行，整体使用流程相比旧版本更为简洁。

以 `AXERA-TECH/Qwen3-0.6B` 为例:

```shell
pip install -U huggingface_hub
hf download AXERA-TECH/Qwen3-0.6B --local-dir Qwen3-0.6B
```

### 命令行运行

```shell
root@ax650:~/llm-test# axllm run Qwen3-0.6B/
[I][                            Init][ 138]: LLM init start
tokenizer_type = 1
 96% | ███████████████████████████████   |  30 /  31 [3.25s<3.36s, 9.23 count/s] init post axmodel ok,remain_cmm(8662 MB)
[I][                            Init][ 199]: max_token_len : 2559
[I][                            Init][ 202]: kv_cache_size : 1024, kv_cache_num: 2559
[I][                            Init][ 205]: prefill_token_num : 128
[I][                            Init][ 209]: grp: 1, prefill_max_kv_cache_num : 1
[I][                            Init][ 209]: grp: 2, prefill_max_kv_cache_num : 512
[I][                            Init][ 209]: grp: 3, prefill_max_kv_cache_num : 1024
[I][                            Init][ 209]: grp: 4, prefill_max_kv_cache_num : 1536
[I][                            Init][ 209]: grp: 5, prefill_max_kv_cache_num : 2048
[I][                            Init][ 214]: prefill_max_token_num : 2048
[I][                            Init][  27]: LLaMaEmbedSelector use mmap
100% | ████████████████████████████████ |  31 /  31 [3.25s<3.25s, 9.54 count/s] embed_selector init ok
[I][                     load_config][ 282]: load config:
{
    "enable_repetition_penalty": false,
    "enable_temperature": false,
    "enable_top_k_sampling": false,
    "enable_top_p_sampling": false,
    "penalty_window": 20,
    "repetition_penalty": 1.2,
    "temperature": 0.9,
    "top_k": 10,
    "top_p": 0.8
}

[I][                            Init][ 272]: LLM init ok
Type "q" to exit
Ctrl+c to stop current running
"reset" to reset kvcache
"dd" to remove last conversation.
"pp" to print history.
----------------------------------------
prompt >> who are you?
[I][                      SetKVCache][ 406]: prefill_grpid:2 kv_cache_num:512 precompute_len:0 input_num_token:23
[I][                      SetKVCache][ 408]: current prefill_max_token_num:2048
[I][                      SetKVCache][ 409]: first run
[I][                             Run][ 457]: input token num : 23, prefill_split_num : 1
[I][                             Run][ 497]: prefill chunk p=0 history_len=0 grpid=1 kv_cache_num=0 input_tokens=23
[I][                             Run][ 519]: prefill indices shape: p=0 idx_elems=128 idx_rows=1 pos_rows=0
[I][                             Run][ 627]: ttft: 173.71 ms
<think>
Okay, the user asked, "Who are you?" I need to respond appropriately. Let me start by acknowledging their question. I should mention that I'm an AI assistant     designed to help with various tasks. It's important to keep the response friendly and open-ended so they feel comfortable sharing more. I should make sure to     highlight that I'm here to assist and that I'm not a person. Let me check if there's any additional information I should include to make the response more     helpful. Alright, that should cover it.
</think>

I'm an AI assistant designed to help with a wide range of tasks and questions. I'm here to assist you with anything you need! Let me know how I can help!

[N][                             Run][ 709]: hit eos,avg 15.68 token/s

[I][                      GetKVCache][ 380]: precompute_len:168, remaining:1880
prompt >> q
```

### 服务运行

`axllm` 支持直接将模型目录启动为兼容 OpenAI API 的服务，这种方式对于二次开发和业务联调更为方便。

```shell
root@ax650:~/llm-test# axllm serve Qwen3-0.6B/
[I][                            Init][ 138]: LLM init start
tokenizer_type = 1
 96% | ███████████████████████████████   |  30 /  31 [2.65s<2.74s, 11.30 count/s] init post axmodel ok,remain_cmm(8662 MB)
[I][                            Init][ 199]: max_token_len : 2559
[I][                            Init][ 202]: kv_cache_size : 1024, kv_cache_num: 2559
[I][                            Init][ 205]: prefill_token_num : 128
[I][                            Init][ 209]: grp: 1, prefill_max_kv_cache_num : 1
[I][                            Init][ 209]: grp: 2, prefill_max_kv_cache_num : 512
[I][                            Init][ 209]: grp: 3, prefill_max_kv_cache_num : 1024
[I][                            Init][ 209]: grp: 4, prefill_max_kv_cache_num : 1536
[I][                            Init][ 209]: grp: 5, prefill_max_kv_cache_num : 2048
[I][                            Init][ 214]: prefill_max_token_num : 2048
[I][                            Init][  27]: LLaMaEmbedSelector use mmap
100% | ████████████████████████████████ |  31 /  31 [2.65s<2.65s, 11.68 count/s] embed_selector init ok
[I][                     load_config][ 282]: load config:
{
    "enable_repetition_penalty": false,
    "enable_temperature": false,
    "enable_top_k_sampling": false,
    "enable_top_p_sampling": false,
    "penalty_window": 20,
    "repetition_penalty": 1.2,
    "temperature": 0.9,
    "top_k": 10,
    "top_p": 0.8
}

[I][                            Init][ 272]: LLM init ok
Starting server on port 8000 with model 'AXERA-TECH/Qwen3-0.6B'...
OpenAI API Server starting on http://0.0.0.0:8000
Max concurrency: 1
Models: AXERA-TECH/Qwen3-0.6B
```

服务启动完成后，即可通过标准 OpenAI 兼容接口进行调用。

### API 调用示例

服务启动完成后，可直接使用标准 HTTP 请求访问 OpenAI 兼容接口。最简调用命令如下：

```shell
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"AXERA-TECH/Qwen3-0.6B","messages":[{"role":"user","content":"你好"}]}'
```

如需使用 `ax-llm` 项目中的示例脚本进行测试，可参考如下方式：

```shell
root@ax650:~/llm-test# curl -sOL https://raw.githubusercontent.com/AXERA-TECH/ax-llm/refs/heads/axllm/scripts/openai_demo.py

root@ax650:~/llm-test# python openai_demo.py --model AXERA-TECH/Qwen3-0.6B --api_url http://127.0.0.1:8000/v1
assistant:
<think>
Okay, the user just said "hello". I need to respond appropriately. Since they're greeting me, I should acknowledge their greeting. Maybe say "Hello!" in a friendly way. Let me check if there's any specific context I should consider, but the user didn't mention anything else. I should keep it simple and welcoming. Alright, time to send a response.
</think>

Hello! How can I assist you today? 😊
```

如需自定义 prompt，可参考如下命令：

```shell
root@ax650:~/llm-test# python openai_demo.py --model AXERA-TECH/Qwen3-0.6B --api_url http://127.0.0.1:8000/v1 --prompt "请介绍一下你自己"
```

需要说明的是，`openai_demo.py` 仅作为接口调用示例，实际应用中推荐直接按照 OpenAI 接口规范进行调用对接。

板端运行程序编译流程，以及更多 `run` / `serve` / API 使用细节，请参考我们在 Github 上的开源项目 [AX-LLM](https://github.com/AXERA-TECH/ax-llm)
