# 大模型编译

**本章节适用于平台**

- AX650A/AX650N/AX8850
- AX630C

**已验证模型**

- DeepSeek-R1-Distill
- Qwen2.5、Qwen3
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
本节示例模型为 `Qwen2.5-0.5B-Instruct-GPTQ-Int8`.

**版本约束**

本文档基于 Pulsar2 4.1 版本进行编写。

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
usage: pulsar2 llm_build [-h] [--input_path INPUT_PATH] [--output_path OUTPUT_PATH] [--prefill_len PREFILL_LEN] [--parallel PARALLEL] [--model_config MODEL_CONFIG]
                         [--kv_cache_len KV_CACHE_LEN] [--post_topk POST_TOPK] [--post_weight_type {bf16,s8,fp8_e5m2,fp8_e4m3}] [-t {fp16,bf16,fp32}]
                         [-w {fp16,bf16,fp32,s8,s4,fp8_e5m2,fp8_e4m3}] [-c CHECK_LEVEL] [--chip {AX620E,AX650,LAMBERT}] [--prompt PROMPT] [--image_size IMAGE_SIZE]
                         [--last_kv_cache_len LAST_KV_CACHE_LEN] [--tensor_parallel_size TENSOR_PARALLEL_SIZE]

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
```

## 下载 ax-llm-build 项目

```shell
git clone https://github.com/AXERA-TECH/ax-llm-build.git
```

## 下载 Qwen2.5-0.5B-Instruct-GPTQ-Int8

```shell
cd ax-llm-build
pip install -U huggingface_hub
huggingface-cli download --resume-download Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8 --local-dir Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8-ctx-ax650
```

## 编译执行

```shell
pulsar2 llm_build --input_path Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8/  --output_path Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8-ctx-ax650 --hidden_state_type bf16 --kv_cache_len 1023 --prefill_len 128 --last_kv_cache_len 128 --last_kv_cache_len 256 --last_kv_cache_len 384 --last_kv_cache_len 512  --chip AX650 -c 1 --parallel 8
```

### log 参考信息

```
pulsar2 llm_build --input_path Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8/  --output_path Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8-ctx-ax650 --hidden_state_type bf16 --kv_cache_len 1023 --prefill_len 128 --last_kv_cache_len 128 --last_kv_cache_len 256 --last_kv_cache_len 384 --last_kv_cache_len 512  --chip AX650 -c 1 --parallel 8
Config(
    model_name='Qwen2.5-0.5B-Instruct-GPTQ-Int8',
    model_type='qwen2',
    num_hidden_layers=24,
    num_attention_heads=14,
    num_key_value_heads=2,
    hidden_size=896,
    head_dim=0,
    intermediate_size=4864,
    vocab_size=151936,
    rope_theta=1000000.0,
    max_position_embeddings=32768,
    rope_partial_factor=1.0,
    rms_norm_eps=1e-06,
    norm_type='rms_norm',
    hidden_act='silu',
    hidden_act_param=0.03,
    scale_depth=1.4,
    scale_emb=1,
    dim_model_base=256,
    origin_model_type='',
    quant=True,
    quant_sym=True,
    quant_bits=8,
    quant_group_size=128,
    rs_factor=32,
    rs_high_freq_factor=4.0,
    rs_low_freq_factor=1.0,
    rs_original_max_position_embeddings=8192,
    rs_rope_type='',
    rs_mrope_section=[16, 24, 24]
)
2025-06-17 19:43:58.341 | SUCCESS  | yamain.command.llm_build:llm_build:179 - prepare llm model done!
building llm decode layers   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 24/24 0:01:57
building llm post layer   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 0:01:24
2025-06-17 19:47:20.855 | SUCCESS  | yamain.command.llm_build:llm_build:275 - build llm model done!
```

:::{note}
该示例所运行的主机配置为:

> - Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz
> - Memory 32G

全流程耗时大约 `6min` , 不同配置的主机转换时间略有差异.
:::

### embed 提取和优化

```shell
chmod +x ./tools/fp32_to_bf16
chmod +x ./tools/embed_process.sh
./tools/embed_process.sh Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8/ Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8-ctx-ax650/
```

### 输出文件说明

```shell
root@xxx:/data/ax-llm-build# tree Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8-ctx-ax650/
Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8-ctx-ax650/
├── model.embed_tokens.weight.bfloat16.bin
├── model.embed_tokens.weight.float32.bin # 临时文件，可删掉
├── model.embed_tokens.weight.npy # 临时文件，可删掉
├── qwen2_p128_l0_together.axmodel
├── qwen2_p128_l10_together.axmodel
├── qwen2_p128_l11_together.axmodel
├── qwen2_p128_l12_together.axmodel
├── qwen2_p128_l13_together.axmodel
├── qwen2_p128_l14_together.axmodel
├── qwen2_p128_l15_together.axmodel
├── qwen2_p128_l16_together.axmodel
├── qwen2_p128_l17_together.axmodel
├── qwen2_p128_l18_together.axmodel
├── qwen2_p128_l19_together.axmodel
├── qwen2_p128_l1_together.axmodel
├── qwen2_p128_l20_together.axmodel
├── qwen2_p128_l21_together.axmodel
├── qwen2_p128_l22_together.axmodel
├── qwen2_p128_l23_together.axmodel
├── qwen2_p128_l2_together.axmodel
├── qwen2_p128_l3_together.axmodel
├── qwen2_p128_l4_together.axmodel
├── qwen2_p128_l5_together.axmodel
├── qwen2_p128_l6_together.axmodel
├── qwen2_p128_l7_together.axmodel
├── qwen2_p128_l8_together.axmodel
├── qwen2_p128_l9_together.axmodel
└── qwen2_post.axmodel

0 directories, 28 files
```

其中 `model.embed_tokens.weight.bfloat16.bin`, `qwen2_p128_l0_together.axmodel ~ qwen2_p128_l23_together.axmodel`, `qwen_post.axmodel` 文件是上板运行所需要

## 开发板运行

本章节介绍如何在 `AX650` 开发板上运行 LLM 模型.

### 使用 ax-llm 运行大模型

运行该实例相关文件已上传网盘，请自行下载和参考

- [Huggingface](https://huggingface.co/AXERA-TECH/Qwen2.5-0.5B-Instruct-CTX-Int8)

先运行 tokenizer 解析器

```shell
root@ax650:/mnt/qtang/llm-test/qwen2.5-0.5b-ctx# python3 qwen2.5_tokenizer_uid.py
Server running at http://0.0.0.0:12345
```

再运行示例

```shell
root@ax650:/mnt/qtang/llm-test/qwen2.5-0.5b-ctx# ./run_qwen2.5_0.5b_gptq_int8_ctx_ax650.sh
[I][                            Init][ 110]: LLM init start
[I][                            Init][  34]: connect http://127.0.0.1:12345 ok
[I][                            Init][  57]: uid: d9e84259-87a2-4c54-9b9b-7da266149e8b
bos_id: -1, eos_id: 151645
100% | ████████████████████████████████ |  27 /  27 [10.21s<10.21s, 2.65 count/s] init post axmodel ok,remain_cmm(11292 MB)
[I][                            Init][ 188]: max_token_len : 1023
[I][                            Init][ 193]: kv_cache_size : 128, kv_cache_num: 1023
[I][                            Init][ 201]: prefill_token_num : 128
[I][                            Init][ 205]: grp: 1, prefill_max_token_num : 1
[I][                            Init][ 205]: grp: 2, prefill_max_token_num : 128
[I][                            Init][ 205]: grp: 3, prefill_max_token_num : 256
[I][                            Init][ 205]: grp: 4, prefill_max_token_num : 384
[I][                            Init][ 205]: grp: 5, prefill_max_token_num : 512
[I][                            Init][ 209]: prefill_max_token_num : 512
[I][                     load_config][ 282]: load config:
{
    "enable_repetition_penalty": false,
    "enable_temperature": false,
    "enable_top_k_sampling": true,
    "enable_top_p_sampling": false,
    "penalty_window": 20,
    "repetition_penalty": 1.2,
    "temperature": 0.9,
    "top_k": 1,
    "top_p": 0.8
}

[I][                            Init][ 218]: LLM init ok
Type "q" to exit, Ctrl+c to stop current running
[I][          GenerateKVCachePrefill][ 271]: input token num : 21, prefill_split_num : 1 prefill_grpid : 2
[I][          GenerateKVCachePrefill][ 308]: input_num_token:21
[I][                            main][ 230]: precompute_len: 21
[I][                            main][ 231]: system_prompt: You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
prompt >> who are you?
[I][                      SetKVCache][ 531]: prefill_grpid:2 kv_cache_num:128 precompute_len:21 input_num_token:12
[I][                      SetKVCache][ 534]: current prefill_max_token_num:384
[I][                             Run][ 660]: input token num : 12, prefill_split_num : 1
[I][                             Run][ 686]: input_num_token:12
[I][                             Run][ 829]: ttft: 135.66 ms
I am Qwen, a large language model created by Alibaba Cloud. I am a language model that can generate human-like text based on the input I receive.
I am designed to assist with a wide range of tasks, from simple questions to complex research papers, and I can even generate creative writing and speech.
I am here to help you with your queries and provide you with the information you need.

[N][                             Run][ 943]: hit eos,avg 34.04 token/s

[I][                      GetKVCache][ 500]: precompute_len:113, remaining:399
prompt >> q
root@ax650:/mnt/qtang/llm-test/qwen2.5-0.5b-ctx#
```

板端运行程序编译流程，请参考我们在 github 上的开源项目 [AX-LLM](https://github.com/AXERA-TECH/ax-llm)

### Tokenizer 解析器说明

ax-llm 项目中的 Tokenizer 解析器采用本地模块与 HTTP Server 两种方案，其中本地方案又尝试了 sentencepiece、tiktoken 两种方案。
但是我们在实际调试过程中发现 sentencepiece 对于不同 LLM 模型的 special tokens 支持不友好，需要用户自行处理 special tokens 的拆分，容易导致板端 token id 与 transformers 库中的 AutoTokenizer 获得的 token id 存在差异，最终影响 LLM 的输出结果正确性。
因此我们建议前期调试的时候使用 Tokenizer HTTP Server 的方式直接调用 transformers 库中的 AutoTokenizer 模块进行测试。

Tokenizer HTTP Server 的特点：

- 保证 token id 正确
- 方便添加 chat template
- 支持本地、远端部署
- 支持多用户接入

以在网盘中已提供基于 Qwen2.5 0.5B 的相关文件为例

```shell
root@ax650:/mnt/qtang/llm-test/qwen2.5-0.5b-ctx# tree -L 1
.
|-- main_ax650
|-- main_axcl_aarch64
|-- main_axcl_x86
|-- post_config.json
|-- qwen2.5-0.5b-gptq-int8-ctx-ax630c
|-- qwen2.5-0.5b-gptq-int8-ctx-ax650
|-- qwen2.5_tokenizer
|-- qwen2.5_tokenizer_uid.py
|-- run_qwen2.5_0.5b_gptq_int8_ctx_ax630c.sh
`-- run_qwen2.5_0.5b_gptq_int8_ctx_ax650.sh
```

- qwen2.5_tokenizer：是 tokenizer 相关文件，从 Qwen/Qwen2.5-3B-Instruct/ 中提取
- qwen2.5_tokenizer_uid.py：是用 python 实现的 Tokenizer HTTP Server

运行说明如下：

- python qwen2.5_tokenizer_uid.py --host xxx.xxx.xxx.xxx --port 12345，其中 --host xxx.xxx.xxx.xxx 设置 tokenizer 解析服务器的 IP 地址，确保 AX650N 能正常访问该地址
- 可以在具备 python 环境的 AX650N 本地运行, 则直接运行 python qwen2.5_tokenizer_uid.py
- 修改 ./run_qwen2.5_0.5b_gptq_int8_ctx_ax650.sh 中 --filename_tokenizer_model 的 IP 信息和步骤1中的一致
- 运行 ./run_qwen2.5_0.5b_gptq_int8_ctx_ax650.sh 即可

```shell
root@ax650:/mnt/qtang/llm-test/qwen2.5-0.5b-ctx# cat run_qwen2.5_0.5b_gptq_int8_ctx_ax650.sh
./main_ax650 \
--system_prompt "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." \
--template_filename_axmodel "qwen2.5-0.5b-gptq-int8-ctx-ax650/qwen2_p128_l%d_together.axmodel" \
--axmodel_num 24 \
--tokenizer_type 2 \
--url_tokenizer_model "http://127.0.0.1:12345" \
--filename_post_axmodel "qwen2.5-0.5b-gptq-int8-ctx-ax650/qwen2_post.axmodel" \
--filename_tokens_embed "qwen2.5-0.5b-gptq-int8-ctx-ax650/model.embed_tokens.weight.bfloat16.bin" \
--tokens_embed_num 151936 \
--tokens_embed_size 896 \
--use_mmap_load_embed 0 \
--live_print 1
```
