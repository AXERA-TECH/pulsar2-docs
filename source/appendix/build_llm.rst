======================
大模型编译(实验阶段)
======================

**本章节适用于平台**

- AX650N
- AX630C

本章节介绍如何将 Huggingface 上的模型转换的基本操作, 使用 ``pulsar2`` 工具将从 Huggingface 下载的项目中 ``*.safetensor`` 或 ``pytorch_model.bin``  模型编译成 ``axmodel`` 模型. 请先参考 :ref:`《开发环境准备》 <dev_env_prepare>` 章节完成开发环境搭建. 
本节示例模型为 ``Qwen2-0.5B-Instruct``.

**版本约束**

Pulsar2 3.0 以上版本已内置 llm build 相关模块。

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
命令说明
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Pulsar2`` 工具链中使用 ``pulsar2 llm_build`` 命令来完成 LLM 模型的转换. 

.. code-block:: shell

    root@xxx:/data# pulsar2 llm_build --help
    usage: pulsar2 llm_build [-h] [--input_path INPUT_PATH] [--output_path OUTPUT_PATH] [--prefill_len PREFILL_LEN] [--parallel PARALLEL] [--model_config MODEL_CONFIG]
                            [--kv_cache_len KV_CACHE_LEN] [--post_topk POST_TOPK] [--post_weight_type {bf16,s8}] [-t {fp16,bf16,fp32}] [-w {fp16,bf16,fp32,s8,s4}] [-c CHECK_LEVEL]
                            [--chip {AX620E,AX650}]

    optional arguments:
    -h, --help            show this help message and exit
    --input_path INPUT_PATH
                            path of model or npy path
    --output_path OUTPUT_PATH
                            path of dumpped ax_model
    --prefill_len PREFILL_LEN
                            token length of prefill
    --parallel PARALLEL   build parallel
    --model_config MODEL_CONFIG
                            config file
    --kv_cache_len KV_CACHE_LEN
                            length of kv_cache
    --post_topk POST_TOPK
                            post model output indices and prob
    --post_weight_type {bf16,s8}
                            post weight type
    -t {fp16,bf16,fp32}, --hidden_state_type {fp16,bf16,fp32}
                            hidden_state dtype
    -w {fp16,bf16,fp32,s8,s4}, --weight_type {fp16,bf16,fp32,s8,s4}
                            weight dtype
    -c CHECK_LEVEL, --check_level CHECK_LEVEL
                            check level 0:run 1:layer_check 2: cal 1+1
    --chip {AX620E,AX650}
                            chip


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
下载 ax-llm-build 项目
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

    git clone https://github.com/AXERA-TECH/ax-llm-build.git

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
下载 Qwen2-0.5B-Instruct
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

    cd ax-llm-build
    pip install -U huggingface_hub
    huggingface-cli download --resume-download Qwen/Qwen2-0.5B-Instruct --local-dir Qwen/Qwen2-0.5B-Instruct

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
编译执行
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

    pulsar2 llm_build --input_path Qwen/Qwen2-0.5B-Instruct/ --output_path Qwen/Qwen2-0.5B-w8a16/ --kv_cache_len 1023 --hidden_state_type bf16 --prefill_len 128 --chip AX650

^^^^^^^^^^^^^^^^^^^^^
log 参考信息
^^^^^^^^^^^^^^^^^^^^^

.. code-block::

    pulsar2 llm_build --input_path Qwen/Qwen2-0.5B-Instruct/ --output_path Qwen/Qwen2-0.5B-w8a16/ --kv_cache_len 1023 --model_config config/qwen2-0.5B.json --hidden_state_type bf16 --weight_type s8 --parallel 8
    Config(
        model_name='Qwen2-0.5B-Instruct',
        model_type='qwen2',
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,
        hidden_size=896,
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
        scale_emb=1
    )
    2024-08-22 16:16:04.364 | SUCCESS  | yamain.command.llm_build:llm_build:100 - prepare llm model done!
    building llm decode layers   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 24/24 0:05:03
    building llm post layer   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 0:01:25
    2024-08-22 16:22:33.485 | SUCCESS  | yamain.command.llm_build:llm_build:160 - build llm model done!
    2024-08-22 16:22:47.861 | SUCCESS  | yamain.command.llm_build:llm_build:337 - check llm model done!

.. note::

    该示例所运行的主机配置为:

        - Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz
        - Memory 32G

    全流程耗时大约 ``6min`` , 不同配置的主机转换时间略有差异.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
embed 提取和优化
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell  

    python tools/extract_embed.py --input_path Qwen/Qwen2-0.5B-Instruct/ --output_path Qwen/Qwen2-0.5B-w8a16/
    python tools/embed-process.py --input Qwen/Qwen2-0.5B-w8a16/model.embed_tokens.weight.npy --output Qwen/Qwen2-0.5B-w8a16/model.embed_tokens.weight.float32.bin
    chmod +x ./tools/fp32_to_bf16
    ./tools/fp32_to_bf16 Qwen/Qwen2-0.5B-w8a16/model.embed_tokens.weight.float32.bin Qwen/Qwen2-0.5B-w8a16/model.embed_tokens.weight.bfloat16.bin

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
输出文件说明
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell  

    root@xxx:/data/ax-llm-build# tree Qwen/Qwen2-0.5B-w8a16
    Qwen/Qwen2-0.5B-w8a16
    ├── model.embed_tokens.weight.bfloat16.bin
    ├── model.embed_tokens.weight.float32.bin
    ├── model.embed_tokens.weight.npy
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


其中 ``model.embed_tokens.weight.bfloat16.bin``, ``qwen_p128_l0.axmodel ~ qwen_p128_l23.axmodel``, ``qwen_post.axmodel`` 文件是上板运行所需要

~~~~~~~~~~~~~~~~~~~~~~~
开发板运行
~~~~~~~~~~~~~~~~~~~~~~~

本章节介绍如何在 ``AX650`` 开发板上运行 LLM 模型. 

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
使用 ax-llm 运行大模型
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

运行该实例相关文件已上传网盘，请自行下载和参考
  
  - `百度网盘(AX650N) <https://pan.baidu.com/s/1_LG-sPKnLS_LTWF3Cmcr7A?pwd=ph0e>`_
  - `百度网盘(AX630C) <https://pan.baidu.com/s/1X0aJTQM0bl8wsraspHnDUw?pwd=ifg5>`_

.. code-block:: shell

    root@ax650:/mnt/qtang/llama_axera_cpp# ./run_qwen2_0.5B.sh
    [I][                            Init][ 128]: LLM init start
    3% | ██                                |   1 /  27 [0.27s<7.29s, 3.70 count/s] tokenizer init ok
    [I][                            Init][  26]: LLaMaEmbedSelector use mmap
    100% | ████████████████████████████████ |  27 /  27 [6.88s<6.88s, 3.92 count/s] init post axmodel ok,remain_cmm(11317 MB)
    [I][                            Init][ 244]: max_token_len : 1023
    [I][                            Init][ 249]: kv_cache_size : 128, kv_cache_num: 1023
    [I][                            Init][ 257]: prefill_token_num : 128
    [I][                            Init][ 266]: LLM init ok
    Type "q" to exit, Ctrl+c to stop current running
    >> who are you?
    [I][                             Run][ 464]: ttft: 129.16 ms
    I am a large language model created by Alibaba Cloud. I am called Qwen.
    
    [N][                             Run][ 603]: hit eos,avg 27.22 token/s

~~~~~~~~~~~~~~~~~~~~~~~
其他示例
~~~~~~~~~~~~~~~~~~~~~~~

请参考我们在 github 上的开源项目：

- `AX-LLM <https://github.com/AXERA-TECH/ax-llm>`_
