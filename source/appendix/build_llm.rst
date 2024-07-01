======================
大模型编译(实验阶段)
======================

**本章节适用于平台**

- AX650N

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
    usage: pulsar2 llm_build [-h] [--input_path INPUT_PATH] [--output_path OUTPUT_PATH] [--input_type INPUT_TYPE] [--prefill_len PREFILL_LEN] [--parallel PARALLEL] [--model_config MODEL_CONFIG]
                            [--kv_cache_len KV_CACHE_LEN] [-t {fp16,bf16,fp32}] [-w {fp16,bf16,fp32,s8,s4}] [-c CHECK_LEVEL]

    optional arguments:
      -h, --help            show this help message and exit
      --input_path INPUT_PATH
                            path of model or npy path
      --output_path OUTPUT_PATH
                            path of dumpped ax_model
      --input_type INPUT_TYPE
                            0=npy 1=safetensors ...
      --prefill_len PREFILL_LEN
                            token length of prefill
      --parallel PARALLEL   build parallel
      --model_config MODEL_CONFIG
                            config file
      --kv_cache_len KV_CACHE_LEN
                            length of kv_cache
      -t {fp16,bf16,fp32}, --hidden_state_type {fp16,bf16,fp32}
                            hidden_state dtype
      -w {fp16,bf16,fp32,s8,s4}, --weight_type {fp16,bf16,fp32,s8,s4}
                            weight dtype
      -c CHECK_LEVEL, --check_level CHECK_LEVEL
                            check level 0:run 1:layer


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

    pulsar2 llm_build --input_path Qwen/Qwen2-0.5B-Instruct/ --output_path Qwen/Qwen2-0.5B-w8a16/ --kv_cache_len 1023 --model_config config/qwen2-0.5B.json --hidden_state_type bf16 --weight_type s8

^^^^^^^^^^^^^^^^^^^^^
log 参考信息
^^^^^^^^^^^^^^^^^^^^^

.. code-block::

    root@gpux2:/data/ax-llm-build# pulsar2 llm_build --input_path Qwen/Qwen2-0.5B-Instruct/ --output_path Qwen/Qwen2-0.5B-w8a16/ --kv_cache_len 1023 --model_config config/qwen2-0.5B.json --hidden_state_type bf16 --weight_type s8
    Config(
        model_name='Qwen/Qwen2-0.5B-Instruct',
        model_type='qwen',
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,
        hidden_size=896,
        intermediate_size=4864,
        vocab_size=151936,
        rope_theta_base=1000000.0,
        max_position_embedings=32768,
        rope_partial_factor=1.0,
        norm_eps=1e-06,
        norm_type='rms_norm',
        hidden_act='silu'
    )
    2024-07-01 11:17:08.009 | SUCCESS  | yamain.command.llm_build:llm_build:85 - prepare llm model done!
    building llm decode layers   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 24/24 0:03:59
    building llm post layer   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 0:01:24
    2024-07-01 11:22:31.941 | SUCCESS  | yamain.command.llm_build:llm_build:128 - build llm model done!
    2024-07-01 11:22:56.925 | SUCCESS  | yamain.command.llm_build:llm_build:277 - check llm model done!


.. note::

    该示例所运行的主机配置为:

        - Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz
        - Memory 32G

    全流程耗时大约 ``5 minutes`` , 不同配置的主机转换时间略有差异.


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
    ├── qwen_l0.axmodel
    ├── qwen_l10.axmodel
    ├── qwen_l11.axmodel
    ├── qwen_l12.axmodel
    ├── qwen_l13.axmodel
    ├── qwen_l14.axmodel
    ├── qwen_l15.axmodel
    ├── qwen_l16.axmodel
    ├── qwen_l17.axmodel
    ├── qwen_l18.axmodel
    ├── qwen_l19.axmodel
    ├── qwen_l1.axmodel
    ├── qwen_l20.axmodel
    ├── qwen_l21.axmodel
    ├── qwen_l22.axmodel
    ├── qwen_l23.axmodel
    ├── qwen_l2.axmodel
    ├── qwen_l3.axmodel
    ├── qwen_l4.axmodel
    ├── qwen_l5.axmodel
    ├── qwen_l6.axmodel
    ├── qwen_l7.axmodel
    ├── qwen_l8.axmodel
    ├── qwen_l9.axmodel
    └── qwen_post.axmodel


其中 ``model.embed_tokens.weight.bfloat16.bin``, ``qwen_l0.axmodel ~ qwen_l23.axmodel``, ``qwen_post.axmodel`` 文件是上板运行所需要

~~~~~~~~~~~~~~~~~~~~~~~
开发板运行
~~~~~~~~~~~~~~~~~~~~~~~

本章节介绍如何在 ``AX650`` 开发板上运行 LLM 模型. 

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
使用 ax-llm 运行大模型
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

运行该实例相关文件已上传网盘，请自行下载和参考
  
  - `百度网盘 <https://pan.baidu.com/s/1_LG-sPKnLS_LTWF3Cmcr7A?pwd=ph0e>`_

.. code-block:: shell

    root@ax650:/mnt/qtang/llama_axera_cpp# ./run_qwen2_0.5B.sh
    [I][                            Init][  71]: LLM init start
      3% | ██                                |   1 /  27 [0.28s<7.48s, 3.61 count/s] tokenizer init ok[I][                            Init][  26]: LLaMaEmbedSelector use mmap
    100% | ████████████████████████████████ |  27 /  27 [7.40s<7.40s, 3.65 count/s] init post axmodel okremain_cmm(11583 MB)
    [I][                            Init][ 180]: max_token_len : 1023
    [I][                            Init][ 185]: kv_cache_size : 128, kv_cache_num: 1023
    [I][                            Init][ 199]: LLM init ok
    Type "q" to exit, Ctrl+c to stop current running
    >> who are you?
    I am a large language model created by Alibaba Cloud. I am called Qwen.
    [N][                             Run][ 388]: hit eos,avg 24.51 token/s

~~~~~~~~~~~~~~~~~~~~~~~
其他示例
~~~~~~~~~~~~~~~~~~~~~~~

请参考我们在 github 上的开源项目：

- `AX-LLM <https://github.com/AXERA-TECH/ax-llm>`_
