======================
大模型编译(实验阶段)
======================

**本章节适用于平台**

- AX650N
- AX630C

**已验证模型**

- DeepSeek-R1-Distill
- Qwen2.5
- MiniCPM、MiniCPM-V 2.0
- InternVL2
- ChatGLM3
- OpenBuddy
- SmolLM
- Llama3.2
- Gemma2
- Phi2、Phi3
- TinyLlama

本章节介绍如何将 Huggingface 上的模型转换的基本操作, 使用 ``pulsar2`` 工具将从 Huggingface 下载的项目中 ``*.safetensor`` 或 ``pytorch_model.bin``  模型编译成 ``axmodel`` 模型. 请先参考 :ref:`《开发环境准备》 <dev_env_prepare>` 章节完成开发环境搭建. 
本节示例模型为 ``Qwen2-0.5B-Instruct``.

**版本约束**

本文档基于 Pulsar2 3.2 版本进行编写。

**LLM ModelZoo**

- `AX650N <https://pan.baidu.com/s/1_LG-sPKnLS_LTWF3Cmcr7A?pwd=ph0e>`_
- `AX630C <https://pan.baidu.com/s/1X0aJTQM0bl8wsraspHnDUw?pwd=ifg5>`_

**关联项目 AX-LLM**

该项目用于探索业界常用 LLM(Large Language Model) 在已有芯片平台上落地的可行性和相关能力边界，方便社区开发者进行快速评估和二次开发自己的 LLM 应用。

- `AX-LLM <https://github.com/AXERA-TECH/ax-llm>`_

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
命令说明
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Pulsar2`` 工具链中使用 ``pulsar2 llm_build`` 命令来完成 LLM 模型的转换. 

.. code-block:: shell

    root@xxx:/data# pulsar2 llm_build --help
    usage: pulsar2 llm_build [-h] [--input_path INPUT_PATH] [--output_path OUTPUT_PATH] [--prefill_len PREFILL_LEN]
                            [--parallel PARALLEL] [--model_config MODEL_CONFIG] [--kv_cache_len KV_CACHE_LEN]
                            [--post_topk POST_TOPK] [--post_weight_type {bf16,s8}] [-t {fp16,bf16,fp32}]
                            [-w {fp16,bf16,fp32,s8,s4}] [-c CHECK_LEVEL] [--chip {AX620E,AX650}] [--prompt PROMPT]

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
    --prompt PROMPT       prompt for check_level==2


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

    chmod +x ./tools/fp32_to_bf16
    chmod +x ./tools/embed_process.sh
    ./tools/embed_process.sh Qwen/Qwen2-0.5B-Instruct/ Qwen/Qwen2-0.5B-w8a16/

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
输出文件说明
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell  

    root@xxx:/data/ax-llm-build# tree Qwen/Qwen2-0.5B-w8a16
    Qwen/Qwen2-0.5B-w8a16
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

板端运行程序编译流程，请参考我们在 github 上的开源项目 `AX-LLM <https://github.com/AXERA-TECH/ax-llm>`_


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Tokenizer 解析器说明
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ax-llm 项目中的 Tokenizer 解析器采用本地模块与 HTTP Server 两种方案，其中本地方案又尝试了 sentencepiece、tiktoken 两种方案。
但是我们在实际调试过程中发现 sentencepiece 对于不同 LLM 模型的 special tokens 支持不友好，需要用户自行处理 special tokens 的拆分，容易导致板端 token id 与 transformers 库中的 AutoTokenizer 获得的 token id 存在差异，最终影响 LLM 的输出结果正确性。
因此我们建议前期调试的时候使用 Tokenizer HTTP Server 的方式直接调用 transformers 库中的 AutoTokenizer 模块进行测试。 

Tokenizer HTTP Server 的特点：

* 保证 token id 正确
* 方便添加 chat template
* 支持本地、远端部署
* 支持多用户接入

以在网盘中已提供基于 Qwen2.5 3B 的相关文件为例

.. code-block:: shell

    root@xxx:/data/ax-llm-build# tree qwen2.5-3b-prefill-ax650/
    qwen2.5-3b-prefill-ax650/
    ├── main_prefill
    ├── qwen2.5-3B-prefill-ax650
    │   ├── model.embed_tokens.weight.bfloat16.bin
    │   ├── qwen2_p128_l0_together.axmodel
        ...
    │   ├── qwen2_p128_l12_together.axmodel
    │   └── qwen2_post.axmodel
    ├── qwen2.5_tokenizer
    │   ├── merges.txt
    │   ├── tokenizer_config.json
    │   ├── tokenizer.json
    │   └── vocab.json
    ├── qwen2.5_tokenizer.py
    ├── qwen.tiktoken
    ├── readme.txt
    └── run_qwen2.5_3B_prefill_ax650.sh

* qwen2.5_tokenizer：是 tokenizer 相关文件，从 Qwen/Qwen2.5-3B-Instruct/ 中提取
* qwen2.5_tokenizer.py：是用 python 实现的 Tokenizer HTTP Server

运行说明如下：

* python qwen2.5_tokenizer.py --host xxx.xxx.xxx.xxx --port 12345，其中 --host xxx.xxx.xxx.xxx 设置 tokenizer解析服务器的 IP 地址，确保 AX650N 能正常访问该地址。可以在具备 python 环境的 AX650N 本地运行
* 修改 run_qwen2.5_3B_prefill_ax650.sh 中 --filename_tokenizer_model 的 IP 信息和步骤1中的一致
* 运行 run_qwen2.5_3B_prefill_ax650.sh 即可

.. code-block:: shell

    root@xxx:/data/ax-llm-build# cat qwen2.5-3b-prefill-ax650/run_qwen2.5_3B_prefill_ax650.sh
    ./main_prefill \
    --template_filename_axmodel "qwen2.5-3B-prefill-ax650/qwen2_p128_l%d_together.axmodel" \
    --axmodel_num 36 \
    --tokenizer_type 2 \
    --filename_tokenizer_model http://xxx.xxx.xxx.xxx:12345 \
    --bos 0 --eos 0 \
    --filename_post_axmodel "qwen2.5-3B-prefill-ax650/qwen2_post.axmodel" \
    --filename_tokens_embed "qwen2.5-3B-prefill-ax650/model.embed_tokens.weight.bfloat16.bin" \
    --tokens_embed_num 151936 \
    --tokens_embed_size 2048 \
    --use_mmap_load_embed 1 \
    --live_print 1 \
    --continue 1 \
    --prompt "$1"

~~~~~~~~~~~~~~~~~~~~~~~
其他示例
~~~~~~~~~~~~~~~~~~~~~~~

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
MiniCPM-V 2.0
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**下载 MiniCPM-V 2.0**


.. code-block:: shell

    cd ax-llm-build
    pip install -U huggingface_hub
    huggingface-cli download --resume-download openbmb/MiniCPM-V-2 --local-dir openbmb/MiniCPM-V-2


**获取 axmodel**

.. code-block:: shell

    pulsar2 llm_build --input_path openbmb/MiniCPM-V-2/ --output_path openbmb/MiniCPM-V-2-ax650 --kv_cache_len 1023 --hidden_state_type bf16 --prefill_len 128 --chip AX650

log 参考信息

.. code-block::

    pulsar2 llm_build --input_path openbmb/MiniCPM-V-2/ --output_path openbmb/MiniCPM-V-2-ax650 --kv_cache_len 1023 --hidden_state_type bf16 --prefill_len 128 --chip AX650 --parallel 8
    Config(
        model_name='openbmb/MiniCPM-V-2',
        model_type='minicpmv',
        num_hidden_layers=40,
        num_attention_heads=36,
        num_key_value_heads=36,
        hidden_size=2304,
        intermediate_size=5760,
        vocab_size=122753,
        rope_theta=10000.0,
        max_position_embeddings=4096,
        rope_partial_factor=1.0,
        rms_norm_eps=1e-05,
        norm_type='rms_norm',
        hidden_act='silu',
        hidden_act_param=0.03,
        scale_depth=1.4,
        scale_emb=12,
        dim_model_base=256
    )
    2024-10-07 15:18:38.605 | SUCCESS  | yamain.command.llm_build:llm_build:101 - prepare llm model done!
    tiling op...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3287/3287 0:00:44
    build op serially...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7610/7610 0:04:09
    build op...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 11485/11485 0:00:00
    add ddr swap...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 253160/253160 0:00:42
    calc input dependencies...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 289230/289230 0:00:31
    calc output dependencies...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 289230/289230 0:00:42
    assign eu heuristic   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 289230/289230 0:00:51
    assign eu onepass   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 289230/289230 0:00:10
    assign eu greedy   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 289230/289230 0:00:12
    building vision model   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 0:14:51
    building llm decode layers   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 40/40 0:04:24
    building llm post layer   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 0:02:19
    2024-10-07 15:40:14.676 | SUCCESS  | yamain.command.llm_build:llm_build:170 - build llm model done!
    2024-10-07 15:40:48.246 | SUCCESS  | yamain.command.llm_build:llm_build:349 - check llm model done!


**获取 embed 文件**

.. code-block:: shell

    chmod +x ./tools/fp32_to_bf16
    chmod +x ./tools/embed_process.sh
    ./tools/embed_process_vl.sh Qwen/Qwen2-0.5B-Instruct/ Qwen/Qwen2-0.5B-w8a16/

最终生成文件如下

.. code-block:: shell

    root@xxx: tree openbmb/MiniCPM-V-2-ax650/
    openbmb/MiniCPM-V-2-ax650/
    ├── minicpmv_p128_l0_together.axmodel
    ├── minicpmv_p128_l10_together.axmodel
    ...
    ├── minicpmv_p128_l19_together.axmodel
    ├── minicpmv_p128_l1_together.axmodel
    ├── minicpmv_p128_l20_together.axmodel
    ...
    ├── minicpmv_p128_l29_together.axmodel
    ├── minicpmv_p128_l2_together.axmodel
    ├── minicpmv_p128_l30_together.axmodel
    ...
    ├── minicpmv_p128_l39_together.axmodel
    ├── minicpmv_p128_l3_together.axmodel
    ...
    ├── minicpmv_p128_l8_together.axmodel
    ├── minicpmv_p128_l9_together.axmodel
    ├── minicpmv_post.axmodel
    ├── model.embed_tokens.weight.bfloat16.bin
    └── vpm_resampler.axmodel


**上板运行**

MiniCPM-V 的上板部署项目需要使用 ax-llm 的 minicpmv 的分支

- `ax-llm/tree/minicpm-v <https://github.com/AXERA-TECH/ax-llm/tree/minicpm-v>`_

.. figure:: ../media/ssd_dog.jpg
    :alt: pipeline
    :align: center

.. code-block:: shell

    root@ax650:/llm-test/minicpm-v-2.0# ./run_minicpmv-2.sh
    [I][                            Init][ 125]: LLM init start
    2% | █                                 |   1 /  44 [0.21s<9.11s, 4.83 count/s] tokenizer init ok
    [I][                            Init][  26]: LLaMaEmbedSelector use mmap
    100% | ████████████████████████████████ |  44 /  44 [33.54s<33.54s, 1.31 count/s] init vpm axmodel ok,remain_cmm(8086 MB)
    [I][                            Init][ 284]: max_token_len : 1023
    [I][                            Init][ 289]: kv_cache_size : 2304, kv_cache_num: 1023
    [I][                            Init][ 297]: prefill_token_num : 128
    [I][                            Init][ 306]: LLM init ok
    Type "q" to exit, Ctrl+c to stop current running
    prompt >> 描述下图片
    image >> ssd_dog.jpg
    [I][                          Encode][ 365]: image encode time : 728.507019 ms
    [I][                             Run][ 589]: ttft: 520.94 ms
    这幅图片展示了一只大而毛茸茸的狗，可能是拉布拉多或类似品种，坐在黄色和红色相间的门廊上。这只狗看起来在休息，它的目光朝向相机，表情平静。在狗的后面，有一辆红色自行车，车架上有黑色的装饰，停放在门廊上。自行车上挂着几个行李袋，表明它可能用于旅行或运输。背景中，可以看到一辆白色车辆，可能是汽车，停在门廊的后面。整个场景暗示了一个家庭环境，可能是在住宅区。

    [N][                             Run][ 728]: hit eos,avg 5.55 token/s

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
调试说明
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``pulsar2 llm_build`` 通过在编译命令中使用 ``--check_level`` 启动调试精度调试功能

* ``--check_level 1``：测试第一层的相似度
* ``--check_level 2``：指定 prompt 输入的内容，用于仿真运行编译生成的模型文件。

^^^^^^^^^^^^^^^^^^^^^
--check_level 1
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

    pulsar2 llm_build --check_level 1 --input_path Qwen/Qwen2-0.5B-Instruct/ --output_path Qwen/Qwen2-0.5B-w8a16/ --kv_cache_len 1023 --hidden_state_type bf16 --prefill_len 128 --chip AX650 

LOG：

.. code-block:: shell

    pulsar2 llm_build --check_level 1 --input_path Qwen/Qwen2-0.5B-Instruct/ --output_path Qwen/Qwen2-0.5B-w8a16/ --kv_cache_len 1023 --hidden_state_type bf16 --prefill_len 128 --chip AX650 --parallel 8
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
        scale_emb=1,
        dim_model_base=256
    )
    2024-10-07 01:23:28.414 | SUCCESS  | yamain.command.llm_build:llm_build:101 - prepare llm model done!
    building llm decode layers   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 24/24 0:00:39
    building llm post layer   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 0:01:26
    2024-10-07 01:25:34.765 | SUCCESS  | yamain.command.llm_build:llm_build:170 - build llm model done!
    2024-10-07 01:25:38.740 | INFO     | yamain.command.llm_build:llm_build:294 - decode layer0_gt layer0_got cos_sim is: 0.9986067835921196
    2024-10-07 01:25:45.421 | INFO     | yamain.command.llm_build:llm_build:325 - prefill layer0_gt layer0_got cos_sim is: 0.9986067835921196
    2024-10-07 01:25:45.421 | SUCCESS  | yamain.command.llm_build:llm_build:349 - check llm model done!

^^^^^^^^^^^^^^^^^^^^^
--check_level 2
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

    pulsar2 llm_build --check_level 2 --prompt "<|im_start|>user\n1+1=?<|im_end|>\n<|im_start|>assistant\n" --input_path Qwen/Qwen2-0.5B-Instruct/ --output_path Qwen/Qwen2-0.5B-w8a16/ --kv_cache_len 1023 --hidden_state_type bf16 --prefill_len 128 --chip AX650 

由于会打印每一层（hidden_layer）的调试信息，信息量有点大，这里就只显示比较关键的一些内容。

.. code-block:: shell

    pulsar2 llm_build --check_level 2 --prompt "<|im_start|>user\n1+1=?<|im_end|>\n<|im_start|>assistant\n" --input_path Qwen/Qwen2-0.5B-Instruct/ --output_path Qwen/Qwen2-0.5B-w8a16/ --kv_cache_len 1023 --hidden_state_type bf16 --prefill_len 128 --chip AX650
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
        scale_emb=1,
        dim_model_base=256
    )
    2024-10-07 01:04:57.881 | SUCCESS  | yamain.command.llm_build:llm_build:101 - prepare llm model done!
    building llm decode layers   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 24/24 0:00:39
    building llm post layer   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 0:01:26
    2024-10-07 01:07:04.398 | SUCCESS  | yamain.command.llm_build:llm_build:170 - build llm model done!
    Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
    load Qwen/Qwen2-0.5B-w8a16/qwen2_p128_l0_together
    load Qwen/Qwen2-0.5B-w8a16/qwen2_p128_l1_together
    ...
    load Qwen/Qwen2-0.5B-w8a16/qwen2_p128_l22_together
    load Qwen/Qwen2-0.5B-w8a16/qwen2_p128_l23_together
    2024-10-07 01:07:05.499 | INFO     | yasched.llm_utils:run:497 - simulate layer 0
    2024-10-07 01:07:11.902 | INFO     | yasched.llm_utils:run:503 - end simulate
    [[[-0.24707 0.0883789 -0.232422 ... -0.294922 0.0644531 -0.65625]
    [0.0649414 -0.183594 -0.251953 ... -0.248047 -0.0231934 -0.138672]
    [0.0766602 -0.0961914 0.152344 ... -0.0125732 0.106445 0.15625]
    ...
    [-0.0737305 -0.210938 -0.455078 ... -0.640625 0.0429688 -0.263672]
    [-0.0737305 -0.210938 -0.455078 ... -0.640625 0.0429688 -0.263672]
    [-0.0737305 -0.210938 -0.455078 ... -0.640625 0.0429688 -0.263672]]]
    2024-10-07 01:07:11.903 | INFO     | yasched.llm_utils:run:497 - simulate layer 1
    ...
    2024-10-07 01:09:35.992 | INFO     | yasched.llm_utils:run:497 - simulate layer 23
    2024-10-07 01:09:42.591 | INFO     | yasched.llm_utils:run:503 - end simulate
    [[[-1.25 0.222656 2.375 ... 2.07812 -0.410156 1.84375]
    [-0.289062 -1.08594 0.234375 ... 1.07812 -0.257812 -1.96094]
    [-0.0839844 -0.542969 0.636719 ... 3.21875 -0.351562 -2.01562]
    ...
    [-3.21875 -0.478516 1.42188 ... 4.8125 1.21875 -0.294922]
    [-3.21875 -0.478516 1.42188 ... 4.8125 1.21875 -0.294922]
    [-3.21875 -0.478516 1.42188 ... 4.8125 1.21875 -0.294922]]]
    2
    posibile ('\n', 0.0),('答案', 0.0),('Result', 0.0),('0', 0.0),('3', 0.0),('2', 1.0),('1', 0.0),('Answer', 0.0),('\\', 0.0),('4', 0.0)
    load Qwen/Qwen2-0.5B-w8a16/qwen2_p128_l0_together
    load Qwen/Qwen2-0.5B-w8a16/qwen2_p128_l1_together
    load Qwen/Qwen2-0.5B-w8a16/qwen2_p128_l2_together
    ...
    start_indice = 12
    2024-10-07 01:10:37.005 | INFO     | yasched.llm_utils:run:556 - simulate layer 23
    2024-10-07 01:10:38.859 | INFO     | yasched.llm_utils:run:562 - end simulate
    [-0.310547 -2.21875 0.871094 -1.86719 -0.546875]
    start_indice = 12
    <|im_end|>
    posibile ('\n', 0.0),('\\t', 0.0),('<|im_start|>', 0.0),(' \\', 0.0),('.', 0.0),('\n\n', 0.0),(' ', 0.0),('\\', 0.0),('<|im_end|>', 1.0),('\\n', 0.0)
    ====================================================================================================
    <|im_start|>user\n1+1=?<|im_end|>\n<|im_start|>assistant\n2<|im_end|>
    ====================================================================================================
    hit eos!
    2024-10-07 01:10:51.637 | SUCCESS  | yamain.command.llm_build:llm_build:349 - check llm model done!

