# 开发环境准备

(dev_env_prepare)=

本节介绍使用 `Pulsar2` 工具链前的环境准备工作：发行包中有哪些文件、三种安装方式如何选择，以及每种方式的具体步骤。

:::{hint}
本文以 **Pulsar2 8.0** 为例。其他版本的操作完全相同，只需将文中的 `8.0` 替换为对应版本号。
:::

## 环境要求

| 项目 | 要求 | 说明 |
| --- | --- | --- |
| 操作系统 | Linux x86_64 / Linux aarch64 | `wheel` 与 `package` 方式仅支持 Linux；`Docker` 方式在 macOS / Windows 上也可用 |
| Python | 3.12 | `axnn` wheel 仅提供 `cp312`，要求 `>=3.12,<3.13` |
| Docker | 20.10 及以上 | 仅 Docker 方式需要 |
| 磁盘空间 | Docker 方式 ≥ 10 GB；wheel 方式 ≥ 5 GB | 镜像导入后约 3.3 GB，wheel 环境约 2.4 GB |
| 内存 | ≥ 8 GB | 大模型编译建议 ≥ 32 GB |

## 获取发行包

`Pulsar2` 通过以下渠道发布，进入对应版本目录后按需下载：

- [Hugging Face](https://huggingface.co/AXERA-TECH/Pulsar2)
- [ModelScope](https://modelscope.cn/models/AXERA-TECH/Pulsar2)

:::{note}
- 优先使用最新正式版，例如 `8.0`。
- 名称带 `temp` 的目录是热修复版本，仅用于验证特定问题，不建议作为生产环境基线。
:::

## 发行包目录树

以 `8.0` 为例，发行目录内容如下（大小以实际下载为准）：

```text
pulsar2_8.0/
├── README.md                                          # 发布说明
├── ax_pulsar2_8.0.tar.gz                              # 完整版 Docker 镜像（3.85 GB）
├── ax_pulsar2_8.0_lite.tar.gz                         # 精简版 Docker 镜像（1.11 GB）
├── ax_pulsar2_8.0_package.tar.gz                      # 完整版 package 便携包（3.49 GB）
├── ax_pulsar2_8.0_lite_package.tar.gz                 # 精简版 package 便携包（0.83 GB）
├── ax_pulsar2_8.0_diff_pulsar2_8.0.tar.gz             # 增量镜像包（升级用）
├── ax_pulsar2_8.0_lite_diff_pulsar2_8.0_lite.tar.gz   # 精简版增量镜像包
└── ax_pulsar2_8.0_doc.tar.gz                          # 离线文档（75 MB）
```

部分 release（例如 hotfix 版本）还会额外提供以下 wheel，可直接安装到 Python 环境：

```text
├── axnn-8.0-cp312-cp312-linux_x86_64.whl              # x86_64 编译器工具链 wheel（28 MB）
├── axnn-8.0-cp312-cp312-linux_aarch64.whl             # aarch64 编译器工具链 wheel（24 MB）
├── torch_axera-0.1.0-cp312-cp312-linux_x86_64.whl    # PyTorch NPU 后端（x86_64，1.4 MB）
├── torch_axera-0.1.0-cp312-cp312-linux_aarch64.whl    # PyTorch NPU 后端（aarch64，1.4 MB）
├── sglang_axera-0.5.18-py3-none-any.whl               # SGLang 适配包（14 MB）
└── sgl_kernel_axera-0.1.0-py3-none-any.whl            # SGLang 自定义 kernel（264 KB）
```

## 文件说明

| 文件 | 作用 | 使用方法 |
| --- | --- | --- |
| `ax_pulsar2_8.0.tar.gz` | 完整版 Docker 镜像 | `docker load -i`，见《Docker 镜像安装》 |
| `ax_pulsar2_8.0_lite.tar.gz` | 精简版 Docker 镜像，体积更小 | `docker load -i` |
| `ax_pulsar2_8.0_package.tar.gz` | 便携包，内置 Python 3.12 与全部依赖 | 解压后执行 `./bin/pulsar2`，见《package 便携包安装》 |
| `ax_pulsar2_8.0_lite_package.tar.gz` | 精简版便携包，体积更小 | 同上 |
| `ax_pulsar2_8.0_diff_pulsar2_8.0.tar.gz` | 镜像增量更新包，前提是本地已有 `8.0` 镜像 | `docker load -i`，无需重新下载完整镜像 |
| `ax_pulsar2_8.0_doc.tar.gz` | 离线文档 | 解压后用浏览器打开 `public/index.html` |
| `axnn-*-whl` | 编译器工具链，提供 `pulsar2` 命令、`axnn.dsl` 与 PYRUN 仿真 | `pip install`，见《Python wheel 安装》 |
| `torch_axera-*-whl` | PyTorch 的 NPU 后端 | 大模型推理部署时按需安装 |
| `sglang_axera-*-whl` / `sgl_kernel_axera-*-whl` | SGLang 推理框架的 AX 适配 | 大模型服务化部署时按需安装 |

三种交付形态的区别：

- **Docker 镜像**：开箱即用、环境隔离，适合快速验证与生产编译。
- **Python wheel**：直接装进 conda / venv，可 `import axnn`，适合开发调试、编写 AXDSL kernel。
- **package 便携包**：解压即用、不需要联网安装依赖，适合没有 Docker 权限且网络受限的机器。

:::{hint}
如需使用 `AXDSL` 编写自定义 NPU kernel、进行 PYRUN 仿真，请参考
[AXDSL 社区教程](https://axdsl-community-tutorial.readthedocs.io/zh-cn/latest/)。
:::

## Docker 镜像安装

### 安装 Docker

不同操作系统的最低配置要求与安装流程请参考：

- [MacOS 安装 Docker 环境](https://docs.docker.com/desktop/setup/install/mac-install/)
- [Linux 安装 Docker 环境](https://docs.docker.com/engine/install/)
- [Windows 安装 Docker 环境](https://docs.docker.com/desktop/setup/install/windows-install/)

安装完成后确认版本：

```shell
$ sudo docker -v
Docker version 20.10.7, build f0df350
```

### 载入镜像

```shell
$ sudo docker load -i ax_pulsar2_8.0.tar.gz
Loaded image: pulsar2:8.0
```

精简版镜像使用 `ax_pulsar2_8.0_lite.tar.gz`，镜像 tag 为 `pulsar2:8.0-lite`。
如果本地已经加载过 `8.0` 镜像，只需加载增量包即可完成升级：

```shell
$ sudo docker load -i ax_pulsar2_8.0_diff_pulsar2_8.0.tar.gz
```

确认镜像已成功导入：

```shell
$ sudo docker image ls
REPOSITORY   TAG   IMAGE ID       CREATED         SIZE
pulsar2      8.0   xxxxxxxxxxxx   9 seconds ago   3.27GB
```

### 启动容器

```shell
$ sudo docker run -it --net host --rm -v $PWD:/data pulsar2:8.0
```

- `-v $PWD:/data`：把当前目录挂载为容器内的 `/data`，模型与数据集放在这里。
- `--net host`：容器直接使用宿主机网络。
- `--rm`：退出后自动删除容器，避免残留。
镜像内已配置好 `pulsar2` 的运行环境，无需额外安装依赖。

### 验证安装

```shell
root@xxx:/data# pulsar2 version
version: 8.0
commit: xxxxxxxx
```

## Python wheel 安装

wheel 方式把工具链装进独立的 Python 环境，适合开发调试与二次集成，要求 **Python 3.12**。

:::{hint}
`axnn` 只有按架构区分的 wheel（`x86_64` / `aarch64`），按下面的命令安装即可。
:::

### 创建 Python 环境

使用 conda：

```shell
$ conda create -y -n axnn python=3.12
$ conda activate axnn
```

或使用 venv：

```shell
$ python3.12 -m venv ~/axnn-env
$ source ~/axnn-env/bin/activate
```

### 安装 axnn

```shell
$ cd /path/to/release
$ pip install --upgrade --upgrade-strategy eager \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "./axnn-8.0-cp312-cp312-linux_x86_64.whl[cpu]"
```

:::{hint}
`[cpu]` 指定安装 CPU 版 `torch` 依赖。安装过程需要联网下载依赖；使用内网镜像时请同时配置 `pip` 的 `index-url`。
:::

### 验证安装

```shell
$ pulsar2 version
version: 8.0
commit: xxxxxxxx

$ which pulsar2
/home/<user>/miniforge3/envs/axnn/bin/pulsar2
```

### 升级与卸载

```shell
$ pip install --upgrade "./axnn-<new_version>-...whl[cpu]"
$ pip uninstall axnn
```

## package 便携包安装

`package` 是免安装的完整运行环境，解压后即可使用，不需要 Docker，也不需要预装 Python。支持 Linux x86_64 / aarch64。

```shell
$ tar -zxvf ax_pulsar2_8.0_package.tar.gz
$ cd ax_pulsar2_8.0_package
$ ./bin/pulsar2 version
```

目录说明：

```text
ax_pulsar2_8.0_package/
├── bin         # 可执行文件，入口为 ./bin/pulsar2
├── install     # HASP 授权文件
├── lib         # 运行所需的动态库，可用 PULSAR2_LIB_PATH 指定其他路径
├── python3     # 内置 Python 3.12 环境，可用 PULSAR2_PYTHON_PATH 指定其他路径
├── pulsar2     # 工具链本体
└── README.md
```

:::{hint}
`lib` 与 `python3` 中的基础依赖在不同版本间变化不大，可以放到公共目录，通过环境变量 `PULSAR2_LIB_PATH`、`PULSAR2_PYTHON_PATH` 复用；升级时只需替换 `pulsar2` 目录。
:::

## 安装方式对比

| 对比项 | Docker 镜像 | Python wheel | package 便携包 |
| --- | --- | --- | --- |
| 安装难度 | 低 | 中 | 低 |
| 权限要求 | 需要 `sudo` / docker 组 | 普通用户 | 普通用户 |
| 体积 | 3.85 GB（lite 1.11 GB） | 约 2.4 GB（含依赖） | 3.49 GB |
| 需要联网 | 否 | 是（下载依赖） | 否 |
| 可调试性 | 一般 | 好，可 `import axnn` | 一般 |
| 适用场景 | 快速验证、生产编译 | 开发调试、AXDSL | 无 Docker 权限、内网机器 |

## 数据准备

(prepare_data)=

:::{hint}
后续 **模型编译**、**仿真运行** 所需的原始模型、数据、图片与仿真工具已在 `quick_start_example` 中提供，{download}`点击下载示例文件 <../examples/quick_start_example.zip>`，解压后拷贝到工作目录：Docker 方式为容器的 `/data`，wheel / package 方式为本地目录。
:::

```shell
root@xxx:~/data# ls
config  dataset  model  output  pulsar2-run-helper
```

- `model`：原始 `ONNX` 模型 `mobilenetv2-sim.onnx`（已使用 `onnxslim` 做过计算图优化）
- `dataset`：离线量化校准（PTQ Calibration）所需的数据集压缩包（支持 `tar`、`tar.gz`、`gz` 等常见格式）
- `config`：运行所需的配置文件 `config.json`
- `output`：结果输出目录
- `pulsar2-run-helper`：支持 `axmodel` 在 x86 环境进行仿真运行的工具

数据准备工作完毕后，目录树结构如下：

```shell
root@xxx:/data# tree -L 2
.
├── config
│   ├── mobilenet_v2_build_config.json
│   └── yolov5s_config.json
├── dataset
│   ├── coco_4.tar
│   └── imagenet-32-images.tar
├── model
│   ├── mobilenetv2-sim.onnx
│   └── yolov5s.onnx
├── output
└── pulsar2-run-helper
    ├── cli_classification.py
    ├── cli_detection.py
    ├── models
    ├── pulsar2_run_helper
    ├── requirements.txt
    ├── setup.cfg
    ├── sim_images
    ├── sim_inputs
    └── sim_outputs
```

## 常见问题

| 现象 | 原因 | 解决 |
| --- | --- | --- |
| `docker: command not found` | 未安装 Docker | 参考上文安装 Docker |
| `permission denied ... docker.sock` | 当前用户不在 `docker` 组 | 使用 `sudo`，或将用户加入 `docker` 组后重新登录 |
| `Unsupported wheel on this platform` | Python 版本或架构不匹配 | 使用 Python 3.12，并选择 `linux_x86_64` / `linux_aarch64` 对应的 wheel |
| `pulsar2: command not found` | 未激活对应环境 | `conda activate axnn`，或使用绝对路径 `./bin/pulsar2` |
| 安装 wheel 时卡在 `torch` 下载 | 默认源较慢 | 保留 `--extra-index-url`，并配置 `pip` 国内镜像 |
| 镜像导入后磁盘占用较大 | 镜像解压后约 3.3 GB | 改用 `lite` 版本，或通过增量包升级 |

完成环境准备后，请继续阅读对应芯片平台的 Quick Start，例如 {doc}`Quick Start(AX650) <quick_start_ax650>`。
