======================
开发环境准备
======================

本节介绍使用 ``Pulsar2`` 工具链前的开发环境准备工作.

``Pulsar2`` 使用 ``Docker`` 容器进行工具链集成, 用户可以通过 ``Docker`` 加载 ``Pulsar2`` 镜像文件, 然后进行模型转换、编译、仿真等工作, 因此开发环境准备阶段只需要正确安装 ``Docker`` 环境即可. 支持的系统 ``MacOS``, ``Linux``, ``Windows``.

.. _dev_env_prepare:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
安装 Docker 开发环境
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `MacOS 安装 Docker 环境 <https://docs.docker.com/desktop/mac/install/>`_

- `Linux 安装 Docker 环境 <https://docs.docker.com/engine/install/##server>`_

- `Windows 安装 Docker 环境 <https://docs.docker.com/desktop/windows/install/>`_

``Docker`` 安装成功后, 输入 ``sudo docker -v``

.. code-block:: shell

    $ sudo docker -v
    Docker version 20.10.7, build f0df350

显示以上内容, 说明 ``Docker`` 已经安装成功. 下面将介绍 ``Pulsar2`` 工具链 ``Image`` 的安装和启动.

~~~~~~~~~~~~~~~~~~~~~~~
安装 Pulsar2 工具链
~~~~~~~~~~~~~~~~~~~~~~~

以系统版本为 ``Ubuntu 20.04``、工具链 ``ax_pulsar2_${version}.tar.gz`` 为例说明 ``Pulsar2`` 工具链的安装方法.

.. hint::

    实际操作时，请务必将 ${version} 替换为对应的工具链版本号。

工具链获取途径：

- `百度网盘 <https://pan.baidu.com/s/1FazlPdW79wQWVY-Qn--qVQ?pwd=sbru>`_
- `Google Drive <https://drive.google.com/drive/folders/10rfQIAm5ktjJ1bRMsHbUanbAplIn3ium?usp=sharing>`_

^^^^^^^^^^^^^^^^^^^^^^^
载入 Docker Image
^^^^^^^^^^^^^^^^^^^^^^^

执行 ``sudo docker load -i ax_pulsar2_${version}.tar.gz`` 导入 docker 镜像文件. 正确导入镜像文件会打印以下日志:

.. code-block:: shell

    $ sudo docker load -i ax_pulsar2_${version}.tar.gz
    Loaded image: pulsar2:${version}

完成后, 执行 ``sudo docker image ls``

.. code-block:: shell

    $ sudo docker image ls
    REPOSITORY   TAG          IMAGE ID       CREATED         SIZE
    pulsar2      ${version}   xxxxxxxxxxxx   9 seconds ago   3.27GB

可以看到工具链镜像已经成功载入, 之后便可以基于此镜像启动容器.

^^^^^^^^^^^^^^^^^^^^^^^
启动工具链镜像
^^^^^^^^^^^^^^^^^^^^^^^

执行以下命令启动 ``Docker`` 容器, 运行成功后进入 ``bash`` 环境

.. code-block:: shell

    $ sudo docker run -it --net host --rm -v $PWD:/data pulsar2:${version}

----------------------
版本查询
----------------------

``pulsar2 version`` 用于获取工具的版本信息.

示例结果

.. code-block:: bash

    root@xxx:/data# pulsar2 version
    version: ${version}
    commit: xxxxxxxx

.. _prepare_data:

----------------------
数据准备
----------------------

.. hint::

    后续内容 **模型编译**、 **仿真运行** 所需要的 **原始模型** 、 **数据** 、 **图片** 、 **仿真工具** 已在 ``quick_start_example`` 文件夹中提供 :download:`点击下载示例文件 <https://github.com/xiguadong/assets/releases/download/v0.1/quick_start_example.zip>` 然后将下载的文件解压后拷贝到 ``docker`` 的 ``/data`` 路径下.

.. code-block:: shell

    root@xxx:~/data# ls
    config  dataset  model  output  pulsar2-run-helper

* ``model``: 存放原始的 ``ONNX`` 模型 ``mobilenetv2-sim.onnx`` (预先已使用 ``onnxsim`` 将 ``mobilenetv2.onnx`` 进行计算图优化)
* ``dataset``: 存放离线量化校准 (PTQ Calibration) 需求的数据集压缩包 (支持 tar、tar.gz、gz 等常见压缩格式)
* ``config``: 存放运行依赖的配置文件 ``config.json``
* ``output``: 存放结果输出
* ``pulsar2-run-helper``: 支持 ``axmodel`` 在 X86 环境进行仿真运行的工具 

数据准备工作完毕后, 目录树结构如下:

.. code-block:: shell

    root@xxx:/data# tree -L 2
    .
    ├── config
    │   ├── mobilenet_v2_build_config.json
    │   └── yolov5s_config.json
    ├── dataset
    │   ├── coco_4.tar
    │   └── imagenet-32-images.tar
    ├── model
    │   ├── mobilenetv2-sim.onnx
    │   └── yolov5s.onnx
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
