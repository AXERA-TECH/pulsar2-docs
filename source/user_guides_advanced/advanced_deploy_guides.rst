.. _model_deploy_advanced:

=========================
模型部署进阶指南
=========================

--------------------
概述
--------------------

本节将提供运行 ``Pulsar2`` 编译生成的 ``axmodel`` 模型的代码示例, 所有示例代码由 ``ax-samples`` 项目提供。
``ax-samples`` 由 AXera 主导的项目，其目的是提供业界主流的开源算法模型运行示例代码，方便开发者快速对 AXera 的芯片进行评估和适配。

~~~~~~~~~~~~~~~~~~~~
获取方式
~~~~~~~~~~~~~~~~~~~~

- :download:`点击下载 <https://github.com/AXERA-TECH/pulsar2-docs/releases/download/v1.9/ax-samples.zip>`

~~~~~~~~~~~~~~~~~~~~
ax-samples 简介
~~~~~~~~~~~~~~~~~~~~

当前 ``ax-samples`` 已验证但不限于以下开源模型:

- 分类模型

    - MobileNetv1
    - MobileNetv2
    - Others

- 检测模型

    - YOLOv5s
  
已验证硬件平台

- AX650
- M76H
- AX620E

``ax-samples`` 目录说明

.. code-block:: bash

    $ tree -L 3
    .
    ├── 3rdparty      # ax-samples 编译依赖的第三方库      
    │   ├── ax620e
    │   │   ├── engine
    │   │   ├── interpreter
    │   │   └── libsys
    │   ├── ax650
    │   │   ├── ax-engine
    │   │   ├── interpreter
    │   │   └── libsys
    │   ├── libopencv-4.5.5-aarch64         # aarch64 版本的 OpenCV, 用于打开测试图片，绘制执行结果
    │   │   ├── bin
    │   │   ├── include
    │   │   ├── lib
    │   │   └── share
    │   ├── libprotobuf-3.19.4-aarch644      # axmodel 解析需要
    │   │   ├── include
    │   │   └── lib
    │   └── protoc-3.19.4-aarch64           # axmodel 解析需要
    │       └── protoc
    ├── CMakeLists.txt
    ├── cmake                           # cmake 工程创建模块
    │   ├── ax620e.cmake
    │   ├── ax650.cmake
    │   ├── check.cmake
    │   └── summary.cmake
    ├── examples                        # samples 
    │   ├── CMakeLists.txt
    │   ├── ax620e
    │   │   ├── ax_classification_steps.cc
    │   │   ├── ax_yolov5s_steps.cc
    │   │   └── middleware
    │   ├── ax650
    │   │   ├── ax_classification_steps.cc  # classification
    │   │   ├── ax_yolov5s_steps.cc         # yolov5s
    │   │   └── middleware
    │   ├── postprocess                     # 后处理模块
    │   │   ├── detection.hpp
    │   │   ├── pose.hpp
    │   │   ├── score.hpp
    │   │   ├── topk.hpp
    │   │   └── yolo.hpp
    │   ├── preprocess                      # 预处理模块
    │   │   └── common.hpp
    │   └── utilities
    │       ├── args.hpp
    │       ├── cmdline.hpp
    │       ├── file.hpp
    │       ├── split.hpp
    │       └── timer.hpp
    └── toolchains
        └── aarch64-linux-gnu.toolchain.cmake        


--------------------
编译示例
--------------------

~~~~~~~~~~~~~~~~~~~~
环境准备
~~~~~~~~~~~~~~~~~~~~

- ``cmake`` 版本大于等于 3.13
- ``AX650A`` 配套的交叉编译工具链 ``aarch64-linux-gnu-gxx`` 已添加到环境变量中, 版本信息为 ``gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu``

^^^^^^^^^^^^^^^^^^^^
安装 cmake
^^^^^^^^^^^^^^^^^^^^

``cmake`` 的安装有多种方式, 如果是 ``Anaconda`` **虚拟环境** 下, 可以通过如下命令安装:

.. code-block:: bash
  
    pip install cmake

如果 **非虚拟环境** , 且系统为 ``Ubuntu``, 可以通过

.. code-block:: bash

    sudo apt-get install cmake

.. _`cmake 官网`: https://cmake.org/download/

如果安装版本较低, 也可以通过下载 **源码编译** ``cmake``, 具体方法如下:

- step 1: `cmake 官网`_ 下载 ``cmake`` 后解压

- step 2: 进入安装文件夹, 依次执行

.. code-block:: bash
    
    ./configure
    make -j4  # 4代表电脑核数, 可以省略
    sudo make install

- step 3: ``cmake`` 安装完毕后, 通过以下命令查看版本信息

.. code-block:: bash

    cmake --version

.. _`aarch64-linux-gnu-gxx`: https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
安装交叉编译工具 aarch64-linux-gnu-gxx
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

交叉编译器有很多种, 推荐使用 ``Linaro`` 出品的交叉编译器, 可以从 `aarch64-linux-gnu-gxx`_ 中下载相关文件, 
其中 ``gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz`` 为 64bit 版本.

.. code-block:: bash

    # 新建文件夹并移动压缩包
    mkdir -p ~/usr/local/lib
    mv gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz ~/usr/local/lib
    
    # 解压
    cd ~/usr/local/lib
    xz -d gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz
    tar -xvf gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar
    
    # 配置环境变量
    vim ~/.bashrc
    export PATH=$PATH:~/usr/local/lib/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu/bin
    
    # 环境生效
    source ~/.bashrc

~~~~~~~~~~~~~~~~~~~~
交叉编译
~~~~~~~~~~~~~~~~~~~~

**解压文件**

.. code-block:: bash

    $ unzip ax-samples.zip

**源码编译**

进入 ax-samples 根目录，创建 cmake 编译任务：

**AX650 或 M76H**

.. code-block:: bash

    $ mkdir build
    $ cd build
    $ cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-none-linux-gnu.toolchain.cmake ..
    $ make install

**AX620E**

.. code-block:: bash

    $ mkdir build
    $ cd build
    $ cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-none-linux-gnu.toolchain.cmake -DAX_TARGET_CHIP=ax620e ..
    $ make install

编译完成后，生成的可执行示例存放在 ``ax-samples/build/install/bin/`` 路径下：

.. code-block:: bash

    /ax-samples/build$ tree install
    install
    └── bin
        ├── ax_classification
        └── ax_yolov5s

--------------------
运行示例
--------------------

**运行准备**

登入 ``AX650A`` 或 ``M76H`` 或 ``AX620E`` 开发板, 在 ``root`` 路径下创建 ``sample`` 文件夹. 

- 将上一章节 ``build/install/bin/`` 中编译生成的可执行示例拷贝到 ``/root/sample/`` 路径下;
- 将 **Pulsar2** 生成的 ``mobilenetv2.axmodel`` 或 ``yolov5s.axmodel`` 模型拷贝到  ``/root/sample/`` 路径下;
- 将测试图片拷贝到 ``/root/sample/`` 路径下.

.. code-block:: bash
  
    /root/sample # ls -l
    total 26628
    -rwxrw-r--    1 1000     1000       5722408 Nov 28 11:09 ax_classification
    -rwxrw-r--    1 1000     1000       5930800 Nov 28 11:09 ax_yolov5s
    -rw-rw-r--    1 1000     1000        140391 Nov  4 16:44 cat.jpg
    -rw-------    1 1000     root        163759 Oct 17 17:18 dog.jpg
    -rw-rw-r--    1 1000     1000       4632857 Nov 28 11:09 mobilenetv2.axmodel
    -rw-rw-r--    1 1000     1000       7873709 Nov 28 11:09 yolov5s.axmodel

如果提示板子空间不足, 可以通过文件夹挂载的方式解决.

**MacOS 挂载 ARM 开发板示例**

.. hint::

    由于板上空间有限, 测试时通常需要进行文件夹共享操作, 这个时候就需要将 ``ARM`` 开发板与主机之间进行共享. 这里仅以 ``MacOS`` 为例.

开发机挂载 ``ARM`` 开发板需要 ``NFS`` 服务, 而 ``MacOS`` 系统自带 ``NFS`` 服务, 只需要创建 ``/etc/exports`` 文件夹, ``nfsd`` 将自动启动并开始用于 ``exports``.

``/etc/exports`` 可以配置如下:

.. code-block:: shell

    /path/your/sharing/directory -alldirs -maproot=root:wheel -rw -network xxx.xxx.xxx.xxx -mask 255.255.255.0

参数释义

.. list-table::
    :widths: 15 40
    :header-rows: 1

    * - 参数名
      - 含义
    * - alldirs
      - 共享 ``/Users`` 目录下所有文件, 如果只想共享一个文件夹可以省略
    * - network
      - 挂载 ARM 开发板 IP 地址, 可以是网段地址
    * - mask
      - 子网掩码, 通常是 255.255.255.0
    * - maproot
      - 映射规则, 当 ``maproot=root:wheel`` 时表示把 ``ARM`` 板的 ``root`` 用户映射为开发机上的 ``root`` 用户, ``ARM`` 的 ``root`` 组 映射为 ``MacOS`` 上的 ``wheel`` (gid=0) 组. 
        如果缺省, 可能会出现 ``nfsroot`` 链接失败错误.
    * - rw
      - 读写操作, 默认开启

修改 ``/etc/exports`` 需要重启 ``nfsd`` 服务

.. code-block:: bash

    sudo nfsd restart

如果配置成功, 可以使用

.. code-block:: bash

    sudo showmount -e
 
命令查看挂载信息, 例如输出 ``/Users/skylake/board_nfs 10.168.21.xx``, 配置好开发机后需要在 ``ARM`` 端执行 ``mount`` 指令

.. code-block:: bash

    mount -t nfs -o nolock,tcp macos_ip:/your/shared/directory /mnt/directory

如果出现权限问题, 需要检查 ``maproot`` 参数是否正确.

.. hint::

    ``network`` 参数可以配置成网段的形式, 如: ``10.168.21.0``, 如果挂载单ip出现 ``Permission denied``, 可以尝试一下网段内挂载.

**分类模型**

对于分类模型, 可以通过执行 ``ax_classification`` 程序实现板上运行.

.. code-block:: bash

    /root/sample # ./ax_classification -m mobilenetv2.axmodel -i cat.jpg --repeat 100
    --------------------------------------
    model file : mobilenetv2.axmodel
    image file : cat.jpg
    img_h, img_w : 224 224
    --------------------------------------
    Engine creating handle is done.
    Engine creating context is done.
    Engine get io info is done.
    Engine alloc io is done.
    Engine push input is done.
    --------------------------------------
    topk cost time:0.10 ms
    9.7735, 285
    9.2452, 283
    8.9811, 281
    8.7169, 282
    7.5283, 463
    --------------------------------------
    Repeat 100 times, avg time 0.78 ms, max_time 0.78 ms, min_time 0.77 ms
    --------------------------------------

**检测模型**

.. code-block:: bash

    /root/sample # ./ax_yolov5s -m yolov5s.axmodel -i dog.jpg -r 100
    --------------------------------------
    model file : yolov5s.axmodel
    image file : dog.jpg
    img_h, img_w : 640 640
    --------------------------------------
    Engine creating handle is done.
    Engine creating context is done.
    Engine get io info is done.
    Engine alloc io is done.
    Engine push input is done.
    --------------------------------------
    post process cost time:1.66 ms
    --------------------------------------
    Repeat 100 times, avg time 7.67 ms, max_time 7.68 ms, min_time 7.67 ms
    --------------------------------------
    detection num: 4
    16:  93%, [ 182,  291,  411,  721], dog
    2:  72%, [ 626,  101,  919,  231], car
    1:  60%, [ 212,  158,  760,  558], bicycle
    7:  46%, [ 628,  101,  916,  232], truck
    --------------------------------------
