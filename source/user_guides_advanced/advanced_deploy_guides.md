(model-deploy-advanced)=

# 模型部署进阶指南

## 概述

本章节介绍开发板上 NPU 相关示例程序的使用方式，相关示例程序源码参考 SDK 中 `msp/sample/npu` 目录，如何编译出 NPU 示例代码请参考 《AX SDK 使用说明》.

## 运行示例

**运行准备**

对于 `AX650A`、 `AX650N`、 `M76H`、 `AX630C` 开发板，NPU 相关示例已预装在 `/opt/bin/` 路径下，分别为 `sample_npu_classification` 和 `sample_npu_yolov5s`.

对于 `AX620Q` 开发板，由于默认采用 16M NorFlash 方案，文件系统中未包含上述2个示例，可通过 NFS 网络挂载的方式将 SDK 中 `msp/out/bin/` 路径挂载到开发板的文件系统中获取以上示例.

如果提示板子空间不足, 可以通过文件夹挂载的方式解决.

**MacOS 挂载 ARM 开发板示例**

:::{hint}
由于板上空间有限, 测试时通常需要进行文件夹共享操作, 这个时候就需要将 `ARM` 开发板与主机之间进行共享. 这里仅以 `MacOS` 为例.
:::

开发机挂载 `ARM` 开发板需要 `NFS` 服务, 而 `MacOS` 系统自带 `NFS` 服务, 只需要创建 `/etc/exports` 文件夹, `nfsd` 将自动启动并开始用于 `exports`.

`/etc/exports` 可以配置如下:

```shell
/path/your/sharing/directory -alldirs -maproot=root:wheel -rw -network xxx.xxx.xxx.xxx -mask 255.255.255.0
```

参数释义

```{eval-rst}
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
```

修改 `/etc/exports` 需要重启 `nfsd` 服务

```bash
sudo nfsd restart
```

如果配置成功, 可以使用

```bash
sudo showmount -e
```

命令查看挂载信息, 例如输出 `/Users/skylake/board_nfs 10.168.21.xx`, 配置好开发机后需要在 `ARM` 端执行 `mount` 指令

```bash
mount -t nfs -o nolock,tcp macos_ip:/your/shared/directory /mnt/directory
```

如果出现权限问题, 需要检查 `maproot` 参数是否正确.

:::{hint}
`network` 参数可以配置成网段的形式, 如: `10.168.21.0`, 如果挂载单ip出现 `Permission denied`, 可以尝试一下网段内挂载.
:::

**分类模型**

以下打印信息基于 AX650N 开发板运行输出，非 AX650N 开发板的打印信息以实际打印为准.

```bash
/root # sample_npu_classification -m /opt/data/npu/models/mobilenetv2.axmodel -i /opt/data/npu/images/cat.jpg -r 10
--------------------------------------
model file : /opt/data/npu/models/mobilenetv2.axmodel
image file : /opt/data/npu/images/cat.jpg
img_h, img_w : 224 224
--------------------------------------
Engine creating handle is done.
Engine creating context is done.
Engine get io info is done.
Engine alloc io is done.
Engine push input is done.
--------------------------------------
topk cost time:0.07 ms
9.5094, 285
9.3773, 282
9.2452, 281
8.5849, 283
7.6603, 287
--------------------------------------
Repeat 10 times, avg time 0.72 ms, max_time 0.72 ms, min_time 0.72 ms
--------------------------------------
```

**检测模型**

```bash
/root # sample_npu_yolov5s -m /opt/data/npu/models/yolov5s.axmodel -i /opt/data/npu/images/dog.jpg -r 10
--------------------------------------
model file : /opt/data/npu/models/yolov5s.axmodel
image file : /opt/data/npu/images/dog.jpg
img_h, img_w : 640 640
--------------------------------------
Engine creating handle is done.
Engine creating context is done.
Engine get io info is done.
Engine alloc io is done.
Engine push input is done.
--------------------------------------
post process cost time:2.25 ms
--------------------------------------
Repeat 10 times, avg time 7.65 ms, max_time 7.66 ms, min_time 7.65 ms
--------------------------------------
detection num: 3
16:  91%, [ 138,  218,  310,  541], dog
2:  69%, [ 470,   76,  690,  173], car
1:  56%, [ 158,  120,  569,  420], bicycle
--------------------------------------
```

## 其他示例

请参考我们在 github 上的开源项目：

- [AX-Samples](https://github.com/AXERA-TECH/ax-samples)
