.. _ax_run_model:

=======================
模型评测工具使用说明
=======================

为了方便用户测评模型，在开发板上预制了 ``ax_run_model`` 工具，此工具有若干参数，可以很方便地测试模型速度和精度。

   .. code:: bash

      root@AXERA:~# ax_run_model
      usage: ax_run_model --model=string [options] ...
         options:
         -m, --model                path to a model file (string)
         -r, --repeat               repeat times running a model (int [=1])
         -w, --warmup               repeat times before running a model to warming up (int [=1])
         -a, --affinity             npu affinity when running a model (int [=1])
         -v, --vnpu                 type of Visual-NPU inited {0=Disable, 1=STD, 2=BigLittle} (int [=0])
         -b, --batch                the batch will running (int [=0])
         -i, --input-folder         the folder of each inputs (folders) located (string [=])
         -o, --output-folder        the folder of each outputs (folders) will saved in (string [=])
         -l, --list                 the list of inputs which will test (string [=])
               --inputs-is-folder     each time model running needs inputs stored in each standalone input folders
               --outputs-is-folder    each time model running saved outputs stored in each standalone output folders
               --use-tensor-name      using tensor names instead of using tensor indexes when loading & saving io files
               --verify               verify outputs after running model
               --save-benchmark       save benchmark result(min, max, avg) as a json file
         -?, --help                 print this message


-----------------------------
参数说明
-----------------------------

测评工具参数主要有两部分.

第一部分是与测速有关的参数：

.. data:: ax_run_model 参数解释

  --model

    - 数据类型：string
    - 是否必选：是
    - 描述：指定测试模型的路径

  --repeat

    - 数据类型：int
    - 是否必选：否
    - 描述：指定要测试的循环次数，然后显示min/max/avg的速度

  --warmup 
  
    - 数据类型：int
    - 是否必选：否
    - 描述：循环测试前，预热的次数

  --affinity
  
    - 数据类型：int
    - 是否必选：否
    - 描述：亲和性的mask值，大于1(0b001)，小于7(0b111)

  --vnpu
  
    - 数据类型：int
    - 是否必选：否
    - 描述：虚拟npu模式；0 禁用虚拟npu；1标准切分模式；2大小核模式

  --batch 
  
    - 数据类型：int
    - 是否必选：否
    - 描述：指定测试的batch

  --input-folder
  
    - 数据类型：string
    - 是否必选：否
    - 描述：指定用于精度测试的输入文件夹
  
  --output-folder
  
    - 数据类型：string
    - 是否必选：否
    - 描述：指定用于精度测试的输出文件夹

  --list
  
    - 数据类型：string
    - 是否必选：否
    - 描述：指定测试列表

  --inputs-is-folder
  
    - 数据类型：string
    - 是否必选：否
    - 描述：指定输入路径参数--input-folder是由文件夹组成的

  --outputs-is-folder
  
    - 数据类型：string
    - 是否必选：否
    - 描述：指定输出径参数--out-folder是由文件夹组成的

  --use-tensor-name
  
    - 数据类型：string
    - 是否必选：否
    - 描述：指定按模型输入输出名字查找激励文件，不设置是按索引查找
  
  --verify
  
    - 数据类型：string
    - 是否必选：否
    - 描述：指定不保存模型输出且指定的目录输出文件已存在，进行逐byte比较

-----------------------------
使用示例
-----------------------------

以测速需求为例，假设已经转换完成了一个单核心的 ``YOLOv5s`` 模型，现在想要知道上板子运行的速度，那么可以参考运行如下命令：

   .. code:: bash

      root@AXERA:~# ax_run_model -m /opt/data/npu/models/yolov5s.axmodel -w 10 -r 100
      [Axera version]: libax_sys.so V1.13.0 Apr 26 2023 16:24:35
      Run AxModel:
            model: /opt/data/npu/models/yolov5s.axmodel
             type: NPU1
             vnpu: Disable
         affinity: 0b001
           repeat: 100
           warmup: 10
            batch: 1
      pulsar2 ver: 1.2-patch2 7e6b2b5f
       engine ver: [Axera version]: libax_engine.so V1.13.0 Apr 26 2023 16:48:53 1.1.0
         tool ver: 1.0.0
         cmm size: 12730188 Bytes
      ------------------------------------------------------
      min =   7.658 ms   max =   7.672 ms   avg =   7.662 ms
      ------------------------------------------------------


从打印的 log 可以看出，VNPU 被初始化成 standard 模式，此时NPU被分作三份；并且这次测速时亲和性设置为亲和序号最大的那个模型。

通过设置亲和性，可以很方便地在不编写代码的情况下，同时跑多个模型进行测速。

比如，在一个 SSH 终端窗口里，运行模型 a 数万次，然后在另一个 SSH 终端里，设置不同的亲和性，观察模型 b 速度相较于没有运行模型a时的速度下降，就可以得知极高负载情况下，模型b受模型 a 运行的影响(这可能比真实情况更严苛)。需要注意的是，两个 SSH 里， ``-v`` 参数需要是一致的。

另一个很常见的需求是转完了模型，想要知道板子上的精度如何，这可以通过精度的参数进行测试。

以分类模型为例，说明目录结构和参数的使用，这里以两个目录结构举例。

下面是模式一：

   .. code:: bash

      root@AXERA:~# tree /opt/data/npu/temp/
      /opt/data/npu/temp/
      |-- input
      |   `-- 0.bin
      |-- list.txt
      |-- mobilenet_v1.axmodel
      `-- output
         `-- 0.bin

      2 directories, 4 files


下面是模式二：

   .. code:: bash

      root@AXERA:~# tree /opt/data/npu/temp/
      /opt/data/npu/temp/
      |-- input
      |   `-- 0
      |       `-- 0.bin
      |-- list.txt
      |-- mobilenet_v1.axmodel
      `-- output
         `-- 0
            `-- 0.bin

      4 directories, 4 files

这是非常常见的两类测试精度的目录结构.
* 模式一比较简单，输入和输出都全部包含在同一个文件夹里，特别适合单输入单输出的模型；
* 模式二则将每一组模型激励作为一个文件夹，特别适合多输入多输出的模型。

此外，将模式二稍加变化，将激励的文件按tensor的名字命名，则有模式三的目录结构如下：

   .. code:: bash

      root@AXERA:~# tree /opt/data/npu/temp/
      /opt/data/npu/temp/
      |-- input
      |   `-- 0
      |       `-- data.bin
      |-- list.txt
      |-- mobilenet_v1.axmodel
      `-- output
         `-- 0
            `-- prob.bin

      4 directories, 4 files

测试精度时必须的参数是 ``-m -i -o -l``，分别指定模型、输入文件夹、输出文件夹、和待测试的输入列表。

* 模式一比较简单，无需附加其他参数；
* 模式二因为输入和输出都是文件夹，则需要附加参数 ``--inputs-is-folder和--outputs-is-folder`` 参数；
* 模式三在模式二的参数基础上，还需要附加参数 ``--use-tensor-name`` 才能运行，需要注意的是， ``--use-tensor-name`` 参数同时影响输入和输出。

此外，这三个模式的输出文件夹都非空，在运行命令时输出文件夹的已有文件会被覆盖；但如果是已经从 ``Pulsar2`` 仿真拿到的输出 ``golden`` 文件，
则可以通过附加 ``--verify`` 参数不覆写输出文件，而是读取输出文件夹的已有文件，和当前模型的输出在内存中进行逐位比较，这个模式在怀疑仿真和上板精度不对齐时特别有用。

参数 ``-l`` 指定的激励列表在这三种模式下都是一样的：

   .. code:: bash

      root@AXERA:~# cat /opt/data/npu/temp/list.txt
      0
      root@AXERA:~#


也就是这三种模式下，指定的都是唯一一个激励文件(夹)。这个参数在数据集很大时非常有用，比如输入文件夹是完整的 ``ImageNet`` 数据集，文件非常多；
但这次测试时只希望测10个文件验证一下，如果没有异常再跑全量的测试，那么这样的需求可以通过创建两个 ``list.txt`` 完成，一个list里保存的只有10行激励，一个list文件里是全部的激励。
以下是模式三，并且进行模型 ``verify`` 的需求进行举例， ``ax_run_model`` 参数运行示例如下：

   .. code:: bash

      root@AXERA:~# ax_run_model -m /opt/data/npu/temp/mobilenet_v1.axmodel -i /opt/data/npu/temp/input/ -o /opt/data/npu/temp/output/ -l /opt/data/npu/temp/list.txt --inputs-is-folder --outputs-is-folder --use-tensor-name --verify
      [Axera version]: libax_sys.so V1.13.0 Apr 26 2023 16:24:35
       total found {1} input drive folders.
       infer model, total 1/1. Done.
       ------------------------------------------------------
       min =   3.347 ms   max =   3.347 ms   avg =   3.347 ms
       ------------------------------------------------------

      root@AXERA:~#

可以看出，这个模型在这组输入输出binary文件下，输出是逐位对齐的。如果没有对齐，打印会报告没有对齐的 ``byte`` 偏移量。
