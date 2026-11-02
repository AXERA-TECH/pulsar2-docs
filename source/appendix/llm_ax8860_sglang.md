# AX8860 大模型部署（SGLang）

:::{note}
本章节内容正在补充中。
:::

`AX8860` 的大模型推理采用**开箱即用**的方式：官方已提供预置 DSL kernel 与 `SGLang-axera` 适配，
不需要使用 `pulsar2 llm_build2` 编译模型。

## 适用范围

- 平台：`AX8860`
- 版本：Pulsar2 8.0

## 环境准备

<!-- TODO: 需要安装的 wheel（axnn / torch_axera / sglang_axera / sgl_kernel_axera）、Python 版本与安装顺序 -->

## 模型获取

<!-- TODO: Hugging Face 模型目录、量化格式与目录结构 -->

## 启动服务

<!-- TODO: 启动命令、端口、常用参数 -->

## 调用验证

<!-- TODO: OpenAI 兼容 API / curl 示例 -->

## 与 llama.cpp 的差异

| 项目 | SGLang-axera | llama.cpp-axera |
| --- | --- | --- |
| 定位 | 服务化、OpenAI 兼容 API | 轻量单机、离线运行 |
| 适用场景 | 多用户、业务联调 | 单机验证、边缘设备 |

## 常见问题

<!-- TODO -->
