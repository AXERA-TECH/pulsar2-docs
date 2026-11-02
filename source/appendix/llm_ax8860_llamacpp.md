# AX8860 大模型部署（llama.cpp）

:::{note}
本章节内容正在补充中。
:::

`AX8860` 的大模型推理采用**开箱即用**的方式：官方已提供预置 DSL kernel 与 `llama.cpp-axera` 适配，
不需要使用 `pulsar2 llm_build2` 编译模型。

## 适用范围

- 平台：`AX8860`
- 版本：Pulsar2 8.0

## 环境准备

<!-- TODO: llama.cpp-axera 的获取方式（仓库 / 预编译二进制 / wheel）与依赖 -->

## 模型获取

<!-- TODO: 支持的模型格式（如 GGUF）与获取方式 -->

## 运行模型

<!-- TODO: 启动命令与常用参数 -->

## 调用验证

<!-- TODO: 命令行交互 / API 调用示例 -->

## 与 SGLang-axera 的差异

| 项目 | SGLang-axera | llama.cpp-axera |
| --- | --- | --- |
| 定位 | 服务化、OpenAI 兼容 API | 轻量单机、离线运行 |
| 适用场景 | 多用户、业务联调 | 单机验证、边缘设备 |

## 常见问题

<!-- TODO -->
