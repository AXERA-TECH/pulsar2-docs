# QAT 4W8F 支持

QAT (Quantization-Aware Training) 量化感知训练是指在模型训练过程中模拟量化过程, 使模型适应低精度
计算, 以减少量化后的精度损失。QAT 通常在前向传播中插入伪量化 (Fake Quantization) 操作, 模拟 INT8 量化, 但
在反向传播时仍使用 FP32 计算梯度.

在量化感知训练中我们可以采用 4 比特权重量化 (Weights) + 8 比特激活量化 (Feature maps/Activations) 的
混合精度量化策略, 旨在平衡模型压缩率与推理精度。这种配置意味着在量化感知训练过程中：

- 权重 (Weights) 会被量化为 4 比特 (INT4) , 以减少模型大小和计算量。
- 激活值 (Activations) 仍保持 8 比特 (INT8) , 以平衡计算效率和精度

考虑到 PyTorch 还不支持正式的 INT4 weight 格式, 用户从 QAT 训练并导出 ONNX 模型, 实际上仍然使用 int8 表示, 不过
位宽已经限制在 INT4 范围, 首先需要执行如下命令将 ONNX 模型转成真正的 INT4 的 ONNX 模型：

```bash
onnxslim model_qat_4w8f.onnx model_qat_4w8f_slim.onnx
convert_to_4w8f_cli --input model_qat_4w8f_slim.onnx --output model_qat_4w8f_opt.onnx
```

这里假设 QAT 得到的 QDQ ONNX 模型名称为 "model_qat_4w8f.onnx", 由此转换出的模型 "model_qat_4w8f_opt.onnx"
是满足 ONNX 语义的 Weights 标记为 INT4 的模型.
