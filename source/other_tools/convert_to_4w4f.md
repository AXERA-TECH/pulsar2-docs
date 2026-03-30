# QAT 4bit 支持

QAT (Quantization-Aware Training) 量化感知训练是指在模型训练过程中模拟量化过程, 使模型适应低精度
计算, 以减少量化后的精度损失。QAT 通常在前向传播中插入伪量化 (Fake Quantization) 操作, 模拟低比特量化, 但
在反向传播时仍使用 FP32 计算梯度.

4bit 量化位宽时参考 [resnet50/config_4w4f](https://github.com/AXERA-TECH/QAT.axera/blob/cc4c50293317e21dc1b7f52854d992df48d4ffd8/resnet50/config_4w4f.json) 配置，
并用 [simplify_and_fix_4bit_dtype](https://github.com/AXERA-TECH/QAT.axera/blob/cc4c50293317e21dc1b7f52854d992df48d4ffd8/utils/quant_utils.py#L12)
替代 onnxsim/onnxslim .
