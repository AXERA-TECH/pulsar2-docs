# NPU 算子支持列表(AX615)

本节介绍 `AX615` 中的 **NPU** 对 `ONNX` 算子支持的情况。
 <br> - 支持的 ONNX opset_version >= 11，详细算子描述可参考 [onnx Operators](https://github.com/onnx/onnx/blob/main/docs/Operators.md) 。 <br> - 部分支持的算子尚无标准的 ONNX 定义，如果模型中包含了此类算子，请咨询技术支持。

> :::{note}
> "暂不支持": 表示当前版本算子实现还不支持，但NPU理论上可以支持，后续版本有可能会支持上。
>
> "无限制": 表示当前算子实现上可以支持，由于测试不一定能覆盖到全部参数空间，如果出现意外可以跟我们进行反馈，我们会当BUG来尽快修复。
>
> "不支持": 表示无法支持该属性的实现。
> :::

| 算子名称              | Attrs 约束                                                                                        |
| --------------------- | ------------------------------------------------------------------------------------------------- |
| Abs                   | 无限制                                                                                            |
| Add                   | 无限制                                                                                            |
| ArgMax                | - axis: 无限制 <br> - keepdims: 无限制 <br> - select_last_index: 只支持设为0                                                                                                   |
| ArgMin                | - axis: 无限制 <br> - keepdims: 无限制 <br> - select_last_index: 只支持设为0                                                                                                   |
| AveragePool           | - auto_pad: 只支持NOTSET <br> - ceil_mode: 无限制 <br> - count_include_pad: 只支持设为1 <br> - kernel_shape: 无限制 <br> - pads: 无限制 <br> - strides: 无限制                                                                                                   |
| BatchNormalization    | - epsilon: 无限制 <br> - momentum: 不支持 <br> - training_mode: 不支持                                                                                                   |
| Cast                  | to:uint8/int8/uint16/int16/uint32/int32/float32                                                   |
| Ceil                  | 无限制                                                                                            |
| Clip                  | - min: 无限制 <br> - max: 无限制                                                                                                   |
| Concat                | axis: 无限制                                                                                      |
| Constant              | 无限制                                                                                            |
| Conv                  | - auto_pad: 只支持NOTSET <br> - dilations: 无限制 <br> - group: 无限制 <br> - kernel_shape: 无限制 <br> - pads: 无限制 <br> - strides: 无限制 <br> - note: 当使用DepthWise/Group Conv， 并且dilation不为1时效率较低。                                                                                                   |
| ConvTranspose         | - auto_pad: 只支持NOTSET <br> - dilations: 暂时只能设为1 <br> - group: 无限制 <br> - kernel_shape: 无限制 <br> - output_shape: 暂不支持 <br> - pads: 无限制 <br> - strides: 无限制 <br> - note: DepthWise ConvTranspose 效率较低。output_padding: output_padding_h \<= pads_bottom, output_padding_w \<= pads_right                                                                                                   |
| DepthToSpace          | - blocksize: 无限制 <br> - mode: 暂时只支持DCR                                                                                                   |
| Div                   | 无限制                                                                                            |
| Elu                   | 无限制                                                                                            |
| Equal                 | 无限制                                                                                            |
| Erf                   | 无限制                                                                                            |
| Exp                   | 无限制                                                                                            |
| Expand                | 无限制                                                                                            |
| Flatten               | 无限制                                                                                            |
| Gather                | - axis: 无限制 <br> - indices: 暂时只支持1维                                                                                                   |
| Gelu                  | 无限制                                                                                            |
| Gemm                  | - alpha: 暂不支持 <br> - beta: 暂不支持 <br> - transA: 无限制 <br> - transB: 无限制                                                                                                   |
| GlobalAveragePool     | 无限制                                                                                            |
| GlobalMaxPool         | 无限制                                                                                            |
| Greater               | 无限制                                                                                            |
| GreaterOrEqual        | 无限制                                                                                            |
| HardSigmoid           | 无限制                                                                                            |
| HardSwish             | 无限制                                                                                            |
| Identity              | 无限制                                                                                            |
| InstanceNormalization | epsilon:无限制 仅限小尺寸                                                                         |
| LayerNormalization    | axis暂时只支持为-1(即最后一维) 仅限小尺寸                                                         |
| Less                  | 无限制                                                                                            |
| LessOrEqual           | 无限制                                                                                            |
| LpNormalization       | - axis暂时只支持-1(即最后一维) <br> - p只支持1或2仅限小尺寸                                                                                                   |
| LSTM                  | - activation_alpha: 暂时不支持 <br> - activation_beta: 暂时不支持 <br> - activations: 暂时不支持 <br> - clip: 暂时不支持 <br> - hidden_size: 无限制 <br> - input_forget: 暂时不支持 <br> - layout: 只支持设为0 <br> - B: 无限制 <br> - sequence_lens: 不支持 <br> - initial_h: 无限制 <br> - initial_c: 无限制 <br> - P: 暂时不支持direction: 支持“bidirectional”、“reverse”、“forward”                                                                                                   |
| LeakyRelu             | 无限制                                                                                            |
| MatMul                | 无限制                                                                                            |
| Max                   | 无限制                                                                                            |
| Min                   | 无限制                                                                                            |
| Mish                  | 无限制                                                                                            |
| MaxPool               | - auto_pad: 只支持设为NOTSET <br> - ceil_mode: 无限制 <br> - dilations: 只支持为1 <br> - kernel_shape: 无限制 <br> - pads: 无限制 <br> - storage_order: 只支持设为0 <br> - strides: 无限制                                                                                                   |
| Mul                   | 无限制                                                                                            |
| Neg                   | 无限制                                                                                            |
| PRelu                 | 4D tensor输入时，channel维度在第二维， 并且slope shape暂时只支持(channel,) 或者(1, channel, 1, 1) |
| Pad                   | - pads: 无限制 <br> - constant_value: 无限制 <br> - mode: 只支持constant <br> - axes: 暂不支持                                                                                                   |
| Pow                   | 不支持elemwise计算， exponent只支持initializer形式且为标量。                                      |
| ReduceL2              | - axes: 无限制 <br> - keepdims: 无限制 <br> - noop_with_empty_axes: 该参数暂不支持                                                                                                   |
| ReduceMax             | - axes: 无限制 <br> - keepdims: 无限制 <br> - noop_with_empty_axes: 该参数暂不支持                                                                                                   |
| ReduceMax             | - axes: 无限制 <br> - keepdims: 无限制 <br> - noop_with_empty_axes: 该参数暂不支持                                                                                                   |
| ReduceMean            | - axes: 无限制 <br> - keepdims: 无限制 <br> - noop_with_empty_axes: 该参数暂不支持                                                                                                   |
| Relu                  | 无限制                                                                                            |
| Reshape               | shape: 无限制                                                                                     |
| Resize                | - mode: 支持"nearest"、”linear“可选 <br> - scales: 无限制nearest_mode: 只支持设为round_prefer_ceil                                                                                                   |
| Sigmoid               | 无限制                                                                                            |
| Slice                 | - starts: 无限制 <br> - ends: 无限制 <br> - axes: 无限制 <br> - steps: 无限制                                                                                                   |
| SpatialTransformer    | 插值方式为 "bilinear", 边界处理方式为 "constant"（值为0） 仅限小尺寸                              |
| Split                 | - axis: 无限制 <br> - num_outputs: 无限制                                                                                                   |
| Sqrt                  | 无限制                                                                                            |
| Silu                  | 无限制                                                                                            |
| Sin                   | 无限制                                                                                            |
| Swish                 | 无限制                                                                                            |
| Squeeze               | axes: 无限制                                                                                      |
| Softmax               | axis: 无限制                                                                                      |
| Softplus              | 无限制                                                                                            |
| SpaceToDepth          | blocksize: 无限制                                                                                 |
| Sub                   | 无限制                                                                                            |
| Tanh                  | 无限制                                                                                            |
| Topk                  | 无限制                                                                                            |
| Transpose             | perm: 无限制                                                                                      |
| Unsqueeze             | axes: 无限制                                                                                      |
