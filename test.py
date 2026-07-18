import torch
import torch.nn as nn

B = 1
C_in = 4
L = 5
C_out = 4
dilation = 2

x = torch.tensor([
    [1.0, 3.0, 2.0, 5.0, 4.0],
    [0.5, 1.2, 3.1, 2.2, 0.8],
    [2.1, 0.9, 1.5, 3.3, 2.6],
    [1.8, 2.4, 0.7, 1.1, 3.5]
]).unsqueeze(0)
print("原始输入 x shape:", x.shape)
print("x = \n", x, "\n")

# 空洞卷积权重：输出通道oc只使用输入通道oc，其余通道权重=0
conv_dilated = nn.Conv1d(C_in, C_out, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
w_dil = torch.zeros((C_out, C_in, 3))
for i in range(4):
    w_dil[i, i, :] = 0.2  # 仅同通道有权重，通道间不交互
conv_dilated.weight.data = w_dil
out_dil = conv_dilated(x)
print("=== 空洞3卷积输出 out_dil（通道独立，无跨通道混合）===")
print(out_dil, "\n")

# 1×1卷积权重不变，做通道融合
conv_1x1 = nn.Conv1d(C_out, C_out, kernel_size=1, bias=False)
w_1 = torch.tensor([
    [0.1, 0.3, 0.2, 0.4],
    [0.5, 0.1, 0.2, 0.2],
    [0.2, 0.2, 0.5, 0.1],
    [0.3, 0.2, 0.1, 0.4],
]).unsqueeze(-1)
conv_1x1.weight.data = w_1
out_1x1 = conv_1x1(out_dil)
print("=== 1×1卷积输出 out_1x1（跨通道融合，数值全部改变）===")
print(out_1x1, "\n")

# 验算t0
t0 = out_dil[0, :, 0]
print("t0 空洞卷积各通道值：", t0)
W = w_1.squeeze(-1)
calc = W @ t0
print("手动1×1融合结果：", calc)
print("网络输出t0：", out_1x1[0, :, 0])