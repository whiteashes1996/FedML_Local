import torch as t
from torch import nn

# in_features由输入张量的形状决定，out_features则决定了输出张量的形状
connected_layer = nn.Linear(in_features = 28*28, out_features = 1)

# 假定输入的图像形状为[64,64,3]
input = t.randn(20,28,28)

# 将四维张量转换为二维张量之后，才能作为全连接层的输入
input = input.view(20,28*28)
print(input.size())
print(input.shape)
output = connected_layer(input) # 调用全连接层
print(output.shape)