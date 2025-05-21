import torch
print(torch.backends.mps.is_available())  # 返回 True 表示有 GPU
print(torch.backends.mps.is_built())  # 返回 GPU 数量

import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"当前使用设备: {device}")