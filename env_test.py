import torch
print(torch.__version__)
print(torch.cuda.is_available())  # 应返回 True（如果安装了 CUDA 版本且 GPU 可用）
print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
