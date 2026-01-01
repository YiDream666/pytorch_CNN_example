import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import yaml
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import sys
import os

# ----------------------------
# 1. 定义与训练时完全一致的模型结构
# ----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class OptimizedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(OptimizedCNN, self).__init__()
        # 注意：这里必须与训练时的"加宽"配置保持一致 (64, 128, 256)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(64, 128, 3, stride=2) 
        self.layer3 = self._make_layer(128, 256, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ----------------------------
# 2. 配置与类别定义
# ----------------------------
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def load_config():
    if not os.path.exists('config_optimized.yaml'):
        print("Error: config_optimized.yaml not found.")
        sys.exit(1)
    with open('config_optimized.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def select_image():
    root = tk.Tk()
    root.withdraw() # 隐藏主窗口
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    return file_path

# ----------------------------
# 3. 预测主逻辑
# ----------------------------
def main():
    cfg = load_config()
    
    # 获取设备 (预测通常用 CPU 也可以，这里自动检测)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on: {device}")

    # 1. 加载模型
    model_path = cfg['saving']['model_path']
    # 如果有 EMA 最终权重，优先加载 EMA 版本 (精度更高)
    ema_path = model_path.replace('.pth', '_final_ema.pth')
    if os.path.exists(ema_path):
        load_path = ema_path
        print(f"Loading EMA model: {load_path}")
    elif os.path.exists(model_path):
        load_path = model_path
        print(f"Loading Best model: {load_path}")
    else:
        print(f"Error: Model file not found at {model_path}")
        return

    model = OptimizedCNN(num_classes=10).to(device)
    
    try:
        # map_location 确保即使在 GPU 训练的模型也能在 CPU 上跑
        state_dict = torch.load(load_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print("\n!!! 模型加载失败 !!!")
        print("原因可能是：代码中的模型定义(OptimizedCNN)与保存的权重文件结构不匹配。")
        print("请确保 predict_gui.py 中的 OptimizedCNN 类定义与您训练时的完全一致（特别是通道数）。")
        print(f"错误详情: {e}")
        return

    model.eval()

    # 2. 准备数据预处理
    # 必须与训练时保持一致的均值和方差
    normalize_mean = cfg['normalization']['mean']
    normalize_std = cfg['normalization']['std']

    transform = transforms.Compose([
        transforms.Resize((32, 32)), # 强制调整大小，防止输入大图报错
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std)
    ])

    # 3. 选择图片
    print("Opening file dialog...")
    img_path = select_image()
    if not img_path:
        print("No image selected.")
        return

    # 4. 预测
    original_img = Image.open(img_path).convert('RGB')
    input_tensor = transform(original_img).unsqueeze(0).to(device) # 增加 batch 维度 [1, 3, 32, 32]

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        prob, predicted_idx = torch.max(probabilities, 1)

    predicted_class = CIFAR10_CLASSES[predicted_idx.item()]
    confidence = prob.item() * 100

    print(f"\nResult: {predicted_class} ({confidence:.2f}%)")

    # 5. 可视化结果
    plt.figure(figsize=(6, 6))
    plt.imshow(original_img)
    plt.title(f"Pred: {predicted_class}\nConf: {confidence:.2f}%")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()