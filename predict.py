# predict.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import os


def load_model_and_predict(model_path, image_paths):
    """
    加载训练好的模型并进行预测。

    Args:
        model_path (str): 训练好的模型权重文件路径 (例如 'model.pth')
        image_paths (list or str): 要预测的图像路径列表，或者单个图像路径
    """
    # --- 直接在脚本中定义配置 ---
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for prediction: {device}")

    # 模型参数 (需要与训练时使用的参数完全一致)
    num_classes = 10  # CIFAR-10 有 10 个类别
    dropout_rate = 0.25  # 训练时使用的 dropout_rate
    classifier_dropout = 0.5  # 训练时使用的 classifier_dropout

    # CIFAR-10 数据集的标准化参数 (固定值)
    normalize_mean = (0.4914, 0.4822, 0.4465)
    normalize_std = (0.2023, 0.1994, 0.2010)

    # 2. 定义模型结构 (必须与训练时完全相同)
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=num_classes):
            super(SimpleCNN, self).__init__()
            self.features = nn.Sequential(
                # 第一层卷积
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(dropout_rate),

                # 第二层卷积
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(dropout_rate),

                # 第三层卷积
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(dropout_rate),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 512), # 128 channels * 4 (32/2/2/2) * 4
                nn.ReLU(),
                nn.Dropout(classifier_dropout),
                nn.Linear(512, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    # 3. 初始化模型
    model = SimpleCNN(num_classes=num_classes)

    # 4. 加载训练好的权重
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    print(f"Model weights loaded from: {model_path}")

    # *** 关键修改点：将模型移动到指定设备 ***
    model = model.to(device)
    print(f"Model moved to device: {device}")

    # 5. 设置模型为评估模式 (关闭 Dropout 和 BatchNorm 的训练行为)
    model.eval()

    # 6. 定义测试时的数据预处理 (与训练时的测试集预处理相同)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # CIFAR-10 图像大小为 32x32
        transforms.ToTensor(),        # 转换为张量
        transforms.Normalize(normalize_mean, normalize_std) # 标准化
    ])

    # CIFAR-10 的类别名称
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # 7. 确保 image_paths 是列表
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    # 8. 对每张图像进行预测
    for img_path in image_paths:
        try:
            # 加载图像
            image = Image.open(img_path).convert('RGB') # 确保图像是 RGB 格式

            # 预处理
            input_tensor = transform(image).unsqueeze(0) # 添加批次维度 (1, C, H, W)
            # *** 关键修改点：将输入张量也移动到与模型相同的设备 ***
            input_tensor = input_tensor.to(device)

            # 预测
            with torch.no_grad(): # 不需要计算梯度
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1) # 计算概率
                predicted_idx = outputs.argmax(1).item() # 获取预测的类别索引
                predicted_class = classes[predicted_idx]
                confidence = probabilities[0][predicted_idx].item() # 获取预测类别的概率

            print(f"Image: {img_path}")
            print(f"  Predicted Class: {predicted_class} (Index: {predicted_idx})")
            print(f"  Confidence: {confidence:.4f}")
            print("-" * 40)

        except Exception as e:
            print(f"Error processing image {img_path}: {e}")


def select_image_and_predict():
    """
    打开文件选择对话框，选择图像后进行预测。
    """
    # 创建一个隐藏的根窗口
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口

    # 打开文件选择对话框
    file_paths = filedialog.askopenfilenames(
        title="选择要预测的图像文件 (CIFAR-10 类别)",
        filetypes=(
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("JPEG files", "*.jpg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        )
    )

    if not file_paths:
        print("未选择任何文件。")
        return

    # --- 请修改以下路径 ---
    # 请确保这个路径指向你实际保存的模型文件
    MODEL_PATH = "v1.pth"  # 你的模型权重文件路径

    print(f"Selected {len(file_paths)} file(s). Starting prediction...")
    load_model_and_predict(MODEL_PATH, file_paths)


if __name__ == "__main__":
    select_image_and_predict()