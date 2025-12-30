# cifar10_cnn_no_aug.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import yaml
import os  # 导入 os 模块用于路径操作


def main():
    # ----------------------------
    # 1. 从 YAML 文件加载配置
    # ----------------------------
    with open('config.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 从 cfg 字典中读取配置
    device = torch.device(cfg['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 训练参数
    batch_size = cfg['training']['batch_size']
    epochs = cfg['training']['epochs']
    learning_rate = cfg['training']['learning_rate']

    # 数据集参数
    data_root = cfg['data']['root']
    num_workers = cfg['data']['num_workers']
    download_data = cfg['data']['download']  # 虽然我们设为 true，但下载后设为 false

    # 模型参数
    num_classes = cfg['model']['num_classes']
    dropout_rate = cfg['model']['dropout_rate']
    classifier_dropout = cfg['model']['classifier_dropout']

    # 标准化参数 (CIFAR-10 数据集的标准参数)
    normalize_mean = cfg['normalization']['mean']
    normalize_std = cfg['normalization']['std']

    # 保存路径
    model_save_path = cfg['saving']['model_path']
    plot_save_path = cfg['saving']['plot_path']

    print(f"Loading data from: {data_root}")
    print(f"Download flag is set to: {download_data}")

    # 调试信息
    print(f"Configured data_root: {data_root}")
    print(f"Absolute path to data_root: {os.path.abspath(data_root)}")

    # ----------------------------
    # 2. 数据预处理与加载 (不使用数据增强，仅使用 ToTensor 和 Normalize)
    # ----------------------------
    # 定义仅包含 ToTensor 和 Normalize 的变换
    transform = transforms.Compose([
        transforms.ToTensor(), # 将 PIL 图像或 numpy 数组转换为张量
        transforms.Normalize(normalize_mean, normalize_std) # 标准化
    ])

    # 使用相同的变换应用于训练集和测试集
    trainset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        download=download_data,  # 传入 yaml 中的 download 标志
        transform=transform      # 使用无增强的变换
    )

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        download=download_data,
        transform=transform      # 使用无增强的变换
    )

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # ----------------------------
    # 3. 定义 CNN 模型 (使用 YAML 中的参数)
    # ----------------------------
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
                nn.Dropout(dropout_rate),  # 使用 YAML 中的参数

                # 第二层卷积
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(dropout_rate),  # 使用 YAML 中的参数

                # 第三层卷积
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(dropout_rate),  # 使用 YAML 中的参数
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 512),
                nn.ReLU(),
                nn.Dropout(classifier_dropout),  # 使用 YAML 中的参数
                nn.Linear(512, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    # ----------------------------
    # 4. 初始化模型、损失函数和优化器
    # ----------------------------
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ----------------------------
    # 5. 训练函数
    # ----------------------------
    def train_one_epoch():
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        acc = 100. * correct / total
        avg_loss = running_loss / len(trainloader)
        return avg_loss, acc

    # ----------------------------
    # 6. 测试函数
    # ----------------------------
    def evaluate():
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        acc = 100. * correct / total
        return acc

    # ----------------------------
    # 7. 主训练循环
    # ----------------------------
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    print("Start training (without data augmentation)...")
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch()
        test_acc = evaluate()

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        print(f"Epoch [{epoch + 1}/{epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

    print("Training finished (without data augmentation)!")

    # ----------------------------
    # 8. 可视化训练过程（可选）
    # ----------------------------
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title('Loss Curve (No Augmentation)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(test_accuracies, label='Test Acc')
    plt.title('Accuracy Curve (No Augmentation)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(plot_save_path)  # 使用 YAML 中的路径
    plt.show()

    # ----------------------------
    # 9. 保存模型（可选）
    # ----------------------------
    torch.save(model.state_dict(), model_save_path)  # 使用 YAML 中的路径
    print(f"Model (without augmentation) saved as '{model_save_path}'")


if __name__ == '__main__':
    main()
    # 可选：添加 freeze_support() 如果你计划将脚本打包成可执行文件
    # from multiprocessing import freeze_support
    # freeze_support()
    # main()