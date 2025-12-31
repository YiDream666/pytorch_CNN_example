# cifar10_cnn_optimized.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.amp as amp_torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import yaml
import os
import numpy as np
from tqdm import tqdm  # 引入进度条库


# 检查 torch.compile 可用性 (仅定义函数，不调用)
def check_compile_availability():
    try:
        # 简单检查，不打印冗余信息
        import torch._dynamo
        return True
    except ImportError:
        return False


def main():
    # ----------------------------
    # 0. 环境初始化与配置加载
    # ----------------------------
    # 修复 Bug: 将检查移入 main 函数，防止多进程重复打印
    compile_available = check_compile_availability()

    with open('config_optimized.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # [优化 1] 开启 CuDNN 自动调优 (针对固定尺寸输入加速显著)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print("Creating optimized CuDNN execution plan...")

    # 提取参数
    batch_size = int(cfg['training']['batch_size'])
    epochs = int(cfg['training']['epochs'])
    learning_rate = float(cfg['training']['learning_rate'])
    weight_decay = float(cfg['training']['weight_decay'])

    data_root = cfg['data']['root']
    num_workers = int(cfg['data']['num_workers'])
    download_data = cfg['data']['download']

    num_classes = int(cfg['model']['num_classes'])
    # dropout_rate = float(cfg['model']['dropout_rate']) # 当前模型未使用此参数，暂保留
    classifier_dropout = float(cfg['model']['classifier_dropout'])

    random_crop_padding = int(cfg['data_augmentation']['random_crop_padding'])
    random_hflip_prob = float(cfg['data_augmentation']['random_hflip_prob'])

    normalize_mean = cfg['normalization']['mean']
    normalize_std = cfg['normalization']['std']

    model_save_path = cfg['saving']['model_path']
    plot_save_path = cfg['saving']['plot_path']

    # ----------------------------
    # 2. 数据加载
    # ----------------------------
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=random_crop_padding, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(p=random_hflip_prob),
        # transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10), # 可选：自动增强
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std)
    ])

    trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=download_data,
                                            transform=transform_train)
    # persistent_workers=True 可以减少每个 epoch 重新创建进程的开销
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))

    testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=download_data,
                                           transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))

    # ----------------------------
    # 3. 模型定义
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
            residual = self.shortcut(x)
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += residual
            out = self.relu(out)
            return out

    class OptimizedCNN(nn.Module):
        def __init__(self, num_classes=num_classes, classifier_dropout=classifier_dropout):
            super(OptimizedCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(32)
            self.relu = nn.ReLU(inplace=True)

            self.layer1 = self._make_layer(32, 32, 2, stride=1)
            self.layer2 = self._make_layer(32, 64, 2, stride=2)
            self.layer3 = self._make_layer(64, 128, 2, stride=2)

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.dropout = nn.Dropout(classifier_dropout)
            self.fc = nn.Linear(128, num_classes)

        def _make_layer(self, in_channels, out_channels, num_blocks, stride):
            layers = []
            layers.append(ResidualBlock(in_channels, out_channels, stride))
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
            x = self.dropout(x)
            x = self.fc(x)
            return x

    # ----------------------------
    # 4. 初始化
    # ----------------------------
    model = OptimizedCNN().to(device)

    # [优化 2] 使用 Channels Last 内存格式 (Tensor Core 友好，加速 10-20%)
    model = model.to(memory_format=torch.channels_last)

    if compile_available:
        print("Torch compile available but skipped (as per config).")

    # [优化 3] 标签平滑 (Label Smoothing) - 防止过拟合
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # [优化 4] 动态设置 AMP 设备类型
    use_amp = (device.type == 'cuda')
    scaler = amp_torch.GradScaler(device.type, enabled=use_amp)

    # ----------------------------
    # 5. 训练函数 (集成 TQDM)
    # ----------------------------
    def train_one_epoch(epoch_index):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # [优化 5] TQDM 进度条
        # leave=False 表示跑完该 Epoch 后进度条消失，保持控制台整洁
        loop = tqdm(trainloader, desc=f'Epoch [{epoch_index}/{epochs}]', leave=False)

        for inputs, labels in loop:
            # 配合 Channels Last 优化输入数据
            inputs = inputs.to(device, memory_format=torch.channels_last, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with amp_torch.autocast(device.type, enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 实时更新进度条后缀
            current_acc = 100. * correct / total
            loop.set_postfix(loss=loss.item(), acc=f"{current_acc:.2f}%")

        avg_loss = running_loss / len(trainloader)
        final_acc = 100. * correct / total
        return avg_loss, final_acc

    # ----------------------------
    # 6. 测试函数
    # ----------------------------
    def evaluate():
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs = inputs.to(device, memory_format=torch.channels_last, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with amp_torch.autocast(device.type, enabled=use_amp):
                    outputs = model(inputs)

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        acc = 100. * correct / total
        return acc

    # ----------------------------
    # 7. 主循环
    # ----------------------------
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    best_test_acc = 0.0

    print(f"\nStart training on {device} with {'AMP' if use_amp else 'FP32'}...")

    try:
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_one_epoch(epoch)
            test_acc = evaluate()

            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step()

            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)

            # 仅在性能更好时保存，并打印醒目信息
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save(model.state_dict(), model_save_path)
                save_msg = f"--> Saved Best Model ({best_test_acc:.2f}%)"
            else:
                save_msg = ""

            # 使用格式化打印，对齐列
            print(f"Epoch {epoch:03d}/{epochs:03d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.2f}% | "
                  f"Test Acc: {test_acc:.2f}% | "
                  f"LR: {current_lr:.5f} {save_msg}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving plots...")

    print(f"\nTraining finished! Best Test Accuracy: {best_test_acc:.2f}%")

    # ----------------------------
    # 8. 可视化
    # ----------------------------
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='tab:red')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Acc', color='tab:blue')
    plt.plot(test_accuracies, label='Test Acc', color='tab:orange')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(plot_save_path)
    print(f"Plot saved to {plot_save_path}")
    plt.show() # 如果在服务器上运行，请注释此行

    # 保存最终模型
    final_model_path = model_save_path.replace('.pth', '_final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved as '{final_model_path}'")


if __name__ == '__main__':
    main()