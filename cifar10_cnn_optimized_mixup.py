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
import random
from tqdm import tqdm
from copy import deepcopy

# ----------------------------
# 辅助类：Model EMA (指数移动平均)
# ----------------------------
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay

    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.module.state_dict()
            for k in msd:
                model_v = msd[k].detach()
                esd[k].copy_(esd[k] * self.decay + model_v * (1. - self.decay))

# ----------------------------
# 辅助函数：Mixup 数据增强
# ----------------------------
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).cuda() if use_cuda else torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ----------------------------
# 模型定义 (ResNet 风格)
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
        return self.relu(out)

class OptimizedCNN(nn.Module):
    def __init__(self, num_classes=10, classifier_dropout=0.1):
        super(OptimizedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(64, 128, 3, stride=2) 
        self.layer3 = self._make_layer(128, 256, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(classifier_dropout)
        self.fc = nn.Linear(256, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
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
        x = self.dropout(x)
        return self.fc(x)

def main():
    # ----------------------------
    # 0. 环境初始化
    # ----------------------------
    with open('config_optimized_mixup.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    target_device_str = cfg['device'] if torch.cuda.is_available() else "cpu"
    device = torch.device(target_device_str)
    print(f"Using device: {device}")

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True 

    batch_size = int(cfg['training']['batch_size'])
    epochs = int(cfg['training']['epochs'])
    lr = float(cfg['training']['learning_rate'])
    wd = float(cfg['training']['weight_decay'])
    model_save_path = cfg['saving']['model_path']
    plot_save_path = cfg['saving']['plot_path']

    # ----------------------------
    # 2. 数据加载 (优化 DataLoader)
    # ----------------------------
    normalize = transforms.Normalize(cfg['normalization']['mean'], cfg['normalization']['std'])
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.2),
    ])

    trainset = torchvision.datasets.CIFAR10(root=cfg['data']['root'], train=True, download=True, transform=transform_train)
    
    # 针对 13600KF 和 Windows 优化的 DataLoader
    trainloader = DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=int(cfg['data']['num_workers']), 
        pin_memory=True,
        persistent_workers=True  # <--- 新增：保持子进程常驻，减少 Epoch 切换开销
    )

    testset = torchvision.datasets.CIFAR10(root=cfg['data']['root'], train=False, download=True, 
                                          transform=transforms.Compose([transforms.ToTensor(), normalize]))
    
    testloader = DataLoader(
        testset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=int(cfg['data']['num_workers']), 
        pin_memory=True,
        persistent_workers=True  # 减少测试时的等待
    )

    # ----------------------------
    # 3. 模型与优化器初始化
    # ----------------------------
    model = OptimizedCNN(num_classes=10).to(device)
    model = model.to(memory_format=torch.channels_last)

    print("Standard mode enabled (torch.compile disabled for stability).")

    ema_model = ModelEMA(model, decay=0.999)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = amp_torch.GradScaler(device.type, enabled=(device.type == 'cuda'))

    # ----------------------------
    # 4. 训练与测试循环
    # ----------------------------
    train_losses, test_accs = [], []
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        loop = tqdm(trainloader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        
        for inputs, labels in loop:
            inputs = inputs.to(device, memory_format=torch.channels_last, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, use_cuda=(device.type=='cuda'))

            optimizer.zero_grad()
            with amp_torch.autocast(device.type, enabled=(device.type == 'cuda')):
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            
            ema_model.update(model)
            running_loss += loss.item()

        # 评估 EMA 模型
        ema_model.module.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs = inputs.to(device, memory_format=torch.channels_last)
                labels = labels.to(device)
                outputs = ema_model.module(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        acc = 100. * correct / total
        test_accs.append(acc)
        train_losses.append(running_loss / len(trainloader))
        
        scheduler.step()
        
        save_msg = ""
        if acc > best_acc:
            best_acc = acc
            torch.save(ema_model.module.state_dict(), model_save_path)
            save_msg = f" (Best: {best_acc:.2f}%)"
        
        print(f"Epoch {epoch:03d} | Loss: {train_losses[-1]:.4f} | Test Acc (EMA): {acc:.2f}%{save_msg}")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1); plt.plot(train_losses); plt.title("Loss (Mixup)")
    plt.subplot(1, 2, 2); plt.plot(test_accs); plt.title("Test Accuracy (EMA)")
    plt.savefig(plot_save_path)
    print(f"\nTraining Complete. Best Accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main()