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
    """
    维护模型的滑动平均权重。
    EMA 模型通常比原始模型具有更好的泛化能力。
    """
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
                ema_v = esd[k]
                esd[k].copy_(ema_v * self.decay + model_v * (1. - self.decay))

# ----------------------------
# 辅助函数：Mixup 数据增强
# ----------------------------
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ----------------------------
# 辅助函数：固定随机种子
# ----------------------------
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True # 会牺牲一点点速度，但保证可复现
    torch.backends.cudnn.benchmark = False    # 注意：如果追求极致速度且输入尺寸固定，设为 True，但放弃完全确定性

def main():
    # ----------------------------
    # 0. 环境初始化
    # ----------------------------
    # 建议固定种子以复现结果，注释掉可获得随机性
    # seed_everything(42) 
    
    with open('config_optimized_mixup.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 如果不强制要求确定性，开启 benchmark 加速
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True 
        print("CuDNN benchmark enabled for speed.")

    # 提取参数
    batch_size = int(cfg['training']['batch_size'])
    epochs = int(cfg['training']['epochs'])
    learning_rate = float(cfg['training']['learning_rate'])
    weight_decay = float(cfg['training']['weight_decay'])

    data_root = cfg['data']['root']
    num_workers = int(cfg['data']['num_workers'])
    download_data = cfg['data']['download']
    num_classes = int(cfg['model']['num_classes'])
    classifier_dropout = float(cfg['model']['classifier_dropout'])
    
    # 获取路径
    model_save_path = cfg['saving']['model_path']
    plot_save_path = cfg['saving']['plot_path']

    # ----------------------------
    # 2. 数据加载 (增强策略保持)
    # ----------------------------
    normalize_mean = cfg['normalization']['mean']
    normalize_std = cfg['normalization']['std']

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3)), # 略微增加概率
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std)
    ])

    trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=download_data, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))

    testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=download_data, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))

    # ----------------------------
    # 3. 模型定义 (添加初始化)
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
        def __init__(self, num_classes=10, classifier_dropout=0.1):
            super(OptimizedCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(32)
            self.relu = nn.ReLU(inplace=True)

            # 稍微加深一点网络深度，如果 GPU 显存不够可改回
            self.layer1 = self._make_layer(32, 32, 3, stride=1) # num_blocks 2 -> 3
            self.layer2 = self._make_layer(32, 64, 3, stride=2) 
            self.layer3 = self._make_layer(64, 128, 3, stride=2)

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.dropout = nn.Dropout(classifier_dropout)
            self.fc = nn.Linear(128, num_classes)
            
            # Kaiming 初始化 (He Initialization) - 对 ReLU 网络至关重要
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
            x = self.fc(x)
            return x

    # ----------------------------
    # 4. 初始化与编译
    # ----------------------------
    model = OptimizedCNN(num_classes=num_classes, classifier_dropout=classifier_dropout).to(device)
    model = model.to(memory_format=torch.channels_last)

    # 真正启用 compile
    try:
        import torch._dynamo
        print("Compiling model with torch.compile (mode='reduce-overhead')...")
        # 'reduce-overhead' 适合小模型多次迭代，'max-autotune' 最慢但最快
        model = torch.compile(model, mode='reduce-overhead') 
    except Exception as e:
        print(f"Torch compile failed or not supported: {e}")

    # 初始化 EMA 模型 (不参与反向传播，只用于更新)
    ema_model = ModelEMA(model, decay=0.999)

    # 损失函数
    # 注意：使用 Mixup 时，我们在 loop 中手动计算 soft loss，
    # 这里定义的 criterion 用于非 Mixup 阶段或验证阶段
    base_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 学习率调度：Warmup + Cosine Annealing
    # 先在 5 个 epoch 内线性增加 LR，然后再进行 Cosine 衰减
    warmup_epochs = 5
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    scheduler_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs])

    # AMP
    use_amp = (device.type == 'cuda')
    scaler = amp_torch.GradScaler(device.type, enabled=use_amp)

    # ----------------------------
    # 5. 训练函数
    # ----------------------------
    def train_one_epoch(epoch_index):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(trainloader, desc=f'Epoch [{epoch_index}/{epochs}]', leave=False)

        for inputs, labels in loop:
            inputs = inputs.to(device, memory_format=torch.channels_last, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            # --- Mixup 逻辑 ---
            # 只有在训练且概率触发时使用 (这里设定每一步都用 mixup)
            use_mixup = True 
            if use_mixup:
                inputs_mixed, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=1.0, use_cuda=(device.type=='cuda'))
                
                with amp_torch.autocast(device.type, enabled=use_amp):
                    outputs = model(inputs_mixed)
                    loss = mixup_criterion(base_criterion, outputs, targets_a, targets_b, lam)
            else:
                with amp_torch.autocast(device.type, enabled=use_amp):
                    outputs = model(inputs)
                    loss = base_criterion(outputs, labels)

            scaler.scale(loss).backward()
            
            # --- 梯度裁剪 (防止梯度爆炸) ---
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            scaler.step(optimizer)
            scaler.update()

            # --- 更新 EMA ---
            ema_model.update(model)

            running_loss += loss.item()
            
            # 计算精度 (如果是 mixup，我们看原始标签的预测情况大概估算，或者跳过精度计算)
            # 为了展示方便，这里简单计算 argmax，但在 mixup 下 train acc 参考意义较小
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item() # 注意：Mixup 时这只是个近似参考

            loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

        return running_loss / len(trainloader), 100. * correct / total

    # ----------------------------
    # 6. 测试函数 (支持测试 EMA 模型)
    # ----------------------------
    def evaluate(target_model):
        target_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs = inputs.to(device, memory_format=torch.channels_last, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with amp_torch.autocast(device.type, enabled=use_amp):
                    outputs = target_model(inputs)

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        return 100. * correct / total

    # ----------------------------
    # 7. 主循环
    # ----------------------------
    train_losses = []
    test_accuracies = [] # 记录 EMA 的精度
    best_acc = 0.0

    print(f"\nStart Advanced Training on {device}...")
    
    try:
        for epoch in range(1, epochs + 1):
            train_loss, _ = train_one_epoch(epoch) # Mixup 下 train_acc 参考意义不大
            
            # 测试 EMA 模型的性能 (通常更稳定)
            # 如果想看当前模型性能，把 ema_model.module 换成 model
            test_acc = evaluate(ema_model.module)
            
            # Step scheduler
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            train_losses.append(train_loss)
            test_accuracies.append(test_acc)

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(ema_model.module.state_dict(), model_save_path) # 保存 EMA 权重
                save_msg = f"--> Saved Best EMA ({best_acc:.2f}%)"
            else:
                save_msg = ""

            print(f"Epoch {epoch:03d} | Loss: {train_loss:.4f} | Test Acc (EMA): {test_acc:.2f}% | LR: {current_lr:.5f} {save_msg}")

    except KeyboardInterrupt:
        print("\nInterrupted.")

    print(f"\nBest Test Accuracy (EMA): {best_acc:.2f}%")

    # ----------------------------
    # 8. 可视化
    # ----------------------------
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss (Mixup)')
    plt.title('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label='Test Acc (EMA)', color='orange')
    plt.title('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(plot_save_path)
    print(f"Plot saved to {plot_save_path}")

    # 保存最终模型
    torch.save(ema_model.module.state_dict(), model_save_path.replace('.pth', '_final_ema.pth'))

if __name__ == '__main__':
    main()