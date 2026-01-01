# Pytorch_CNN_example

## 项目概述
此项目使用Pytorch框架，自动下载cifar-10数据集。实现使用optimized_CNN神经卷积网络对**cifar-10**数据集的训练，目前使用仓库中的原参数可以达到测试集93%左右的识别率

## 项目目的
设计专为 CIFAR-10 设计的高效 ResNet 变体，使其在精度和性能之间达到一种平衡，适合在家用计算机上进行项目测试

## 项目特色
**使用外置yaml管理配置**
### 原生CNN+未处理数据集
原版CNN+原始数据集为pytorch_cifar10_cnn.py，对应配置为config.yaml
可以达到84%左右训练集识别率
### 变体CNN（OptimizedCNN）
使用OptimizedCNN 3x3卷积（自定义残差网络3x3卷积）并
使用**图像增强手段**：*随机裁剪、随机水平旋转、归一化、随机擦除*
使用**训练优化技术**：AMP自动混合精度，Channels Last（*针对NVIDIA Tensor*），Label Smoothing标签平滑，AdamW+余弦退火，cuDNN Benchmark自动调优。
### mixup技术
在后缀带有mixup的脚本中使用了竞赛级的mixup技术-混合种类训练，和EMA指数移动平均技术-集成过去步数的模型，使后期模型质量更稳定，此外优化Kaiming Initialization避免学习梯度消失。
同时在mixup版本中加宽了模型通道数，**故预测脚本不能通用**

## 本仓库OptimizedCNN结构
1. 模型架构 (Model Architecture)模型采用了类 ResNet 的轻量化设计，旨在平衡计算开销与特征提取能力：基础骨干: 由多个 ResidualBlock 堆叠而成，每个 Block 包含两个 $3 \times 3$ 卷积及跳跃连接（Skip Connection），有效缓解梯度消失问题。特征演进: 采用三级阶段设计，通道数从 32 提升至 128，并在阶段间通过 stride=2 进行下采样。分类器: * 全局自适应平均池化 (Global Adaptive Avg Pooling): 显著减少全连接层参数量。Dropout 层: 引入随机失活防止过拟合。线性输出: 最终映射至 10 个类别。

2. 核心技术特性 (Core Enhancements)为了提升准确率和训练效率，代码集成了以下优化手段：
	A. 图像增强 (Data Augmentation)基础组合: 随机裁剪 (Random Crop)、随机水平翻转 (Horizontal Flip)。正则化增强: 引入 随机擦除 (Random Erasing)，强制模型学习鲁棒的局部特征。数据标准化: 使用 CIFAR-10 全局均值与方差进行归一化。

	B. 训练性能优化混合精度训练 (AMP): 使用 torch.amp 自动混合精度，在 NVIDIA GPU 上提升约 2-3 倍训练速度并降低显存占用。内存格式优化 (Channels Last): 采用 NHWC 内存布局，充分发挥 Tensor Cores 的计算算力。DataLoader 加速: 开启 pin_memory 和 persistent_workers，最小化 CPU 与 GPU 之间的数据传输瓶颈。

	C. 算法改进标签平滑 (Label Smoothing): 在交叉熵损失中加入 $0.1$ 的平滑因子，提高泛化能力。权重衰减 (AdamW): 使用 AdamW 优化器，比传统 Adam 具有更好的权重衰减效果。学习率调度: 采用 余弦退火算法 (Cosine Annealing LR)，使模型在后期能更平稳地收敛到局部最优。
	D. 运行监控代码集成 tqdm 进度条，实时显示每个 Epoch 的：
	Loss: 训练损失
	Acc: 实时训练准确率
	LR: 当前动态学习率

## Optimized项目结构
```php
├── cifar-10-batches-py/           # cifar-10训练集路径
├── config_optimized.yaml          # optimized配置文件（包含超参数、路径、增强设置）
├── config_optimized_mixup.yaml    # mixup配置文件（包含超参数、路径、增强设置）
├── cifar10_cnn_optimized.py       # optimized核心训练脚本
├── cifar10_cnn_optimized_mixup.py # mixup技术的核心训练脚本
├── {best_model}.pth               # 验证集表现最好的权重，可在对应yaml中修改文件名
└── {training_plot}.png            # 损失与准确率演进图，可在对应yaml中修改文件名
```
