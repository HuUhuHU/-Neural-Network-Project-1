# Project 1: MNIST 手写数字分类

## 项目结构

```
PJ1/
├── README.md                 # 本文件
├── project_1.pdf             # 课程作业说明
└── codes/
    ├── project1.ipynb        # 主实验 Notebook（训练、评估、可视化）
    ├── download_mnist.py     # MNIST 数据下载脚本
    ├── test_train.py         # 命令行训练脚本（MLP 示例）
    ├── test_model.py         # 命令行测试脚本
    └── mynn/                 # 手写神经网络框架
        ├── op.py             # Linear, conv2D, CrossEntropy 等算子
        ├── models.py         # Model_MLP, Model_CNN
        ├── optimizer.py      # SGD, MomentGD
        ├── lr_scheduler.py     # MultiStepLR 等
        └── runner.py           # 训练循环
```

## 环境要求

- Python 3.8+
- NumPy
- Matplotlib
- tqdm

## 已实现内容

| 模块 | 文件 | 说明 |
|------|------|------|
| 线性层 | `mynn/op.py` | `Linear.forward` / `backward` |
| 损失函数 | `mynn/op.py` | `MultiCrossEntropyLoss`（含 Softmax） |
| 卷积层 | `mynn/op.py` | `conv2D`（im2col 实现） |
| MLP 模型 | `mynn/models.py` | `Model_MLP` 基线 |
| CNN 模型 | `mynn/models.py` | `Model_CNN`（3 层卷积 + 2 层全连接） |
| 优化器 | `mynn/optimizer.py` | `SGD`, `MomentGD`（动量） |
| 学习率调度 | `mynn/lr_scheduler.py` | `MultiStepLR` |

## 实验设计

- **Part A**: MLP `784→600→10`，SGD + MultiStepLR，完整 MNIST 训练集训练 5 epoch
- **Part B**: 自实现 CNN，与 MLP 在相同超参数下对比
- **Part C**:
  - 方向 1（优化）: SGD vs Momentum (μ=0.9)
  - 方向 5（误差分析）: 混淆矩阵、错分样本、权重和卷积核可视化
