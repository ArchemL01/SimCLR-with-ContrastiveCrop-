# ContrastiveCrop: 面向对比学习的语义感知裁剪方法

基于论文simple framework for contrastive learning of visual representations{https://arxiv.org/abs/2202.03278} 中的相关内容，
本项目实现了 **ContrastiveCrop** 方法，该方法改进了对比学习中的裁剪策略，使其具有更好的语义感知能力。通过在 CIFAR-10 和 CIFAR-100 数据集上的实验，展示了该方法的有效性，并与 SimCLR 基线模型进行了对比。

---

## 目录
1. [简介](#简介)
2. [文件结构](#文件结构)
3. [安装方法](#安装方法)
4. [使用说明](#使用说明)
5. [实验设计与结果](#实验设计与结果)
6. [致谢](#致谢)

---

## 简介

ContrastiveCrop 是一种用于对比学习的增强方法。通过热力图生成语义感知的裁剪区域，该方法能够减少随机裁剪可能导致的假阳性样本，提升对比学习模型的特征表示能力。本方法基于 SimCLR 框架开发，但可以应用于任何自监督学习任务。

---

## 文件结构

| 文件名                | 描述                                              |
|-----------------------|---------------------------------------------------|
| `ContrastiveCrop.py`  | ContrastiveCrop 方法的核心实现，包括语义感知裁剪逻辑。 |
| `simCLR.py`           | 基线模型 SimCLR 的实现代码，用于对比实验。          |
| `main.py`             | 使用 ContrastiveCrop 进行模型训练与评估的主程序。  |
| `cc_test.py`          | ContrastiveCrop 剪切效果测试代码。                 |
| `dataset.py`          | 数据集加载与预处理代码，支持 CIFAR-10 和 CIFAR-100。 |

---

## 安装方法

### 克隆代码库：
   git clone https://github.com/ArchemL01/SimCLR-with-ContrastiveCrop-
   cd contrastivecrop

### 环境要求
- Python >= 3.8
- PyTorch >= 1.11
- torchvision >= 0.12
- ...

## 使用说明

### 剪切测试 
python cc_test.py --dataset CIFAR-100

### 预训练模型
python main.py --dataset CIFAR-100 --epochs 200 --batch_size 512

### 线性评估
python main.py --dataset CIFAR-100 --eval --linear --epochs 50

## 实验设计与结果
与传统的simCLR相比，整合了**ContrastiveCrop**方法的模型在CIFAR-100/10数据集上的预测效果均有所提升，具体表现在约3%的accuracy提升。

## 致谢
本项目基于 ResNet-18 和 SimCLR 框架，并根据Xiangyu Peng, Kai Wang等人的论文simple framework for contrastive learn-
ing of visual representations扩展了 ContrastiveCrop 方法，提升了其对比学习效果。
