import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random

# 加载预训练的ResNet-18模型
model = models.resnet18(pretrained=True)
model.eval()

# 替换ResNet的最后一层，使其适应特征提取
model.fc = nn.Identity()  # 移除分类层，仅保留特征提取部分


# 图像预处理
transform = transforms.Compose([
    transforms.CenterCrop(224),  # 只裁剪中心区域，不做缩放
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# 获取卷积层特征的钩子函数
features = []

def hook_fn(module, input, output):
    # 记录该层的输出特征
    features.append(output)

# 注册钩子到ResNet的某一卷积层（例如，最后一个卷积层）
hook = model.layer4[1].register_forward_hook(hook_fn)


# Beta分布采样函数
def beta_distribution_sample(alpha, size=1):
    return torch.distributions.Beta(alpha, alpha).sample([size])


# 计算裁剪区域
def center_suppressed_crop(s, r, B, alpha=0.5):
    # 从包围框 B 中获取坐标
    B_x0, B_y0, B_x1, B_y1 = B

    # 裁剪的高度和宽度
    h = np.sqrt(s * r)
    w = np.sqrt(s / r)

    # 使用Beta分布从 B 中采样 x 和 y
    u = beta_distribution_sample(alpha)
    v = beta_distribution_sample(alpha)

    # 计算裁剪区域的中心 (x, y)
    x = B_x0 + (B_x1 - B_x0) * u
    y = B_y0 + (B_y1 - B_y0) * v

    return x, y, h, w


# 可视化裁剪区域
def visualize_crop(image, x, y, h, w):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    rect = plt.Rectangle((x - w / 2, y - h / 2), w, h, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show()


# 定义语义感知定位 (Semantic-aware Localization)
def semantic_aware_localization(features, k):
    # 归一化特征以生成热图
    M = F.relu(torch.sum(features, dim=1))  # 对通道维度求和得到热图
    M = M / M.max()  # 归一化到 [0, 1]

    # 使用阈值 k 进行热图的二值化
    mask = M > k  # 生成二值掩码，阈值化处理

    # 获取非零的坐标（去掉批次维度）
    nonzero_indices = mask[0].nonzero(as_tuple=False)  # 只取 7x7 的非零坐标

    if nonzero_indices.size(0) == 0:
        # 如果没有满足条件的点，返回一个默认的包围框（全图）
        return 0, 0, M.shape[2], M.shape[1]

    # 获取最小和最大的坐标
    B_y0, B_x0 = torch.min(nonzero_indices, dim=0)[0]  # 最小坐标 (y, x)
    B_y1, B_x1 = torch.max(nonzero_indices, dim=0)[0]  # 最大坐标 (y, x)

    return B_x0.item(), B_y0.item(), B_x1.item(), B_y1.item()



# 定义主函数来实现 ContrastiveCrop
def contrastive_crop(image, features, s, r, alpha=0.5, k=0.2, min_ratio=0.5, max_ratio=2.0):
    # 获取语义感知定位的包围框
    B = semantic_aware_localization(features, k)

    # 使用中心抑制采样生成裁剪区域
    x, y, h, w = center_suppressed_crop(s, r, B, alpha)
    # 限制裁剪的宽高比
    # r = np.clip(r, min_ratio, max_ratio)
    # h = max(h, 32)  # 确保最小高度
    # w = max(w, 32)  # 确保最小宽度

    # 返回裁剪区域 (x, y, h, w)
    return x, y, h, w


# 创建一个CIFAR-10数据集的自定义数据集类
class ContrastiveCIFAR10(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        # 使用torchvision的CIFAR-10数据集
        self.cifar10 = datasets.CIFAR10(root=root, train=train, download=True)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.cifar10)

    def __getitem__(self, idx):
        image, label = self.cifar10[idx]

        # 预处理图像
        img_tensor = self.transform(image).unsqueeze(0)

        # 清空特征列表
        features.clear()
        # 提取特征
        with torch.no_grad():
            model(img_tensor)

        # 获取裁剪区域
        x, y, h, w = contrastive_crop(image, features[0], s=512, r=1, alpha=0.5, k=0.2)

        # 将坐标转换为整数，确保能被用于PIL的crop方法
        x, y, h, w = int(x), int(y), int(h), int(w)
        # 确保裁剪区域不超出图像边界
        x0 = max(x - w // 2, 0)
        y0 = max(y - h // 2, 0)
        x1 = min(x + w // 2, image.width)
        y1 = min(y + h // 2, image.height)

        # 使用裁剪区域对图像进行裁剪
        crop_image = image.crop((x0, y0, x1, y1))

        if self.target_transform:
            label = self.target_transform(label)
        # 将裁剪后的图像转换为Tensor
        crop_image_tensor = transform(crop_image)
        return crop_image_tensor, label


# 使用自定义数据集和DataLoader
transform = transforms.Compose([
    transforms.Resize(256),  # 调整图像大小
    transforms.CenterCrop(224),  # 中心裁剪到224x224
    transforms.ToTensor(),  # 转换为Tensor
])

# CIFAR-10数据加载器
train_dataset = ContrastiveCIFAR10(root='data', train=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 查看数据集中的一个batch
data_iter = iter(train_loader)
images, labels = next(data_iter)

# 可视化一个batch
grid_img = torchvision.utils.make_grid(images)
plt.imshow(grid_img.permute(1, 2, 0))
plt.show()
