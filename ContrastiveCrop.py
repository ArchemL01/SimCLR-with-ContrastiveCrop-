import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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
def contrastive_crop(images, features, s, r, alpha=0.5, k=0.2, min_ratio=0.5, max_ratio=2.0):
    batch_size = images.size(0)  # 获取批量大小
    cropped_images = []

    for i in range(batch_size):
        # 获取语义感知定位的包围框
        B = semantic_aware_localization(features[i], k)

        # 使用中心抑制采样生成裁剪区域
        x, y, h, w = center_suppressed_crop(s, r, B, alpha)

        # 将坐标转换为整数，确保能被用于PIL的crop方法
        x, y, h, w = int(x), int(y), int(h), int(w)
        # 确保裁剪区域不超出图像边界
        x0 = max(x - w // 2, 0)
        y0 = max(y - h // 2, 0)
        x1 = min(x + w // 2, images.size(2))  # 图像的宽度
        y1 = min(y + h // 2, images.size(3))  # 图像的高度

        # 对图像进行裁剪
        cropped_image = images[i, :, y0:y1, x0:x1]
        cropped_images.append(cropped_image)

    # 将裁剪后的图像堆叠成一个批次
    cropped_images = torch.stack(cropped_images, dim=0)

    return cropped_images
