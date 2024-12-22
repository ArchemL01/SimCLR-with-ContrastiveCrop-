import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 设置数据转换操作
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 图像归一化
])

# 下载训练集
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# 创建训练集数据加载器
trainloader = DataLoader(trainset, batch_size=512, shuffle=True, num_workers=4)

# 下载测试集
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 创建测试集数据加载器
testloader = DataLoader(testset, batch_size=512, shuffle=False, num_workers=4)

print("CIFAR-10 数据集下载完成！")
# 下载训练集
trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

# 创建训练集数据加载器
trainloader = DataLoader(trainset, batch_size=512, shuffle=True, num_workers=4)

# 下载测试集
testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

# 创建测试集数据加载器
testloader = DataLoader(testset, batch_size=512, shuffle=False, num_workers=4)

print("CIFAR-100 数据集下载完成！")
