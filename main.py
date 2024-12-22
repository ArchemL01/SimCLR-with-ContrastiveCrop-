import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
import time
import matplotlib.pyplot as plt
from torchvision.transforms import ColorJitter, GaussianBlur

from ContrastiveCrop import contrastive_crop

# 设置临时文件目录为 E 盘
os.environ['TMPDIR'] = 'E:/tempp'
os.environ['TEMP'] = 'E:/tempp'
os.environ['TMP'] = 'E:/tempp'
# 数据增强
def get_default_aug(image_size, channels):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406] if channels == 3 else [0.5],
                             std=[0.229, 0.224, 0.225] if channels == 3 else [0.5])
    ])

# 辅助函数
def default(value, default):
    return value if value is not None else default

def get_module_device(module):
    return next(module.parameters()).device

def flatten(tensor):
    return tensor.view(tensor.size(0), -1)

# 绘图函数
def plot_metrics(metrics, title, xlabel='Epochs', ylabel='Value'):
    plt.figure(figsize=(10, 5))
    for key, values in metrics.items():
        plt.plot(range(1, len(values) + 1), values, label=key)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.show()
# 特征提取器包装器
class NetWrapper(nn.Module):
    def __init__(self, net, project_dim, layer=-2):
        super().__init__()
        self.net = net
        self.projector = nn.Sequential(
            nn.Linear(512, 512),  # 对于 ResNet-18 的特征维度是 512
            nn.ReLU(),
            nn.Linear(512, project_dim)
        )
        self.hidden_layer = layer

    def forward(self, x):
        # 提取网络的中间特征
        feats = self.net(x)
        return self.projector(feats), feats

class SimCLR(nn.Module):
    def __init__(
        self,
        net,
        image_size,
        channels=3,
        hidden_layer=-2,
        project_hidden=True,
        project_dim=128,
        augment_both=True,
        use_nt_xent_loss=True,
        augment_fn=None,
        temperature=0.1,
        use_contrastive_crop=False,  # 新增 use_contrastive_crop 参数
        crop_params=None  # 用于裁剪的额外参数
    ):
        super().__init__()
        self.net = NetWrapper(net, project_dim, layer=hidden_layer)
        self.augment = default(augment_fn, get_default_aug(image_size, channels))
        self.augment_both = augment_both
        self.temperature = temperature
        self.use_contrastive_crop = use_contrastive_crop  # 赋值给实例
        self.crop_params = crop_params if crop_params is not None else {}  # 默认参数为空字典

        # 将模型移动到设备
        device = get_module_device(net)
        self.to(device)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise ValueError("Input to the SimCLR model must be a tensor.")

        if self.use_contrastive_crop:
            # 生成增强后的图像
            augmented_1 = torch.cat([self.augment(img).unsqueeze(0) for img in x], dim=0)
            augmented_2 = torch.cat([self.augment(img).unsqueeze(0) for img in x], dim=0)

            # 获取网络特征（中间层输出）
            with torch.no_grad():
                _, features_1 = self.net(augmented_1)  # 获取特征
                _, features_2 = self.net(augmented_2)  # 获取特征

            # 使用裁剪
            augmented_1 = contrastive_crop(augmented_1, features_1, **self.crop_params)
            augmented_2 = contrastive_crop(augmented_2, features_2, **self.crop_params)
        else:
            if self.augment_both:
                augmented_1 = torch.cat([self.augment(img).unsqueeze(0) for img in x], dim=0)
                augmented_2 = torch.cat([self.augment(img).unsqueeze(0) for img in x], dim=0)
            else:
                augmented_1 = x
                augmented_2 = torch.cat([self.augment(img).unsqueeze(0) for img in x], dim=0)

        queries, _ = self.net(augmented_1)  # 第一种增强视图
        keys, _ = self.net(augmented_2)  # 第二种增强视图

        queries, keys = map(flatten, (queries, keys))
        loss = nt_xent_loss(queries, keys, temperature=self.temperature)  # 计算对比损失
        return loss


# NT-Xent 损失函数
def nt_xent_loss(queries, keys, temperature=0.1):
    b, device = queries.shape[0], queries.device

    # 合并 queries 和 keys，构建对比样本
    projs = torch.cat((queries, keys), dim=0)  # projs: [2b, dim]
    logits = projs @ projs.T  # logits: [2b, 2b]

    # 设置对角线为无穷小，避免与自己比较
    mask = torch.eye(logits.shape[0], device=device).bool()
    logits[mask] = float('-inf')

    # 归一化温度
    logits /= temperature

    # 构建正确的标签，正样本是 queries 和 keys 成对
    labels = torch.cat([torch.arange(b, device=device) + b, torch.arange(b, device=device)], dim=0)

    # 损失计算：选取对应的正样本
    loss = F.cross_entropy(logits, labels, reduction='sum') / logits.size(0)
    return loss

# 训练和测试函数
def train(model, trainloader, optimizer, epochs, device):
    model.train()
    metrics = {'Loss': []}  # 用于记录指标
    for epoch in range(epochs):
        epoch_loss = 0
        start_time = time.time()

        for images, _ in trainloader:
            images = images.to(device)
            optimizer.zero_grad()
            loss = model(images)  # 直接传递 tensor
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(trainloader)
        metrics['Loss'].append(avg_loss)
        print(f"Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.4f} Time: {time.time() - start_time:.2f}s")

    # 绘制训练曲线
    plot_metrics(metrics, title="Training Loss Curve")


def test(model, testloader, device):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            _, feats = model.net(images)
            preds = torch.argmax(feats, dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

def finetune_classifier(pretrained_model, trainloader, testloader, device, finetune_epochs=50, learning_rate=1e-3):
    """
    微调预训练模型以进行分类任务
    Args:
        pretrained_model: 预训练的 SimCLR 模型
        trainloader: CIFAR-10 训练集 DataLoader
        testloader: CIFAR-10 测试集 DataLoader
        device: 使用的设备
        finetune_epochs: 微调的 epoch 数
        learning_rate: 微调的学习率
    """

    # 加载预训练模型
    pretrained_model.load_state_dict(torch.load('simclr_cc.pth'))
    # 冻结特征提取层
    for param in pretrained_model.net.parameters():
        param.requires_grad = False

    # 动态获取特征维度
    sample_input = torch.randn(1, 3, image_size, image_size).to(device)  # 创建与数据集一致的输入大小
    with torch.no_grad():
        _, features = pretrained_model.net(sample_input)
    feature_dim = features.shape[1]  # 获取特征的维度

    # 根据特征维度定义分类头
    classifier = nn.Sequential(
        nn.Linear(feature_dim, 10)
    ).to(device)

    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    metrics = {'Train Loss': [], 'Train Accuracy': []}
    # 开始微调
    for epoch in range(finetune_epochs):
        pretrained_model.eval()  # 冻结部分仍然是 eval 模式
        classifier.train()  # 分类头进入训练模式

        epoch_loss = 0
        correct, total = 0, 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            # 获取冻结特征
            with torch.no_grad():
                _, features = pretrained_model.net(images)

            # 分类训练
            optimizer.zero_grad()
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 统计训练损失和准确率
            epoch_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        avg_loss = epoch_loss / len(trainloader)
        train_acc = 100 * correct / total
        metrics['Train Loss'].append(avg_loss)
        metrics['Train Accuracy'].append(train_acc)
        print(f"Epoch [{epoch + 1}/{finetune_epochs}], Loss: {avg_loss:.4f}, Accuracy: {train_acc:.2f}%")

    # 测试分类头性能
    classifier.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            _, features = pretrained_model.net(images)
            outputs = classifier(features)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    test_acc = 100 * correct / total
    print(f"Test Accuracy after fine-tuning: {test_acc:.2f}%")
    # 绘制微调曲线
    plot_metrics(metrics, title="Fine-tuning Loss and Accuracy")

# 主程序
if __name__ == '__main__':
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载 CIFAR-10 数据集
    image_size = 32
    batch_size = 512
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 定义模型和优化器
    base_model = torchvision.models.resnet18(pretrained=False)
    base_model.fc = nn.Identity()  # 去掉最后一层全连接层
    model = SimCLR(base_model, image_size=image_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # 开始训练和测试
    epochs = 200
    train(model, trainloader, optimizer, epochs, device)
    # 保存预训练模型
    torch.save(model.state_dict(), 'simclr_cc.pth')
    # 加载预训练模型
    model.load_state_dict(torch.load('simclr_cc.pth'))
    test(model, testloader, device)
    # 在预训练完成后调用微调函数
    finetune_classifier(model, trainloader, testloader, device, finetune_epochs=50)
