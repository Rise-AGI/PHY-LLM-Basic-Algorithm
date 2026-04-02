import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------------------- 1. 基础配置（内存优化版） ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 超参数（极致控制内存+缩短运行时长）
batch_size = 32       # 小批次降低内存峰值
learning_rate = 0.001
epochs = 3            # 3轮即可达到98%+准确率
num_classes = 10

# 路径配置
data_dir = "./data"
model_save_path = "./mnist_light_model.pth"

# ---------------------- 2. 数据准备（官方源自动下载，总占用<60M） ----------------------
# 极简预处理，无冗余内存开销
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 自动从多伦多大学官方服务器下载MNIST数据集（压缩包11M，解压后55M）
print("正在加载MNIST官方数据集...")
train_dataset = datasets.MNIST(
    root=data_dir, train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root=data_dir, train=False, download=True, transform=transform
)

# 数据加载器：限制多进程数，减少内存开销
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
print(f"数据集加载完成：训练集 {len(train_dataset)} 张，测试集 {len(test_dataset)} 张")

# ---------------------- 3. 超轻量CNN模型（参数量<10万，权重文件<500KB） ----------------------
class LightWeightCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 极简卷积层，极致压缩参数量和内存占用
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 轻量分类头
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 初始化模型
model = LightWeightCNN(num_classes=num_classes).to(device)
print(f"模型初始化完成，总参数量：{sum(p.numel() for p in model.parameters())}")

# ---------------------- 4. 损失函数与优化器 ----------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ---------------------- 5. 训练与测试函数 ----------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, pred = torch.max(output, 1)
        total += target.size(0)
        correct += (pred == target).sum().item()

    avg_loss = total_loss / len(loader)
    acc = 100 * correct / total
    return avg_loss, acc

def test_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            _, pred = torch.max(output, 1)
            total += target.size(0)
            correct += (pred == target).sum().item()

    avg_loss = total_loss / len(loader)
    acc = 100 * correct / total
    return avg_loss, acc

# ---------------------- 6. 主训练循环 ----------------------
print(f"开始训练，共 {epochs} 轮...")
for epoch in range(epochs):
    print(f"\n===== 轮次 {epoch+1}/{epochs} =====")
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    print(f"训练 | 损失: {train_loss:.4f} | 准确率: {train_acc:.2f}%")
    
    test_loss, test_acc = test_model(model, test_loader, criterion, device)
    print(f"测试 | 损失: {test_loss:.4f} | 准确率: {test_acc:.2f}%")

# 保存最终模型
torch.save(model.state_dict(), model_save_path)
print(f"\n训练完成！模型已保存至: {model_save_path}")
print(f"最终测试集准确率: {test_acc:.2f}%")
print("程序执行完毕！")