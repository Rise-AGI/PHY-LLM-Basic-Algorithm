import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import base64
import json
import urllib.request
import os
from urllib.error import HTTPError

# 从环境变量读取Token
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()

REPO = "Rise-AGI/PHY-LLM-Basic-Algorithm"
FILE_PATH = "magnus_code/zyz/mnist_light_model.pth"
BRANCH = "main"
COMMIT_MSG = "auto upload trained model"

def get_github_file_sha(url, token):
    """获取已存在文件的sha，用于覆盖上传"""
    try:
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"token {token}")
        with urllib.request.urlopen(req) as f:
            data = json.loads(f.read().decode())
            return data.get("sha")
    except:
        return None

def upload_file_to_github(file_path):
    if not GITHUB_TOKEN:
        print("⚠️ 未检测到GitHub Token，跳过上传")
        return
    
    try:
        with open(file_path, "rb") as f:
            content = base64.b64encode(f.read()).decode("utf-8")
        
        url = f"https://api.github.com/repos/{REPO}/contents/{FILE_PATH}"
        data = {
            "message": COMMIT_MSG,
            "content": content,
            "branch": BRANCH
        }
        
        # 获取sha（如果文件已存在）
        sha = get_github_file_sha(url, GITHUB_TOKEN)
        if sha:
            data["sha"] = sha
        
        data = json.dumps(data).encode("utf-8")
        
        req = urllib.request.Request(url, data=data, method="PUT")
        req.add_header("Authorization", f"token {GITHUB_TOKEN}")
        req.add_header("Content-Type", "application/json")
        req.add_header("User-Agent", "Python-Script")
        
        with urllib.request.urlopen(req) as f:
            print(f"✅ 模型上传GitHub成功！")
    except HTTPError as e:
        print(f"❌ 上传失败: {e.code} {e.reason}")
        print(f"错误信息: {e.read().decode()}")
    except Exception as e:
        print(f"❌ 错误: {str(e)}")

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

batch_size = 32
learning_rate = 0.001
epochs = 3
num_classes = 10
data_dir = "./data"
model_save_path = "./mnist_light_model.pth"

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载数据集
print("正在加载MNIST官方数据集...")
train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

print(f"数据集加载完成：训练集 {len(train_dataset)} 张，测试集 {len(test_dataset)} 张")

# 轻量CNN模型
class LightWeightCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
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

model = LightWeightCNN(num_classes=num_classes).to(device)
print(f"模型初始化完成，总参数量：{sum(p.numel() for p in model.parameters())}")

# 训练配置
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练函数
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
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    avg_loss = total_loss / len(loader)
    acc = 100.0 * correct / total
    return avg_loss, acc

# 测试函数
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
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    avg_loss = total_loss / len(loader)
    acc = 100.0 * correct / total
    return avg_loss, acc

# 开始训练
print(f"开始训练，共 {epochs} 轮...")
for epoch in range(1, epochs + 1):
    print(f"\n===== 轮次 {epoch}/{epochs} =====")
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = test_model(model, test_loader, criterion, device)
    print(f"训练 | 损失: {train_loss:.4f} | 准确率: {train_acc:.2f}%")
    print(f"测试 | 损失: {test_loss:.4f} | 准确率: {test_acc:.2f}%")

# 保存模型
torch.save(model.state_dict(), model_save_path)
print(f"\n训练完成！模型已保存至: {model_save_path}")
print(f"最终测试集准确率: {test_acc:.2f}%")

# 上传模型
print("\n正在上传模型到 GitHub...")
upload_file_to_github(model_save_path)