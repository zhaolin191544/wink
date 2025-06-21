# 基于弹性图的E值回归预测
# written by Lin
# 方法：直接回归 (图像 -> E值)
# 模型：ResNet18 (预训练)

# 整体代码逻辑：
# [JSON文件 & 图像文件]
#          |
#          v
# [RegressionDataset]
#   |__ __init__(json_path, root_dir, dataset_type, transform)
#   |__ __len__()
#   |__ __getitem__(idx) --> (image, e_value_tensor)
#          |
#          v
# [DataLoader] --> (images, labels) 批次数据
#          |
#          v
# [主执行流程]
#   |__ 定义图像变换 (由ResNet18预训练权重提供)
#   |__ 创建训练/验证/测试 DataLoader
#   |__ 初始化回归模型
#   |      |
#   |      v
#   |   [create_regression_model()]
#   |      |__ 加载预训练的 ResNet18
#   |      |__ 替换最后的全连接层，使其输出为1 (E值)
#   |      |
#   |      v
#   |__ 初始化损失函数 (MSELoss) & 优化器 (Adam)
#   |__ 循环: NUM_EPOCHS
#   |      |
#   |      v
#   |   [train_fn(model, train_loader, optimizer, loss_fn)] --> 训练一个周期的模型
#   |      |__ 遍历训练批次
#   |      |__ 模型预测: predictions = model(images)
#   |      |__ 计算损失: loss = loss_fn(predictions, labels)
#   |      |__ 反向传播 & 更新优化器
#   |      |
#   |      v
#   |   [evaluate_fn(model, val_loader)] --> 在验证集上评估
#   |      |__ 遍历验证批次
#   |      |__ 收集所有真实标签和模型预测值
#   |      |__ 计算整体的 MAE 和 RMSE
#   |      |__ 检查并保存最佳模型 (基于最低的 val_mae)
#   |      |
#   |      v
#   |   [训练结束]
#   |      |
#   |      v
#   |   [最终测试]
#   |      |__ 加载最佳回归模型 (best_regression_model.pth)
#   |      |__ 在测试集上调用 evaluate_fn
#   |      |__ 打印最终的 MAE 和 RMSE 评估结果

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from PIL import Image
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

#参数
DEVICE = torch.device("cuda")
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
NUM_EPOCHS = 50

# 数据路径
JSON_FILE_PATH = "./dataset_E.json"
IMAGE_ROOT_DIR = "./images/"

# 数据集加载器
class RegressionDataset(Dataset):
    def __init__(self, json_path, root_dir, dataset_type="train", transform=None):
        print(f"从{json_path}加载{dataset_type}数据...")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if dataset_type not in data:
            raise ValueError(f"错误：JSON文件中未找到 '{dataset_type}' 键。")

        self.image_data_list = list(data[dataset_type].items())
        self.root_dir = root_dir
        self.transform = transform
        print(f"从{dataset_type}集读取到{len(self.image_data_list)}张图片")

    def __len__(self):
        return len(self.image_data_list)

    def __getitem__(self, idx):
        img_relative_path, data_dict = self.image_data_list[idx]
        e_value = float(data_dict["E"])
        img_full_path = os.path.join(self.root_dir, img_relative_path)
        image = Image.open(img_full_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # 回归任务的标签是一个数值
        return image, torch.tensor(e_value, dtype=torch.float32)

# 模型定义
def create_regression_model():
    # 加载ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    
    return model.to(DEVICE)

# 训练与评估函数
def train_fn(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        # 模型预测
        predictions = model(images)
        
        # 计算损失
        loss = loss_fn(predictions.squeeze(1), labels)
        total_loss += loss.item()
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    avg_loss = total_loss / len(loader)
    print(f"训练平均损失: {avg_loss:.4f}")

def evaluate_fn(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            predictions = model(images)
            all_preds.extend(predictions.squeeze(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # 计算评估指标
    mae = mean_absolute_error(all_labels, all_preds)
    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    
    return mae, rmse

# 主执行
if __name__ == "__main__":
    print(f"使用设备: {DEVICE}")
    # Ref：👇
    # ResNet预训练模型要求的图像变换
    # 使用其推荐的均值和标准差
    weights = models.ResNet18_Weights.DEFAULT
    transform = weights.transforms()
    print("由预训练模型提供的图像变换流程:")
    print(transform)

    # 创建数据集和加载器
    train_dataset = RegressionDataset(JSON_FILE_PATH, IMAGE_ROOT_DIR, 'train', transform)
    val_dataset = RegressionDataset(JSON_FILE_PATH, IMAGE_ROOT_DIR, 'val', transform)
    test_dataset = RegressionDataset(JSON_FILE_PATH, IMAGE_ROOT_DIR, 'test', transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 初始化模型、损失函数和优化器
    model = create_regression_model()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # floop train
    best_val_mae = float('inf')
    best_epoch = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{NUM_EPOCHS} ---")
        train_fn(model, train_loader, optimizer, loss_fn)
        
        # 在val上评估
        val_mae, val_rmse = evaluate_fn(model, val_loader)
        print(f"--- val评估 (Epoch {epoch}) ---")
        print(f"MAE: {val_mae:.4f}")
        print(f"RMSE: {val_rmse:.4f}")

        # 保存最佳模型
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            torch.save(model.state_dict(), "best_regression_model.pth")
            print(f"最佳模型已保存 (Epoch {epoch})，val MAE: {best_val_mae:.4f}")

    print("\n--- 训练完成 ---")
    print(f"最佳模型出现在 Epoch {best_epoch}，其在val上的MAE为 {best_val_mae:.4f}")

    # 加载最佳模型
    # 在test上进行评估
    print("\n加载最佳模型进行最终评估...")
    best_model = create_regression_model()
    best_model.load_state_dict(torch.load("best_regression_model.pth"))
    
    test_mae, test_rmse = evaluate_fn(best_model, test_loader)
    print("\n最终测试集评估结果")
    print(f"MAE: {test_mae:.4f}")
    print(f"RMSE: {test_rmse:.4f}")