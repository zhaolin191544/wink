# 验证生成模型的有效性
# 流程：(真实E值) -> [生成器] -> (合成图像) -> [回归器] -> (预测E值)
#      然后比较 (真实E值) 与 (预测E值) 的差异

import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import matplotlib.pyplot as plt
# 导入模型定义

# 从 regression.py 复制过来
def create_regression_model():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    return model

# 从 cGAN 训练脚本复制过来
class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, e_value_embedding_dim=10):
        super(Generator, self).__init__()
        self.e_value_embedding = nn.Linear(1, e_value_embedding_dim)
        self.gen = nn.Sequential(
            self._block(z_dim + e_value_embedding_dim, 64 * 8, 4, 1, 0),
            self._block(64 * 8, 64 * 4, 4, 2, 1),
            self._block(64 * 4, 64 * 2, 4, 2, 1),
            self._block(64 * 2, 64, 4, 2, 1),
            nn.ConvTranspose2d(64, channels_img, 4, 2, 1),
            nn.Tanh(),
        )
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
    def forward(self, z, e_value):
        e_embedding = self.e_value_embedding(e_value.unsqueeze(1)).unsqueeze(2).unsqueeze(3)
        x = torch.cat([z, e_embedding], dim=1)
        return self.gen(x)

# 配置参数
DEVICE = torch.device("cuda")
Z_DIM = 100
IMAGE_SIZE = 64
CHANNELS_IMG = 3

# 文件路径
JSON_FILE_PATH = "./dataset_E.json"
GENERATOR_MODEL_PATH = "./best_generator.pth"
REGRESSOR_MODEL_PATH = "./best_regression_model.pth"
OUTPUT_DIR = "./validation_results/"

# 基本验证 确保使用显卡运行
def main():
    print(f"使用设备: {DEVICE}")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 加载训练好的模型
    print(f"加载生成器模型: {GENERATOR_MODEL_PATH}")
    generator = Generator(Z_DIM, CHANNELS_IMG).to(DEVICE)
    generator.load_state_dict(torch.load(GENERATOR_MODEL_PATH, map_location=DEVICE))
    generator.eval()

    print(f"加载回归器模型: {REGRESSOR_MODEL_PATH}")
    regressor = create_regression_model().to(DEVICE)
    regressor.load_state_dict(torch.load(REGRESSOR_MODEL_PATH, map_location=DEVICE))
    regressor.eval()
    
    # 准备数据和变换
    with open(JSON_FILE_PATH, 'r', encoding="utf-8") as f:
        data = json.load(f)
    test_data_list = list(data['test'].items())
    print(f"找到 {len(test_data_list)} 条测试数据，开始验证流程...")

    transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # 开始验证循环
    results_list = []

    with torch.no_grad():
        for i, (img_path, data_dict) in enumerate(test_data_list):
            original_e_value = float(data_dict['E'])
            e_tensor = torch.tensor([original_e_value], dtype=torch.float32, device=DEVICE)
            noise = torch.randn(1, Z_DIM, 1, 1).to(DEVICE)
            synthetic_image_tensor = generator(noise, e_tensor)
            
            synthetic_image_0_1 = (synthetic_image_tensor + 1) / 2.0
            image_for_regressor = transform(synthetic_image_0_1)

            predicted_e_tensor = regressor(image_for_regressor)
            predicted_e_value = predicted_e_tensor.item()
            
            # 计算误差并将所有信息存入列表
            error = abs(original_e_value - predicted_e_value)
            img_to_save = np.transpose(synthetic_image_0_1.squeeze(0).cpu().numpy(), (1, 2, 0))
            
            results_list.append({
                "original_e": original_e_value,
                "predicted_e": predicted_e_value,
                "error": error,
                "image": img_to_save
            })
            
            if (i + 1) % 50 == 0:
                print(f"已处理 {i + 1}/{len(test_data_list)} 个样本...")

    sorted_results = sorted(results_list, key=lambda x: x['error'])
    best_10_results = sorted_results[:10]
    
    print("挑选10个样本...")
    for i, result in enumerate(best_10_results):
        plt.figure(figsize=(6, 6))
        plt.imshow(result['image'])
        title = f"Sample #{i+1} | Error: {result['error']:.2f}\nOriginal E: {result['original_e']:.2f} -> Predicted E: {result['predicted_e']:.2f}"
        plt.title(title, fontsize=12)
        plt.axis('off')
        plt.savefig(f"{OUTPUT_DIR}/validation_sample_{i+1}.png")
        plt.close()

    print(f"10个样本的可视化结果已保存至 '{OUTPUT_DIR}' 文件夹。")

    

if __name__ == "__main__":
    main()