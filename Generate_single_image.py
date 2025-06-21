# 模型基本交互测试
# 根据从控制台输入的E值，生成单张弹性图。

# 整体代码逻辑：
# [用户 & 已训练模型文件]
#          |
#          v
# [主互动流程 main()]
#   |__ 创建输出文件夹
#   |__ 检查并加载最佳生成器模型 (best_generator.pth)
#   |      |
#   |      v
#   |   [进入无限循环 while True]
#   |      |__ 提示用户在控制台输入E值
#   |      |__ 检查用户输入是否为 'exit'，若是则退出程序
#   |      |__ 验证输入是否为有效数字，若否则重新提示
#   |      |
#   |      v
#   |   [图像生成模块]
#   |      |__ 将用户输入的E值转换为Tensor
#   |      |__ 创建一个随机噪声向量
#   |      |__ 调用 generator_model(noise, e_value_tensor) 生成图像
#   |      |__ 后处理生成的图像 (范围转换, 维度调整)
#   |      |
#   |      v
#   |   [显示与保存模块]
#   |      |__ 使用 matplotlib 创建图像窗口
#   |      |__ 设置标题，显示生成的图像
#   |      |__ 构建基于E值的文件名，并保存图像文件
#   |      |__ 弹出图像窗口供用户实时查看
#   |      |
#   |      v
#   |   [返回循环起点，等待下一次输入]

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 配置参数
# !!重要!!: 确保这些参数与训练时使用的完全一致
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 100

# 文件路径
# !!重要!!: 确保路径正确
BEST_GENERATOR_MODEL_PATH = "./best_generator.pth" # 已训练的最佳生成器模型
OUTPUT_DIR = "./interactive_generated_images/"     # 保存单张生成图的文件夹

# 设备配置
DEVICE = torch.device("cuda")

# 定义生成器模型架构
# !!重要!!: 这里的模型架构必须与训练时使用的Generator完全相同
class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, e_value_embedding_dim=10):
        super(Generator, self).__init__()
        self.e_value_embedding = nn.Linear(1, e_value_embedding_dim)
        self.gen = nn.Sequential(
            self._block(z_dim + e_value_embedding_dim, IMAGE_SIZE * 8, 4, 1, 0),
            self._block(IMAGE_SIZE * 8, IMAGE_SIZE * 4, 4, 2, 1),
            self._block(IMAGE_SIZE * 4, IMAGE_SIZE * 2, 4, 2, 1),
            self._block(IMAGE_SIZE * 2, IMAGE_SIZE, 4, 2, 1),
            nn.ConvTranspose2d(IMAGE_SIZE, channels_img, 4, 2, 1),
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

# 主互动流程
def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建输出文件夹: {OUTPUT_DIR}")

    if not os.path.exists(BEST_GENERATOR_MODEL_PATH):
        print(f"错误：找不到模型文件 '{BEST_GENERATOR_MODEL_PATH}'。")
        print("请先运行训练脚本以生成 'best_generator.pth'。")
        return

    # 加载最佳模型
    print(f"正在从 '{BEST_GENERATOR_MODEL_PATH}' 加载模型...")
    generator_model = Generator(Z_DIM, CHANNELS_IMG).to(DEVICE)
    generator_model.load_state_dict(torch.load(BEST_GENERATOR_MODEL_PATH, map_location=DEVICE))
    generator_model.eval() # 设置为评估模式
    print("模型加载完毕，准备生成图像。")
    print("-" * 30)

    # 开始循环
    while True:
        # 提示用户输入
        user_input = input("请输入一个E值 (例如 5.7, 或输入 'exit' 退出): ")

        # 检查是否退出
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("程序结束。")
            break

        # 验证输入是否为数字
        try:
            e_value = float(user_input)
        except ValueError:
            print("输入无效，请确保输入的是一个数字。")
            continue

        # 生成图像
        print(f"正在为 E = {e_value} 生成图像...")
        
        e_value_tensor = torch.tensor([e_value], dtype=torch.float32, device=DEVICE)
        noise_vector = torch.randn(1, Z_DIM, 1, 1).to(DEVICE)
        
        with torch.no_grad(): 
            fake_image_tensor = generator_model(noise_vector, e_value_tensor)
            # 将图像从[-1, 1]范围转换回[0, 1]
            fake_image_numpy = (fake_image_tensor.cpu().numpy() + 1) / 2.0
            fake_image_numpy = np.squeeze(fake_image_numpy) # 去掉批次维度
            fake_image_numpy = np.transpose(fake_image_numpy, (1, 2, 0)) # 转换维度为 (H, W, C)

        # 保存图像 -> 显示
        plt.figure(figsize=(6, 6))
        plt.imshow(fake_image_numpy)
        plt.title(f"E={e_value:.2f}")
        plt.axis('off')
        file_name = f"generated_E_{str(e_value).replace('.', '_')}.png"
        output_filepath = os.path.join(OUTPUT_DIR, file_name)
        plt.savefig(output_filepath)
        print(f"图像已保存至: {output_filepath}")
        plt.show()


if __name__ == "__main__":
    main()