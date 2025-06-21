# -*- coding: utf-8 -*-
# 基于E值的弹性图生成任务
# torch cGAN
# written by: Lin

# train：每个批次训练过程中 学习训练 直接更新模型权重
# val: 每个批次训练结束以后 不更新权重 仅用于选择最佳魔心
# test: 所有训练和模型选择结束以后 不更新权重 用于最终性能评估

# 整体代码逻辑：
# [JSON文件 & 图像文件]
#          |
#          v
# [DataSetLoader]
#   |__ __init__(json_path, root_dir, dataset_type, transform)
#   |__ __len__()
#   |__ __getitem__(idx) --> (image, E_value_tensor)
#          |
#          v
# [DataLoader] --> (real_img, e_val) 批次数据
#          |
#          v
# [主执行流程]
#   |__ 定义图像变换 (Resize, ToTensor, Normalize)
#   |__ 创建训练/验证/测试 DataLoader
#   |__ 初始化生成器 & 判别器
#   |      |
#   |      v
#   |   [initialize_weights(model)] --> 初始化模型权重
#   |      |
#   |      v
#   |__ 初始化优化器 (Adam) & 损失函数 (BCEWithLogitsLoss, L1Loss)
#   |__ 循环: NUM_EPOCHS
#   |      |
#   |      v
#   |   [train_fn(disc, gen, train_loader, ...)] --> 训练一个周期的模型
#   |      |__ 判别器训练:
#   |      |      |__ 生成伪造图像: [Generator(noise, e_val)]
#   |      |      |__ disc(real_img, e_val) --> 真实图像损失
#   |      |      |__ disc(fake_img, e_val) --> 伪造图像损失
#   |      |      |__ loss_disc = (真实损失 + 伪造损失) / 2
#   |      |      |__ 反向传播 & 更新判别器
#   |      |__ 生成器训练:
#   |      |      |__ disc(fake_img, e_val) --> 对抗损失
#   |      |      |__ L1 loss(fake_img, real_img) * L1_LAMBDA
#   |      |      |__ loss_gen = 对抗损失 + L1损失
#   |      |      |__ 反向传播 & 更新生成器
#   |      |__ 每10个批次打印一次损失
#   |      |
#   |      v
#   |   [evaluate_fn(gen, val_loader, ...)] --> 在验证集上评估
#   |      |__ 计算验证集上的 PSNR & SSIM
#   |      |__ 根据验证集表现保存最佳模型
#   |      |
#   |      v
#   |   [最终测试与图像保存]
#   |      |__ 加载最佳模型
#   |      |__ 在测试集上评估并保存对比图
#   |      |
#   |      v
#   |   [保存最终模型]
#   |      |__ 保存 final_generator.pth

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# 超参数配置 && 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4  # 学习率
BETA1 = 0.5           # Adam优化器的beta1参数
BETA2 = 0.999         # Adam优化器的beta2参数
BATCH_SIZE = 128      # 批处理大小
IMAGE_SIZE = 64       # 图像将被调整到的大小 (64x64)
CHANNELS_IMG = 3      # 图像通道数 (RGB)
Z_DIM = 100           # 噪声向量的维度
NUM_EPOCHS = 100      # 训练周期数
L1_LAMBDA = 100       # L1损失的权重

# 数据路径
JSON_FILE_PATH = "./dataset_E.json"
IMAGE_ROOT_DIR = "./images/"


# 弹性图数据集加载器
class DataSetLoader(Dataset):
    def __init__(self, json_path, root_dir, dataset_type="train", transform=None):
        print(f"从{json_path}加载{dataset_type}数据...")
        with open(json_path, "r") as f:
            data = json.load(f)
        
        # 增加对val数据集的健壮性判断
        if dataset_type not in data:
            raise ValueError(f"错误：JSON文件中未找到 '{dataset_type}' 键。请检查JSON文件结构。")

        self.image_data_list = list(data[dataset_type].items())
        self.root_dir = root_dir
        self.transform = transform
        print(f"从{dataset_type}集读取到{len(self.image_data_list)}图片")

    def __len__(self):
        return len(self.image_data_list)

    def __getitem__(self, idx):
        img_relative_path, data_dict = self.image_data_list[idx]
        E_value = float(data_dict["E"])
        img_full_path = os.path.join(self.root_dir, img_relative_path)
        image = Image.open(img_full_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        E_value_tensor = torch.tensor(E_value, dtype=torch.float32)

        return image, E_value_tensor


# 生成器模型
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

# 判别器模型
class Discriminator(nn.Module):
    def __init__(self, channels_img, e_value_embedding_dim=10):
        super(Discriminator, self).__init__()
        self.e_value_embedding = nn.Linear(1, IMAGE_SIZE * IMAGE_SIZE)
        self.disc = nn.Sequential(
            nn.Conv2d(channels_img + 1, IMAGE_SIZE, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            self._block(IMAGE_SIZE, IMAGE_SIZE * 2, 4, 2, 1),
            self._block(IMAGE_SIZE * 2, IMAGE_SIZE * 4, 4, 2, 1),
            self._block(IMAGE_SIZE * 4, IMAGE_SIZE * 8, 4, 2, 1),
            nn.Conv2d(IMAGE_SIZE * 8, 1, 4, 1, 0),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x, e_value):
        e_embedding = self.e_value_embedding(e_value.unsqueeze(1)).view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
        x = torch.cat([x, e_embedding], dim=1)
        return self.disc(x)

# 初始化权重
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

# 训练函数
def train_fn(disc, gen, loader, opt_disc, opt_gen, bce_loss, l1_loss):
    disc.train()
    gen.train()
    for batch_idx, (real_img, e_val) in enumerate(loader):
        real_img = real_img.to(DEVICE)
        e_val = e_val.to(DEVICE)

        # 训练判别器
        noise = torch.randn(real_img.size(0), Z_DIM, 1, 1).to(DEVICE) # 动态适应最后一个batch的大小
        fake_img = gen(noise, e_val)
        disc_real = disc(real_img, e_val).reshape(-1)
        loss_disc_real = bce_loss(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake_img.detach(), e_val).reshape(-1)
        loss_disc_fake = bce_loss(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # 训练生成器
        output = disc(fake_img, e_val).reshape(-1)
        loss_gen_adv = bce_loss(output, torch.ones_like(output))
        loss_gen_l1 = l1_loss(fake_img, real_img) * L1_LAMBDA
        loss_gen = loss_gen_adv + loss_gen_l1
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx % 10 == 0:
            print(f"批次 [{batch_idx}/{len(loader)}] 判别器损失: {loss_disc:.4f}, 生成器损失: {loss_gen:.4f}")

# 评估函数 (通用)
def evaluate_fn(gen, loader, device):
    gen.eval()
    psnr_scores = []
    ssim_scores = []
    with torch.no_grad():
        for i, (real_img, e_val) in enumerate(loader):
            real_img = real_img.to(device)
            e_val = e_val.to(device)
            noise = torch.randn(real_img.size(0), Z_DIM, 1, 1).to(device)
            fake_img = gen(noise, e_val)
            real_img_np = (real_img.cpu().numpy() + 1) / 2
            fake_img_np = (fake_img.cpu().numpy() + 1) / 2
            for j in range(real_img.size(0)):
                real = np.transpose(real_img_np[j], (1, 2, 0))
                fake = np.transpose(fake_img_np[j], (1, 2, 0))
                psnr_val = psnr(real, fake, data_range=1.0)
                ssim_val = ssim(real, fake, data_range=1.0, channel_axis=2, win_size=min(real.shape[0], real.shape[1], 3))
                psnr_scores.append(psnr_val)
                ssim_scores.append(ssim_val)
    return np.mean(psnr_scores), np.mean(ssim_scores)

# 在测试集上生成并保存最终图像的函数
def save_final_images(gen, loader, device, output_dir="results"):
    print(f"\n在测试集上生成最终对比图并保存至 '{output_dir}'...")
    gen.eval()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with torch.no_grad():
        for i, (real_img, e_val) in enumerate(loader):
            real_img = real_img.to(device)
            e_val = e_val.to(device)
            noise = torch.randn(real_img.size(0), Z_DIM, 1, 1).to(device)
            fake_img = gen(noise, e_val)
            real_img_np = (real_img.cpu().numpy() + 1) / 2
            fake_img_np = (fake_img.cpu().numpy() + 1) / 2

            # 遍历批次中的每张图片进行保存
            for j in range(real_img.size(0)):
                fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                fig.suptitle(f'E-value: {e_val[j].item():.2f}', fontsize=16)
                
                # 真实图像
                axes[0].imshow(np.transpose(real_img_np[j], (1, 2, 0)))
                axes[0].set_title("Real Image")
                axes[0].axis('off')
                
                # 生成图像
                axes[1].imshow(np.transpose(fake_img_np[j], (1, 2, 0)))
                axes[1].set_title("Generated Image")
                axes[1].axis('off')

                # 保存对比图
                img_index = i * loader.batch_size + j
                plt.savefig(f"{output_dir}/comparison_{img_index}.png")
                plt.close(fig)
            
            # 为了演示，只生成第一个批次的对比图，以免文件过多
            # 如果您想生成所有测试集的对比图，请注释掉下面这行
            break 
    print("最终对比图已保存。")


# 主执行流程
if __name__ == "__main__":
    print(f"使用设备: {DEVICE}")

    # 定义图像变换
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)])
    ])
    
    # 创建数据集和加载器
    if not os.path.exists(JSON_FILE_PATH):
        raise FileNotFoundError(f"JSON文件未找到: {JSON_FILE_PATH}，请检查路径。")

    train_dataset = DataSetLoader(JSON_FILE_PATH, IMAGE_ROOT_DIR, dataset_type='train', transform=transform)
    val_dataset = DataSetLoader(JSON_FILE_PATH, IMAGE_ROOT_DIR, dataset_type='val', transform=transform)
    test_dataset = DataSetLoader(JSON_FILE_PATH, IMAGE_ROOT_DIR, dataset_type='test', transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 初始化模型
    gen = Generator(Z_DIM, CHANNELS_IMG).to(DEVICE)
    disc = Discriminator(CHANNELS_IMG).to(DEVICE)
    initialize_weights(gen)
    initialize_weights(disc)

    # 初始化优化器和损失函数
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    bce_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    # 用于追踪最佳模型的变量
    best_val_psnr = 0.0
    best_epoch = 0

    # 开始训练
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{NUM_EPOCHS} ---")
        train_fn(disc, gen, train_loader, opt_disc, opt_gen, bce_loss, l1_loss)

        # 在验证集上评估
        val_psnr, val_ssim = evaluate_fn(gen, val_loader, DEVICE)
        print(f"--- 验证集评估 (Epoch {epoch}) ---")
        print(f"平均PSNR值: {val_psnr:.4f}")
        print(f"平均SSIM值: {val_ssim:.4f}")
        
        # 检查是否是最佳模型并保存
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            best_epoch = epoch
            torch.save(gen.state_dict(), "best_generator.pth")
            print(f"*** 新的最佳模型已保存 (Epoch {epoch})，验证集PSNR: {best_val_psnr:.4f} ***")
        print("---------------------------------")


    print("\n--- 训练完成 ---")
    print(f"最佳模型出现在 Epoch {best_epoch}，其在验证集上的PSNR为 {best_val_psnr:.4f}")

    # 加载最佳模型并在测试集上进行最终评估
    print("\n加载最佳模型进行最终测试...")
    best_gen = Generator(Z_DIM, CHANNELS_IMG).to(DEVICE)
    best_gen.load_state_dict(torch.load("best_generator.pth"))
    
    test_psnr, test_ssim = evaluate_fn(best_gen, test_loader, DEVICE)
    print("\n--- 最终测试集评估结果 ---")
    print(f"平均PSNR值: {test_psnr:.4f}")
    print(f"平均SSIM值: {test_ssim:.4f}")
    print("----------------------------")
    
    # 使用最佳模型生成并保存最终的对比图像
    save_final_images(best_gen, test_loader, DEVICE)

    # (可选) 保存最后一个周期的模型
    # torch.save(gen.state_dict(), "final_generator.pth")
    # torch.save(disc.state_dict(), "final_discriminator.pth")
    # print("\n最终周期的模型已保存。")