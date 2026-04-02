# ======================================================================================
# Straw6D Kaggle Training Script (Grid-based Multi-Object Detection)
# --------------------------------------------------------------------------------------
# 该脚本由本地多个模块 (Dataset, Model, Loss, Train Loop) 缝合而成。
# 专门适配 Kaggle Kernel 环境：
# 1. 解除本地 import 依赖，所有核心类定义在此文件中直接实现。
# 2. 自动检测 GPU 环境。
# 3. 数据集路径强制指向 Kaggle 挂载点: /kaggle/input/datasets/chenvision/straw6d-raw/
# 4. 模型保存路径强制指向 Kaggle 工作区: /kaggle/working/
# 
# 升级说明 (Grid-based Single-Shot):
# - 输出维度从 [B, 22] 变为 [B, 22, 18, 25]
# - Dataset 使用 Grid 划分，将目标映射到对应的 Grid Cell
# - Loss 函数使用掩码 (Mask) 仅在有物体的 Grid 上计算回归损失
# ======================================================================================

import os
import csv
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import time

# --------------------------------------------------------------------------------------
# [Part 1] 数据集类定义 (Straw6DDataset) - Grid-based
# --------------------------------------------------------------------------------------
class Straw6DDataset(Dataset):
    """
    Straw6D 草莓 6D 位姿估计数据集类 (Grid-based)。
    """
    def __init__(self, root_dir, transform=None):
        """
        初始化 Dataset。
        root_dir: 数据集根目录 (Kaggle: /kaggle/input/datasets/chenvision/straw6d-raw/)
        """
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.box_dir = os.path.join(root_dir, 'boxes')
        
        # 获取所有 PNG 图像文件并排序
        if not os.path.exists(self.image_dir):
            print(f"Warning: Image directory not found at {self.image_dir}")
            self.image_files = []
        else:
            self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
        
        # 定义默认的数据增强和预处理流程
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((600, 800)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        # 相机内参 (Camera Intrinsics)
        self.fx = 400.32
        self.fy = 400.32
        self.cx = 400.0
        self.cy = 300.0
        
        # Grid 参数
        self.grid_h = 18
        self.grid_w = 25
        self.downsample = 32 # 800/25=32, 600/18=33.33... (Resize导致非精确整数，以32为准)
                             # 注意：这里我们使用 32 倍下采样作为近似

    def __len__(self):
        return len(self.image_files)

    def euler_to_rotation_matrix(self, roll, pitch, yaw):
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll), np.cos(roll)]])
        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])
        R = np.dot(R_z, np.dot(R_y, R_x))
        return R

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        box_path = os.path.join(self.box_dir, img_name.replace('.png', '.csv'))

        # 1. 读取图像
        image = cv2.imread(img_path)
        if image is None:
             raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image)
        image_tensor = self.transform(image_pil)
        
        # 2. 初始化 Grid Target Tensor [22, 18, 25]
        # 维度顺序：Channels, Grid_H, Grid_W
        target_tensor = torch.zeros((22, self.grid_h, self.grid_w), dtype=torch.float32)

        # 3. 解析 CSV 标签
        if os.path.exists(box_path):
            with open(box_path, 'r') as f:
                reader = csv.reader(f)
                try:
                    header = next(reader)
                except StopIteration:
                    pass
                
                for row in reader:
                    if not row: continue
                    
                    try:
                        label = float(row[0])
                        x, y, z = float(row[1]), float(row[2]), float(row[3])
                        w, h, l = float(row[4]), float(row[5]), float(row[6])
                        roll, pitch, yaw = float(row[7]), float(row[8]), float(row[9])
                    except ValueError:
                        continue

                    # 计算 3D 旋转矩阵与投影
                    R = self.euler_to_rotation_matrix(roll, pitch, yaw)
                    x_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2, 0]
                    y_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2, 0]
                    z_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2, 0]
                    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
                    corners_3d = np.dot(R, corners) + np.array([x, y, z], dtype=np.float32).reshape(3, 1)
                    corners_3d = corners_3d.transpose(1, 0)

                    pts_2d = []
                    for i in range(9):
                        X_cam, Y_cam, Z_cam = corners_3d[i][0], -corners_3d[i][1], -corners_3d[i][2]
                        if abs(Z_cam) < 1e-6:
                            u, v = 0, 0
                        else:
                            u = self.fx * X_cam / Z_cam + self.cx
                            v = self.fy * Y_cam / Z_cam + self.cy
                        pts_2d.append([u, v])
                    
                    center_u, center_v = pts_2d[8]
                    
                    # 简单的可见性检查
                    if not (0 <= center_u < 800 and 0 <= center_v < 600):
                        continue

                    # --- Grid 映射核心逻辑 ---
                    # 计算中心点所在的 Grid 坐标
                    grid_x = int(center_u / 32)
                    grid_y = int(center_v / 32) # 这里使用32作为近似步长，实际上 600/18=33.33
                                                # 为了保证索引不越界，需要进行截断
                    
                    # 边界保护
                    if grid_x >= self.grid_w: grid_x = self.grid_w - 1
                    if grid_y >= self.grid_h: grid_y = self.grid_h - 1
                    if grid_x < 0: grid_x = 0
                    if grid_y < 0: grid_y = 0

                    # 构建 22 维向量
                    vec_center = [center_u, center_v]
                    vec_offsets = []
                    for i in range(8):
                        du = pts_2d[i][0] - center_u
                        dv = pts_2d[i][1] - center_v
                        vec_offsets.extend([du, dv])
                    vec_dims = [w, h, l]
                    vec_conf = [1.0] # 置信度设为 1
                    
                    target = vec_center + vec_offsets + vec_dims + vec_conf
                    
                    # 赋值到对应网格
                    # 如果同一个格子有多个物体，简单的覆盖策略 (YOLOv1 风格)
                    target_tensor[:, grid_y, grid_x] = torch.tensor(target, dtype=torch.float32)

        return image_tensor, target_tensor

# --------------------------------------------------------------------------------------
# [Part 2] 模型定义 (Darknet Backbone + Transformer Neck + Grid Head)
# --------------------------------------------------------------------------------------
class ConvBlock(nn.Module):
    """基础卷积块"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.leaky_relu(self.bn(self.conv(x)))

class TransformerNeck(nn.Module):
    """轻量化 Transformer Neck"""
    def __init__(self, in_channels=1024, d_model=256, nhead=8, num_layers=1):
        super(TransformerNeck, self).__init__()
        self.reduce_conv = ConvBlock(in_channels, d_model, kernel_size=1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.reduce_conv(x) # [B, 256, 18, 25]
        x_seq = x.flatten(2).transpose(1, 2) # [B, 450, 256]
        x_enhanced = self.transformer(x_seq)
        x_out = x_enhanced.transpose(1, 2).reshape(B, -1, H, W)
        return x_out

class Darknet19(nn.Module):
    """Darknet-19 完整架构 (Grid-based Head)"""
    def __init__(self, num_outputs=22):
        super(Darknet19, self).__init__()
        # Backbone
        self.conv1 = ConvBlock(3, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = ConvBlock(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = ConvBlock(64, 128, 3, padding=1)
        self.conv4 = ConvBlock(128, 64, 1, padding=0)
        self.conv5 = ConvBlock(64, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv6 = ConvBlock(128, 256, 3, padding=1)
        self.conv7 = ConvBlock(256, 128, 1, padding=0)
        self.conv8 = ConvBlock(128, 256, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv9 = ConvBlock(256, 512, 3, padding=1)
        self.conv10 = ConvBlock(512, 256, 1, padding=0)
        self.conv11 = ConvBlock(256, 512, 3, padding=1)
        self.conv12 = ConvBlock(512, 256, 1, padding=0)
        self.conv13 = ConvBlock(256, 512, 3, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.conv14 = ConvBlock(512, 1024, 3, padding=1)
        self.conv15 = ConvBlock(1024, 512, 1, padding=0)
        self.conv16 = ConvBlock(512, 1024, 3, padding=1)
        self.conv17 = ConvBlock(1024, 512, 1, padding=0)
        self.conv18 = ConvBlock(512, 1024, 3, padding=1)
        
        # Neck
        self.neck = TransformerNeck(in_channels=1024, d_model=256)
        
        # Head (Modified for Grid-based Output)
        # 输入: [B, 256, 18, 25]
        # 输出: [B, 22, 18, 25]
        # 使用 1x1 卷积代替全连接层
        self.head_conv = nn.Conv2d(256, num_outputs, kernel_size=1)

    def forward(self, x):
        # Backbone
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool3(self.conv5(x))
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool4(self.conv8(x))
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.pool5(self.conv13(x))
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        
        # Neck
        x = self.neck(x)
        
        # Head (Spatial Output)
        x = self.head_conv(x) # [B, 22, 18, 25]
        return x

class Darknet(Darknet19):
    def __init__(self):
        super(Darknet, self).__init__(num_outputs=22)

# --------------------------------------------------------------------------------------
# [Part 3] 掩码多任务损失函数 (Masked Multi-task Loss)
# --------------------------------------------------------------------------------------
class Straw6DLoss(nn.Module):
    def __init__(self, w_coord=1.0, w_dim=0.5, w_conf=1.0):
        super(Straw6DLoss, self).__init__()
        self.w_coord = w_coord
        self.w_dim = w_dim
        self.w_conf = w_conf
        # --- FIX: 使用 SmoothL1Loss 解决大目标漏检问题 ---
        self.reg_loss = nn.SmoothL1Loss(reduction='sum')
        # -----------------------------------------------
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, pred, target):
        """
        pred: [B, 22, 18, 25]
        target: [B, 22, 18, 25]
        """
        batch_size = pred.shape[0]

        # 1. 生成正样本掩码 (Object Mask)
        # target[:, 21, :, :] 是置信度通道，有物体为 1，无物体为 0
        obj_mask = target[:, 21, :, :] > 0 # [B, 18, 25] BoolTensor
        noobj_mask = ~obj_mask
        
        num_pos = obj_mask.sum().item()
        if num_pos == 0:
            # 避免除以零，如果没有正样本，只计算置信度损失
            loss_conf = self.bce_loss(pred[:, 21, :, :], target[:, 21, :, :])
            return self.w_conf * loss_conf, {"loss_coord": 0.0, "loss_dim": 0.0, "loss_conf": loss_conf.item()}

        # 2. 提取正样本的预测值和真实值
        # 使用掩码索引，结果形状为 [N_pos, C]
        # permute 到 [B, 18, 25, 22] 以便使用 mask
        pred_perm = pred.permute(0, 2, 3, 1)     # [B, 18, 25, 22]
        target_perm = target.permute(0, 2, 3, 1) # [B, 18, 25, 22]
        
        pred_pos = pred_perm[obj_mask]     # [N_pos, 22]
        target_pos = target_perm[obj_mask] # [N_pos, 22]

        # 3. 坐标损失 (只计算正样本)
        # 0:2 Center, 2:18 Vertex Offsets
        loss_center = self.reg_loss(pred_pos[:, 0:2], target_pos[:, 0:2])
        loss_vertex = self.reg_loss(pred_pos[:, 2:18], target_pos[:, 2:18])
        loss_coord = (loss_center + loss_vertex) / num_pos

        # 4. 尺寸损失 (只计算正样本)
        # 18:21 Dimensions
        loss_dim = self.reg_loss(pred_pos[:, 18:21], target_pos[:, 18:21]) / num_pos

        # 5. 置信度损失 (所有样本)
        # 21 Confidence
        # 这里使用 BCEWithLogitsLoss，自动处理 Sigmoid
        loss_conf = self.bce_loss(pred[:, 21, :, :], target[:, 21, :, :])
        
        # 6. 总损失
        total_loss = self.w_coord * loss_coord + \
                     self.w_dim * loss_dim + \
                     self.w_conf * loss_conf
                     
        return total_loss, {
            "loss_coord": loss_coord.item(),
            "loss_dim": loss_dim.item(),
            "loss_conf": loss_conf.item()
        }

# --------------------------------------------------------------------------------------
# [Part 4] 训练主循环 (Training Loop)
# --------------------------------------------------------------------------------------
def main():
    print(">>> Initializing Kaggle Training Environment (Grid-based Multi-Object)...")
    
    # 1. 超参数设置
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 300
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # 2. 路径配置 (Kaggle 环境)
    DATASET_ROOT = '/kaggle/input/datasets/chenvision/straw6d-raw/'
    CHECKPOINT_PATH = '/kaggle/working/darknet_strawberry_checkpoint.pth'
    
    # 简单的路径检查
    if not os.path.exists(DATASET_ROOT):
        print(f"ERROR: Dataset root {DATASET_ROOT} does not exist!")
        print("Please check your Kaggle dataset mount path.")
        return

    # 3. 数据集加载
    print(f"Loading dataset from {DATASET_ROOT}...")
    dataset = Straw6DDataset(root_dir=DATASET_ROOT)
    print(f"Dataset size: {len(dataset)} samples")
    
    # 注意：移除了自定义 collate_fn，因为现在的 target 已经是统一形状的 Grid Tensor
    train_loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2,    # Kaggle 双核 CPU 优化
        pin_memory=True   # 加速 Host 到 Device 传输
    )

    # 4. 模型初始化
    print("Building Darknet model (Grid-based)...")
    model = Darknet().to(DEVICE)
    
    # 5. 优化器与损失函数
    criterion = Straw6DLoss(w_coord=1.0, w_dim=0.5, w_conf=1.0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 6. 训练循环
    print(">>> Start Training...")
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        batch_count = 0
        
        for i, (images, targets) in enumerate(train_loader):
            # images: [B, 3, 600, 800]
            # targets: [B, 22, 18, 25]
            
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            # Forward
            outputs = model(images) # [B, 22, 18, 25]
            
            # Loss Calculation
            loss, loss_dict = criterion(outputs, targets)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1
            
            # Log every 50 batches
            if (i + 1) % 50 == 0:
                print(f"[Epoch {epoch+1}/{NUM_EPOCHS}][Batch {i+1}] Loss: {loss.item():.4f}")

        # End of Epoch Stats
        epoch_avg_loss = running_loss / max(1, batch_count)
        elapsed = time.time() - start_time
        print(f"===> Epoch [{epoch+1}/{NUM_EPOCHS}] Finished. Avg Loss: {epoch_avg_loss:.4f}. Time Elapsed: {elapsed:.1f}s")
        
        # Save Checkpoint
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"Checkpoint saved to {CHECKPOINT_PATH}")

    print(">>> Training Complete!")

if __name__ == '__main__':
    main()
