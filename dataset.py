import os
import csv
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class Straw6DDataset(Dataset):
    """
    Straw6D 草莓 6D 位姿估计数据集类。
    
    该类负责：
    1. 读取图像并预处理（Resize, BGR->RGB, 归一化, 转 Tensor）。
    2. 解析 CSV 标注文件，计算 2D 投影点。
    3. 构建 22 维回归目标向量。
    4. 应用数据增强（随机亮度、对比度等）。
    """
    def __init__(self, root_dir, transform=None):
        """
        初始化 Dataset。

        参数:
            root_dir (str): 数据集根目录，包含 'images' 和 'boxes' 子目录。
            transform (callable, optional): 可选的图像变换。如果为 None，将使用默认的变换（Resize + ColorJitter + ToTensor）。
        """
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.box_dir = os.path.join(root_dir, 'boxes')
        
        # 获取所有 PNG 图像文件并排序
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
        
        # 定义默认的数据增强和预处理流程
        # 1. Resize: 强制调整到 800x600 (H=600, W=800)
        # 2. ColorJitter: 随机调整亮度、对比度、饱和度、色调
        # 3. ToTensor: 转为 Tensor (C, H, W) 并归一化到 [0, 1]
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((600, 800)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        # 相机内参 (Camera Intrinsics)
        # 来源于 view_dataset.py
        self.fx = 400.32
        self.fy = 400.32
        self.cx = 400.0
        self.cy = 300.0

    def __len__(self):
        """返回数据集样本数量"""
        return len(self.image_files)

    def euler_to_rotation_matrix(self, roll, pitch, yaw):
        """
        将欧拉角转换为旋转矩阵。
        顺序: Z * Y * X
        """
        # 绕 X 轴旋转
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll), np.cos(roll)]])

        # 绕 Y 轴旋转
        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])

        # 绕 Z 轴旋转
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])

        # 组合旋转矩阵 R = Rz * Ry * Rx
        R = np.dot(R_z, np.dot(R_y, R_x))
        return R

    def __getitem__(self, idx):
        """
        获取一个样本。

        返回:
            image_tensor (Tensor): [3, 600, 800], RGB, 归一化 [0, 1]
            target_tensor (Tensor): [22, 18, 25], Grid-based 标签。
                每个有目标的 Grid Cell 存储 22 维向量:
                - [0:2]:  2D 中心点绝对像素坐标 (u, v)
                - [2:18]: 16D 顶点偏移 (8个顶点相对于中心的 du, dv)
                - [18:21]: 3D 尺寸 (w, h, l)，单位米
                - [21]:   置信度 (1.0)
        """
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

        # 2. 初始化 Grid Target Tensor [22, 18, 25]，全零表示无目标
        grid_h, grid_w = 18, 25
        target_tensor = torch.zeros((22, grid_h, grid_w), dtype=torch.float32)

        # 3. 解析 CSV 标签，将每个目标映射到对应 Grid Cell
        if os.path.exists(box_path):
            with open(box_path, 'r') as f:
                reader = csv.reader(f)
                try:
                    next(reader)  # 跳过表头
                except StopIteration:
                    pass

                for row in reader:
                    if not row:
                        continue
                    try:
                        x, y, z = float(row[1]), float(row[2]), float(row[3])
                        w, h, l = float(row[4]), float(row[5]), float(row[6])
                        roll, pitch, yaw = float(row[7]), float(row[8]), float(row[9])
                    except (ValueError, IndexError):
                        continue

                    R = self.euler_to_rotation_matrix(roll, pitch, yaw)

                    x_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2, 0]
                    y_corners = [h/2,  h/2,  h/2, h/2, -h/2, -h/2, -h/2, -h/2, 0]
                    z_corners = [l/2,  l/2, -l/2, -l/2, l/2,  l/2, -l/2, -l/2, 0]
                    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
                    corners_3d = (np.dot(R, corners)
                                  + np.array([x, y, z], dtype=np.float32).reshape(3, 1))
                    corners_3d = corners_3d.transpose(1, 0)  # [9, 3]

                    pts_2d = []
                    for i in range(9):
                        X_cam = corners_3d[i][0]
                        Y_cam = -corners_3d[i][1]
                        Z_cam = -corners_3d[i][2]
                        if abs(Z_cam) < 1e-6:
                            u, v = 0.0, 0.0
                        else:
                            u = self.fx * X_cam / Z_cam + self.cx
                            v = self.fy * Y_cam / Z_cam + self.cy
                        pts_2d.append([u, v])

                    center_u, center_v = pts_2d[8]

                    # 中心点不在图像内则跳过
                    if not (0 <= center_u < 800 and 0 <= center_v < 600):
                        continue

                    # --- Grid 映射（与 kaggle_train.py 保持一致，步长 32）---
                    gx = int(center_u / 32)
                    gy = int(center_v / 32)
                    gx = min(max(gx, 0), grid_w - 1)
                    gy = min(max(gy, 0), grid_h - 1)

                    # 构建 22 维向量
                    vec_center  = [center_u, center_v]
                    vec_offsets = []
                    for i in range(8):
                        vec_offsets.extend([pts_2d[i][0] - center_u,
                                            pts_2d[i][1] - center_v])
                    vec_dims = [w, h, l]
                    vec_conf = [1.0]

                    target_vec = vec_center + vec_offsets + vec_dims + vec_conf
                    # 同一格子有多个目标时后者覆盖前者（YOLOv1 风格）
                    target_tensor[:, gy, gx] = torch.tensor(target_vec, dtype=torch.float32)

        return image_tensor, target_tensor

if __name__ == '__main__':
    # 快速验证：检查 Dataset 输出形状是否符合预期
    dataset = Straw6DDataset(root_dir=r'd:\HELLO\huggingface\bishe\Straw6D_Raw')
    print(f"数据集样本数: {len(dataset)}")

    if len(dataset) > 0:
        img, tgt = dataset[0]
        print(f"图像 Tensor 形状: {img.shape}")   # 期望 [3, 600, 800]
        print(f"标签 Tensor 形状: {tgt.shape}")   # 期望 [22, 18, 25]
        num_obj = int((tgt[21] > 0).sum().item())
        print(f"该样本中有目标的 Grid Cell 数量: {num_obj}")
