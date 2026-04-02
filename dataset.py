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
            targets (Tensor): [N, 22], 其中 N 是该图片中草莓的数量。
                              如果图片中没有草莓，返回 [0, 22] 的空 Tensor。
                              22 维向量定义:
                              - [0:2]: 2D 中心点 (u, v)
                              - [2:18]: 16D 顶点偏移 (8个顶点相对于中心的偏移 du, dv)
                              - [18:21]: 3D 尺寸 (w, h, l)
                              - [21]: 置信度 (1.0)
        """
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        box_path = os.path.join(self.box_dir, img_name.replace('.png', '.csv'))

        # 1. 读取图像 (OpenCV 读取为 BGR)
        image = cv2.imread(img_path)
        if image is None:
             raise FileNotFoundError(f"Image not found: {img_path}")
        
        # 2. BGR 转 RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 3. 转为 PIL Image (为了使用 torchvision transforms)
        image_pil = Image.fromarray(image)

        # 4. 应用变换 (Resize -> ColorJitter -> ToTensor)
        image_tensor = self.transform(image_pil)
        
        # 5. 解析 CSV 标签
        targets = []
        if os.path.exists(box_path):
            with open(box_path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader) # 跳过表头
                
                for row in reader:
                    if not row: continue
                    
                    # 读取参数
                    label = float(row[0])
                    x, y, z = float(row[1]), float(row[2]), float(row[3])
                    w, h, l = float(row[4]), float(row[5]), float(row[6])
                    roll, pitch, yaw = float(row[7]), float(row[8]), float(row[9])

                    # 计算 3D 旋转矩阵
                    R = self.euler_to_rotation_matrix(roll, pitch, yaw)

                    # 定义 3D 边界框的 8 个顶点 + 中心点(0,0,0)
                    # 顺序参考 view_dataset.py
                    x_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2, 0]
                    y_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2, 0]
                    z_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2, 0]

                    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
                    
                    # 变换到世界坐标系: P_world = R * P_local + T
                    corners_3d = np.dot(R, corners) + np.array([x, y, z], dtype=np.float32).reshape(3, 1)
                    corners_3d = corners_3d.transpose(1, 0) # [9, 3]

                    # 投影到 2D 图像平面
                    pts_2d = []
                    for i in range(9):
                        # 坐标系转换: World -> Camera (X=X, Y=-Y, Z=-Z)
                        X_cam = corners_3d[i][0]
                        Y_cam = -corners_3d[i][1]
                        Z_cam = -corners_3d[i][2]
                        
                        # 投影公式
                        if abs(Z_cam) < 1e-6:
                            u, v = 0, 0
                        else:
                            u = self.fx * X_cam / Z_cam + self.cx
                            v = self.fy * Y_cam / Z_cam + self.cy
                        
                        pts_2d.append([u, v])
                    
                    # 提取中心点 (第9个点)
                    center_u, center_v = pts_2d[8]
                    
                    # 简单的可见性检查 (如果在图像外太远可能需要过滤，这里暂不过滤，或者只过滤中心点)
                    if not (0 <= center_u < 800 and 0 <= center_v < 600):
                        continue

                    # 构建 22 维向量
                    # 1. 中心点 (2)
                    vec_center = [center_u, center_v]
                    
                    # 2. 顶点偏移 (16): 顶点坐标 - 中心点坐标
                    vec_offsets = []
                    for i in range(8):
                        du = pts_2d[i][0] - center_u
                        dv = pts_2d[i][1] - center_v
                        vec_offsets.extend([du, dv])
                    
                    # 3. 尺寸 (3)
                    vec_dims = [w, h, l]
                    
                    # 4. 置信度 (1)
                    vec_conf = [1.0]
                    
                    # 拼接
                    target = vec_center + vec_offsets + vec_dims + vec_conf
                    targets.append(target)

        # 转为 Tensor
        if len(targets) == 0:
            targets_tensor = torch.zeros((0, 22), dtype=torch.float32)
        else:
            targets_tensor = torch.tensor(targets, dtype=torch.float32)

        return image_tensor, targets_tensor

# 自定义 collate_fn 示例 (用于 DataLoader)
def straw6d_collate_fn(batch):
    """
    处理变长 targets 的 collate_fn。
    batch: list of (image, targets) tuples
    """
    images = []
    targets = []
    for img, target in batch:
        images.append(img)
        targets.append(target)
    
    # 将图像堆叠为一个 batch [B, 3, 600, 800]
    images = torch.stack(images, dim=0)
    
    # targets 保持为列表，因为每个样本的目标数量不同 [B, N_i, 22]
    # 或者可以设计为 padded tensor
    return images, targets

if __name__ == '__main__':
    # 测试代码
    dataset = Straw6DDataset(root_dir=r'd:\HELLO\huggingface\bishe\Straw6D_Raw')
    print(f"数据集样本数: {len(dataset)}")
    
    if len(dataset) > 0:
        img, tgt = dataset[0]
        print(f"图像 Tensor 形状: {img.shape}")
        print(f"目标 Tensor 形状: {tgt.shape}")
        if tgt.shape[0] > 0:
            print(f"第一个目标向量 (前5维): {tgt[0][:5]}")
