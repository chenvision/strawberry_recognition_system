import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
import albumentations as A
import cv2
#欧拉角转旋转矩阵
def euler_to_rotation(theta):
    R_x=np.array([[1,0,0],
                 [0,np.cos(theta[0]),-np.sin(theta[0])],
                 [0,np.sin(theta[0]),np.cos(theta[0])]])
    R_y=np.array([[np.cos(theta[1]),0,np.sin(theta[1])],
                 [0,1,0],
                 [-np.sin(theta[1]),0,np.cos(theta[1])]])
    R_z=np.array([[np.cos(theta[2]),-np.sin(theta[2]),0],
                 [np.sin(theta[2]),np.cos(theta[2]),0],
                 [0,0,1]])
    return np.dot(R_z,np.dot(R_y,R_x))
class Straw6DDataset(Dataset):
    def __init__(self,root_dir,split='train',transforms=None):
        self.root_dir=root_dir
        self.split=split
        self.transforms=transforms
        self.img_dir=os.path.join(root_dir,'images')
        self.box_dir=os.path.join(root_dir,'boxes')
        # TODO: 在这里收集所有样本的路径或文件名
        self.file_names=[
            f.split('.')[0]
            for f in os.listdir(self.img_dir)
            if f.endswith('.png')
        ]
    def __len__(self):
        return len(self.file_names)
    def __getitem__(self,idx):
        # 1. 拼接图片和CSV的具体路径
        file_name=self.file_names[idx]
        img_path=os.path.join(self.img_dir,f"{file_name}.png")
        csv_path=os.path.join(self.box_dir,f"{file_name}.csv")
        # 2. 读取图片
        image=cv2.imread(img_path)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        # 3. 读取并解析csv标签
        # CSV 包含: label, x, y, z, w, h, l, roll, pitch, yaw
        df=pd.read_csv(csv_path)
        # 提取第一行（单株）数据转为字典方便调用
        labels=df.iloc[0].to_dict()
        x,y,z=labels['x'],labels['y'],labels['z']
        w,h,l=labels['w'],labels['h'],labels['l']
        roll,pitch,yaw=labels['roll'],labels['pitch'],labels['yaw']
        #3D到2D关键点投影计算
        R=euler_to_rotation([roll,pitch,yaw])
        #定义局部坐标系下的八个顶点和一个中心点（索引8是中心点[0,0,0]）
        x_corners=[w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2,0]
        y_corners=[h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2,0]
        z_corners=[l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2,0]
        corners=np.array([x_corners,y_corners,z_corners],dtype=np.float32)
        #旋转并平移到相机坐标系
        corners_3d=np.dot(R,corners)+np.array([x,y,z],dtype=np.float32).reshape(3,1)
        corners_3d=corners_3d.transpose(1,0)
        #投影到2D像素坐标系（结合相机的内参（fx,fy和cx,cy））
        pts_2d = []
        for i in range(9):
            #注意坐标轴翻转规则
            X, Y, Z = corners_3d[i][0], -corners_3d[i][1], -corners_3d[i][2]
            u, v = 400.32 * X / Z + 400, 400.32 * Y / Z + 300  # Assuming these are your camera parameters
            pts_2d.append([u, v])
        # 4. 执行 Albumentations 数据增强
        if self.transforms:
            #要同时传入image和keypoints
            augmented=self.transforms(image=image,keypoints=pts_2d)
            image=augmented['image']
            pts_2d=augmented['keypoints']
        # 5. 格式转换：转换为Pytorch需要的Tensor格式
        image=torch.tensor(image,dtype=torch.float32).permute(2,0,1)/255.0
        # 假设 W, H 为图像宽高，S 为网格划分数量
        W, H, S = 800, 600, 20
        grid_w, grid_h = W / S, H / S
        # 提取中心点坐标
        center_u, center_v = pts_2d[8]

        # 中心点所在网格的行列索引
        grid_x, grid_y = int(center_u / grid_w), int(center_v / grid_h)

        # 中心点相对于网格左上角的偏移量
        offset_x = (center_u / grid_w) - grid_x
        offset_y = (center_v / grid_h) - grid_y

        # 计算 8 个顶点的偏移量，并交替存入一个列表中 [du0, dv0, du1, dv1, ...]
        vertex_offsets = []
        for i in range(8):
            delta_u = pts_2d[i][0] - center_u
            delta_v = pts_2d[i][1] - center_v
            vertex_offsets.extend([delta_u, delta_v])
        center_offsets=[offset_x,offset_y]
        size_3d_list=[w,h,l]
        confidence=[1.0]
        target_vector=center_offsets+vertex_offsets+size_3d_list+confidence
        target_tensor=torch.tensor(target_vector, dtype=torch.float32)
        return image,target_tensor
if __name__=="__main__":
    # 1. 实例化数据集，路径指向上一层的 Straw6D_Raw 文件夹 📂
    dataset = Straw6DDataset(root_dir='../Straw6D_Raw')
    train_loader=DataLoader(dataset=dataset,batch_size=16,shuffle=True)
    
    # 2. 像使用列表一样，取出第 0 个样本（这会自动触发我们写的 __getitem__ 方法）
    images, targets = next(iter(train_loader))
    
    # 3. 打印它们的维度 🖨️
    print(f"提取成功！数据集总大小: {len(dataset)}")
    print(f"图片张量维度 (Image Shape): {images.shape}")
    print(f"标签张量维度 (Target Shape): {targets.shape}")