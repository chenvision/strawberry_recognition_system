import torch
import cv2
import numpy as np
from models.darknet import Darknet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 准备“躯壳”
model = Darknet().to(device)

# 2. 读取硬盘里的权重字典
checkpoint = torch.load('darknet_strawberry_checkpoint.pth', map_location=device)

# 3. 注入“灵魂” (请结合 load_state_dict 补全下面这行代码)
model.load_state_dict(checkpoint)

# 2. 切换到测试/评估模式 🛡️
model.eval()
# 2. 读取并预处理图片 🖼️
image_path = "test_strawberry.jpg"
image = cv2.imread(image_path)
image = cv2.resize(image, (800, 600)) # 调整为模型需要的 宽800，高600
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # OpenCV 默认是 BGR，转成模型训练时的 RGB

# 此时 image 的形状是 [600, 800, 3] (即 HWC: 高, 宽, 通道)
# 我们先把它转成 PyTorch 张量并调整为 [3, 600, 800] (即 CHW 格式)，同时归一化
image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0