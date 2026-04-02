import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import Straw6DDataset
from models.darknet import Darknet
import numpy as np

# --- 1. 自定义 Collate Function (单目标选择) ---
def single_target_collate_fn(batch):
    """
    由于 Darknet 模型设计为输出单目标预测 [B, 22]，而数据集中一张图片可能包含多个草莓。
    我们需要在 batch 组装时，从每张图片的所有标注中选择一个"主要目标"。
    这里我们选择距离图像中心 (400, 300) 最近的草莓作为训练目标。
    """
    images = []
    targets = []
    
    img_center = np.array([400.0, 300.0])
    
    for img, tgt in batch:
        # tgt shape: [N, 22]
        if tgt.shape[0] > 0:
            # 计算每个目标的中心点距离图像中心的距离
            # tgt[:, 0:2] 是中心点 (u, v)
            centers = tgt[:, 0:2].numpy()
            distances = np.linalg.norm(centers - img_center, axis=1)
            
            # 找到距离最小的索引
            min_idx = np.argmin(distances)
            
            # 选择该目标作为训练目标
            selected_target = tgt[min_idx] # [22]
            
            images.append(img)
            targets.append(selected_target)
        else:
            # 如果图片中没有目标，可以选择跳过该样本，或者添加一个全0的目标（置信度为0）
            # 这里为了简单起见，我们添加一个全0目标，并将置信度设为0
            # 注意：全0目标的坐标为0，可能会产生较大的坐标损失，但置信度损失会使其被抑制
            # 更稳妥的做法是在 Dataset 中就过滤掉没有目标的图片，或者在这里跳过
            # 我们选择跳过没有目标的样本，以避免噪声
            continue

    if len(images) == 0:
        return None, None

    # 堆叠成 Batch Tensor
    images = torch.stack(images, dim=0) # [B, 3, 600, 800]
    targets = torch.stack(targets, dim=0) # [B, 22]
    
    return images, targets

# --- 2. 多任务损失函数 ---
class Straw6DLoss(nn.Module):
    def __init__(self, w_coord=1.0, w_dim=0.5, w_conf=1.0):
        super(Straw6DLoss, self).__init__()
        self.w_coord = w_coord
        self.w_dim = w_dim
        self.w_conf = w_conf
        
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        """
        pred: [B, 22]
        target: [B, 22]
        """
        # 拆分预测向量
        # 0:2 -> 中心点 (2D)
        pred_center = pred[:, 0:2]
        target_center = target[:, 0:2]
        
        # 2:18 -> 顶点偏移 (16D)
        pred_vertex = pred[:, 2:18]
        target_vertex = target[:, 2:18]
        
        # 18:21 -> 尺寸 (3D)
        pred_dim = pred[:, 18:21]
        target_dim = target[:, 18:21]
        
        # 21 -> 置信度 (1D)
        pred_conf = pred[:, 21] # Logits
        target_conf = target[:, 21] # 0.0 or 1.0

        # --- 计算各部分损失 ---
        
        # 1. 坐标损失 (Center + Vertex Offsets)
        # 只有在有目标时才计算坐标损失 (Target Conf = 1)
        # mask = target_conf > 0.5
        # if mask.sum() > 0:
        #    loss_center = self.mse_loss(pred_center[mask], target_center[mask])
        #    loss_vertex = self.mse_loss(pred_vertex[mask], target_vertex[mask])
        # else:
        #    loss_center = torch.tensor(0.0, device=pred.device)
        #    loss_vertex = torch.tensor(0.0, device=pred.device)
        
        # 简单起见，我们假设所有样本都有目标（collate_fn已过滤），直接计算MSE
        loss_center = self.mse_loss(pred_center, target_center)
        loss_vertex = self.mse_loss(pred_vertex, target_vertex)
        loss_coord = loss_center + loss_vertex
        
        # 2. 尺寸损失
        loss_dim = self.mse_loss(pred_dim, target_dim)
        
        # 3. 置信度损失
        loss_conf = self.bce_loss(pred_conf, target_conf)
        
        # --- 加权求和 ---
        total_loss = self.w_coord * loss_coord + \
                     self.w_dim * loss_dim + \
                     self.w_conf * loss_conf
                     
        return total_loss, {
            "loss_coord": loss_coord.item(),
            "loss_dim": loss_dim.item(),
            "loss_conf": loss_conf.item()
        }

# --- 3. 训练主循环 ---
def main():
    # 配置参数
    BATCH_SIZE = 8 # 根据显存调整
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # 1. 数据集与 DataLoader
    dataset = Straw6DDataset(root_dir=r'd:\HELLO\huggingface\bishe\Straw6D_Raw')
    print(f"Dataset size: {len(dataset)}")
    
    train_loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0, # Windows下建议设为0或2
        collate_fn=single_target_collate_fn
    )

    # 2. 模型
    model = Darknet().to(DEVICE)
    
    # 3. 损失函数与优化器
    criterion = Straw6DLoss(w_coord=1.0, w_dim=0.5, w_conf=1.0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. 训练循环
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, (images, targets) in enumerate(train_loader):
            if images is None: # 跳过空 Batch
                continue
                
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            # 前向传播
            outputs = model(images)
            
            # 计算损失
            loss, loss_dict = criterion(outputs, targets)
            
            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f} "
                      f"(Coord: {loss_dict['loss_coord']:.4f}, Dim: {loss_dict['loss_dim']:.4f}, Conf: {loss_dict['loss_conf']:.4f})")

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Average Loss: {epoch_loss:.4f}")

    # 5. 保存模型
    save_path = "darknet_strawberry.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Training finished. Model saved to {save_path}")

if __name__ == '__main__':
    main()
