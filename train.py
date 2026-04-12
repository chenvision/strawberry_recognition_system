import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import Straw6DDataset
from models.darknet import Darknet

# --- 1. 掩码多任务损失函数 (与 kaggle_train.py 保持一致) ---
class Straw6DLoss(nn.Module):
    def __init__(self, w_coord=1.0, w_dim=0.5, w_conf=1.0):
        super(Straw6DLoss, self).__init__()
        self.w_coord = w_coord
        self.w_dim = w_dim
        self.w_conf = w_conf
        self.reg_loss = nn.SmoothL1Loss(reduction='sum')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, pred, target):
        """
        pred:   [B, 22, 18, 25]
        target: [B, 22, 18, 25]
        """
        # 生成正样本掩码：置信度通道 > 0 的格子
        obj_mask = target[:, 21, :, :] > 0  # [B, 18, 25]

        num_pos = obj_mask.sum().item()
        if num_pos == 0:
            loss_conf = self.bce_loss(pred[:, 21, :, :], target[:, 21, :, :])
            return self.w_conf * loss_conf, {"loss_coord": 0.0, "loss_dim": 0.0, "loss_conf": loss_conf.item()}

        # 提取正样本 [N_pos, 22]
        pred_perm   = pred.permute(0, 2, 3, 1)
        target_perm = target.permute(0, 2, 3, 1)
        pred_pos   = pred_perm[obj_mask]
        target_pos = target_perm[obj_mask]

        # 坐标损失（中心点 + 顶点偏移，只算正样本）
        loss_center = self.reg_loss(pred_pos[:, 0:2],  target_pos[:, 0:2])
        loss_vertex = self.reg_loss(pred_pos[:, 2:18], target_pos[:, 2:18])
        loss_coord  = (loss_center + loss_vertex) / num_pos

        # 尺寸损失（只算正样本）
        loss_dim = self.reg_loss(pred_pos[:, 18:21], target_pos[:, 18:21]) / num_pos

        # 置信度损失（全部格子）
        loss_conf = self.bce_loss(pred[:, 21, :, :], target[:, 21, :, :])

        total_loss = self.w_coord * loss_coord + self.w_dim * loss_dim + self.w_conf * loss_conf
        return total_loss, {
            "loss_coord": loss_coord.item(),
            "loss_dim":   loss_dim.item(),
            "loss_conf":  loss_conf.item()
        }


# --- 2. 训练主循环 ---
def main():
    BATCH_SIZE    = 8       # 本地显存较小，调低；Kaggle 用 16
    LEARNING_RATE = 0.001
    NUM_EPOCHS    = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # 数据集（dataset.py 返回 Grid Tensor [22, 18, 25]，可直接 stack）
    dataset = Straw6DDataset(root_dir=r'd:\HELLO\huggingface\bishe\Straw6D_Raw')
    print(f"Dataset size: {len(dataset)}")

    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,   # Windows 下建议 0
    )

    model     = Darknet().to(DEVICE)
    criterion = Straw6DLoss(w_coord=1.0, w_dim=0.5, w_conf=1.0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for i, (images, targets) in enumerate(train_loader):
            images  = images.to(DEVICE)   # [B, 3, 600, 800]
            targets = targets.to(DEVICE)  # [B, 22, 18, 25]

            outputs = model(images)
            loss, loss_dict = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f} "
                      f"(Coord: {loss_dict['loss_coord']:.4f}, "
                      f"Dim: {loss_dict['loss_dim']:.4f}, "
                      f"Conf: {loss_dict['loss_conf']:.4f})")

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Average Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), "darknet_strawberry.pth")
    print("Training finished. Model saved to darknet_strawberry.pth")


if __name__ == '__main__':
    main()
