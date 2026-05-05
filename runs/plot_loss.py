import re
import matplotlib.pyplot as plt

# 1. 读取你的日志文件
log_file = 'run.txt' # 确保 run.txt 和这个脚本在同一个文件夹
epochs = []
losses = []

# 用于记录每个 Epoch 的最后一次 Loss (或者平均 Loss)
current_epoch = -1
epoch_loss_list = []

print("正在解析庞大的日志文件，请稍候...")
with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        # 使用正则表达式提取 Epoch 和 Loss 的数值
        # 匹配格式例如: [Epoch 1/300] ... Loss: 6.47
        match = re.search(r'\[Epoch (\d+)/\d+\].*?Loss:\s*([\d.]+)', line)
        if match:
            epoch = int(match.group(1))
            loss = float(match.group(2))
            
            if epoch != current_epoch:
                # 结算上一个 Epoch 的平均 Loss
                if current_epoch != -1 and epoch_loss_list:
                    epochs.append(current_epoch)
                    losses.append(sum(epoch_loss_list) / len(epoch_loss_list))
                current_epoch = epoch
                epoch_loss_list = [loss]
            else:
                epoch_loss_list.append(loss)

# 把最后一个 Epoch 也加进去
if epoch_loss_list:
    epochs.append(current_epoch)
    losses.append(sum(epoch_loss_list) / len(epoch_loss_list))

print(f"成功提取了 {len(epochs)} 个 Epoch 的 Loss 数据！")

# 2. 开始画图 (专门为毕业论文排版定制)
plt.figure(figsize=(8, 5))
plt.plot(epochs, losses, color='#2c7bb6', linewidth=2, label='Training Loss')

plt.title('Multi-task Loss Convergence Curve', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)

# 防止前期 Loss 太大导致曲线看不清，限制一下 Y 轴范围 (如果需要的话可以取消注释)
# plt.ylim(0, max(losses[10:])) 

plt.tight_layout()
plt.savefig('loss_curve.png', dpi=300) # 保存为高清图片
print("已成功生成高清曲线图: loss_curve.png")
plt.show()