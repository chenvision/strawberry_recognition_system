import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator
from matplotlib.ticker import ScalarFormatter

# 1. 核心数据准备
epochs = np.array([1, 50, 200, 300])
losses = np.array([742.38, 29.24, 8.93, 6.47])

# 使用 Pchip 插值生成平滑曲线，避免曲线出现反常的震荡或跌破最小值
spline = PchipInterpolator(epochs, losses)
epochs_smooth = np.linspace(1, 300, 500)
losses_smooth = spline(epochs_smooth)

# 2. 初始化画布 (高分辨率，适合论文/报告)
fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

# 3. 视觉样式设置 (还原截图的橙色主题)
color_line = '#ED7D31'  # 核心折线橙色
color_fill = '#FDF3EB'  # 下方填充浅橙色

# 绘制平滑曲线与真实数据点
ax.plot(epochs_smooth, losses_smooth, color=color_line, linewidth=3, zorder=3)
ax.scatter(epochs, losses, color=color_line, s=60, zorder=4)

# 填充曲线下方区域
ax.fill_between(epochs_smooth, losses_smooth, 0, color=color_fill, alpha=0.8, zorder=2)

# 4. 坐标轴处理 (核心：使用对数坐标解决数值落差过大问题)
ax.set_yscale('log')
ax.yaxis.set_major_formatter(ScalarFormatter()) # 让Y轴显示具体数字而非科学计数法

# 5. 背景与边框美化
ax.set_xlabel('Epochs', fontsize=12, fontweight='bold', color='#333333')
ax.set_ylabel('Avg Loss', fontsize=12, fontweight='bold', color='#333333')

# 隐藏上、右边框，弱化左、下边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#CCCCCC')
ax.spines['bottom'].set_color('#CCCCCC')
ax.tick_params(colors='#555555')
ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=1)

# 6. 划分三个训练阶段的背景带
ax.axvspan(1, 50, color='#E8F8F5', alpha=0.6, label='Phase 1: Rapid Convergence (1-50)', zorder=0)
ax.axvspan(50, 200, color='#EAF2F8', alpha=0.6, label='Phase 2: Fine Optimization (51-200)', zorder=0)
ax.axvspan(200, 300, color='#FEF9E7', alpha=0.6, label='Phase 3: Stable Convergence (201-300)', zorder=0)

# 添加图例
ax.legend(loc='upper right', frameon=True, fontsize=10)

plt.tight_layout()
# 7. 保存并展示图表
plt.savefig('loss_curve_optimized.png', bbox_inches='tight')
plt.show()