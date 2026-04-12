# 草莓 6D 位姿估计系统 - 项目状态与任务追踪 (STATUS.md)

## 1. 当前里程碑 (Milestone: 中期答辩已准备)
*   **模型训练:** 已完成 300 Epoch，Avg Loss 稳定在 **6.47**。
*   **权重文件:** `darknet_strawberry_checkpoint.pth` (已加载并验证)。
*   **数据逻辑:** **[已完成]** `straw6d_dataset.py` 已重构为支持多目标网格化加载，验证维度为 `[Batch, 22, 18, 25]`。
*   **核心闭环:** 支持单张图像上传、3D 框绘制、Base64 结果反馈。

## 2. 工程化记录 (Engineering Notes)
*   **终端编码处理:** 在 Windows 环境下，为了避免 UTF-8 中文乱码，所有脚本输出应优先重定向到临时文件 (`> tmp.txt`) 后再读取。

## 3. 已知瓶颈 (Bottlenecks)
*   **推理后处理逻辑:** 目前的 `utils/image_processing.py` 仍停留在旧的单目标/简单逻辑，需适配 $18 \times 25$ 网格解码。
*   **实验数据真实性:** `benchmark_comparison.py` 目前使用随机 2D 关键点进行位姿模拟，需接入真实模型输出。

## 4. 下一步任务清单 (Next Tasks)
- [ ] **重构 Inference:** 在 `utils/image_processing.py` 中实现网格解码、阈值筛选及 PnP 闭环。
- [ ] **完善推理逻辑:** 在 API 中接入重构后的 `run_inference`。
- [ ] **生成学术图表:** 基于 `训练日志.md` 生成 Loss 收敛曲线图。


## 4. 待解决疑问
*   是否需要引入多尺度特征融合 (FPN) 来提升极小草莓的检测精度？
