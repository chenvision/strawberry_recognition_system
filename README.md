# 草莓三维姿态与尺寸估计系统 (Strawberry 3D Pose & Dimension Estimation)

![Project Status](https://img.shields.io/badge/Status-Completed-success)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![Vue](https://img.shields.io/badge/Vue.js-3.x-green)
![License](https://img.shields.io/badge/License-MIT-orange)

本课题旨在实现一个高精度、实时性强的单目图像草莓三维信息感知系统。通过结合 **CNN 局部特征提取** 与 **Transformer 全局空间建模**，系统能够直接从单帧图像中回归草莓的 3D 投影顶点及物理尺寸，并利用几何约束解算 6D 位姿。

---

## 🚀 核心特性

*   **单阶段网格预测 (Grid-based Single-shot)**：将位姿估计任务转化为 $18 \times 25$ 的网格空间多任务回归，支持多目标并行预测。
*   **CNN-Transformer 混合架构**：
    *   **Backbone**: 改进型 Darknet-19，实现高效特征提取。
    *   **Neck**: 引入 Transformer Encoder 模块，增强遮挡场景下的全局约束建模能力。
*   **6DoF 位姿解算**：集成 EPnP 算法，结合相机内参实时计算草莓的三轴坐标（X, Y, Z）与旋转角度。
*   **工业级控制台**：
    *   基于 FastAPI 的高性能推理后端。
    *   全中文汉化的 Vue.js 交互界面，支持实时视频流反馈、遥测数据展示及静态分析报告。

---

## 🛠️ 技术栈

*   **深度学习**: PyTorch, Torchvision, OpenCV (EPnP Solver)
*   **后端**: FastAPI, Uvicorn, Python 3.13
*   **前端**: Vue.js, Axios, Echarts/Dashboard
*   **训练环境**: Straw6D Dataset, NVIDIA Tesla GPU

---

## 📂 项目结构

```text
.
├── app.py                # FastAPI 后端推理接口
├── models/               # 网络模型架构定义 (Darknet-19 + Transformer)
├── datasets/             # 数据加载逻辑与 Straw6D 处理
├── frontend/             # Vue.js 前端控制台源码
├── run_project.ps1       # 一键启动脚本 (PowerShell)
├── train.py              # 本地训练脚本
├── kaggle_train.py       # Kaggle 环境训练专用代码
├── predict.py            # 离线预测验证脚本
├── 系统架构设计.md        # 学术方案设计文档
├── 实验结果分析.md        # 训练指标与误差来源分析
└── 研究方案及进度安排.md  # 课题开发路线图
```

---

## 🏁 快速启动

### 1. 环境准备
确保已安装 Python 3.13+ 和 Node.js。

### 2. 一键运行
在项目根目录下，使用 PowerShell 运行以下脚本：
```powershell
.\run_project.ps1
```
该脚本会自动检测依赖、启动 FastAPI 后端（端口 8000）并运行 Vue 前端开发服务器（端口 8080）。

### 3. 访问系统
打开浏览器访问：[http://localhost:8080](http://localhost:8080)

---

## 📊 实验表现

*   **训练周期**: 300 Epochs (Adam Optimizer)
*   **收敛精度**: 最终 Avg Loss 为 **6.47**
*   **特性表现**: Transformer Neck 模块显著提升了在密集群聚场景下的目标顶点关联准确度。
*   **误差来源**: 主要受极端遮挡及单目视觉固有的深度模糊性影响。

---

## 🎓 论文引用参考

系统架构细节、数学公式推导及实验分析请参阅本项目中的 `*.md` 文档，这些文档已按照学术论文规范整理，可直接引用。

---

## 📧 联系与反馈
如有任何疑问或技术交流，请通过 GitHub Issue 或提交 Pull Request 进行反馈。
