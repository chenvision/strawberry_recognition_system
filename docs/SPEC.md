# 草莓 6D 位姿估计系统 - 技术规格说明书 (SPEC.md)

## 1. 硬件与环境参数
*   **相机内参 (Camera Intrinsics):**
    ```python
    K = [[400.32, 0, 400],
         [0, 400.32, 300],
         [0, 0, 1]]
    ```
*   **输入图像分辨率:** 800 x 600 (RGB)
*   **推理设备:** NVIDIA GPU (CUDA) / CPU (Fallback)

## 2. 模型输出张量定义 (22-Dimensional Vector)
输出特征图尺寸: $18 \times 25$ (下采样 32 倍)。每个网格包含一个 22 维向量 $\mathbf{v}$：

| 索引 | 符号 | 描述 |
| :--- | :--- | :--- |
| 0, 1 | $u_c, v_c$ | 目标中心点在当前网格内的相对偏移 (0~1) |
| 2-17 | $\Delta u_{1-8}, \Delta v_{1-8}$ | 3D 边界框 8 个顶点相对于中心点的 2D 像素偏移 |
| 18-20 | $l, w, h$ | 草莓的三轴物理尺寸估计值 (mm) |
| 21 | $s$ | 目标存在置信度得分 (Confidence Score) |

## 3. 核心算法选型
*   **Backbone:** 改进型 Darknet-19 (轻量化、亚像素特征提取)
*   **Neck:** Transformer Encoder (1层, 8头自注意力, 全局几何约束)
*   **Head:** 单阶段空间参数回归头 (1x1 Conv)
*   **位姿解算:** EPnP (Efficient Perspective-n-Point) 算法
*   **损失函数:** Smooth-L1 (回归) + BCE (置信度)
