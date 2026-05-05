# 草莓三维姿态估计系统：后续优化路线图

本文件记录了系统在完成基础重构后，针对实地应用及工程化提升的建议方向。

## 1. 算法鲁棒性优化 (Domain Gap Mitigation)
针对从“仿真数据集”迁移至“实地场景”的识别效果差异：
- [ ] **实地数据微调 (Fine-tuning)**：采集 100+ 张真实背景下的草莓图像进行标注，对现有模型进行微调，解决反光、阴影及叶片遮挡问题。
- [ ] **相机内参标定 (Calibration)**：使用棋盘格对实地摄像头进行标定，更新后端推理中的 `K` 矩阵，确保 6D 位姿（X, Y, Z）的物理精度。
- [ ] **数据增强策略**：在训练 Pipeline 中加入运动模糊 (Motion Blur)、传感器噪点 (Gaussian Noise) 及随机对比度增强。

## 2. 前端架构进阶 (Frontend Refinement)
基于当前的 Composable 重构进一步演进：
- [ ] **样式变量解耦**：将 `StrawberryDashboard.vue` 中冗长的 CSS 提取至 `src/assets/styles/theme.css`。
- [ ] **状态管理引入**：若业务逻辑继续增加（如多设备切换），考虑引入 Pinia 管理后端连接状态及全局日志。
- [ ] **JSDoc/TS 类型化**：在 `services` 和 `composables` 中引入 JSDoc 或逐步迁移至 TypeScript，提升 API 响应的确定性。

## 3. 性能基准与稳定性 (Performance & Stability)
- [ ] **请求频率自适应**：根据后端处理耗时动态调整 `INFERENCE_INTERVAL_MS`，避免在低配机器上产生请求堆积。
- [ ] **Web Worker 离屏渲染**：考虑将 Canvas 绘图与 Blob 转换移至 Web Worker，彻底解放 UI 主线程。
- [ ] **断线自动重连**：增强 `useCamera` 的鲁棒性，在 USB 摄像头意外拔插时实现无感重连。

## 4. 工程化与部署 (DevOps)
- [ ] **环境变量配置**：通过 `.env.development` 和 `.env.production` 彻底消除 `apiBase.js` 中的硬编码。
- [ ] **Docker 容器化**：封装前后端运行环境，解决跨平台（Windows/Linux）运行时的依赖冲突。
- [ ] **自动化测试覆盖**：增加对 Composable 的单元测试，特别是针对 `AbortController` 取消逻辑的边界测试。

---
*更新日期：2026-04-23*
*记录人：Gemini CLI*
