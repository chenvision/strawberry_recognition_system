import cv2
import torch
import numpy as np
import uvicorn
import base64
import math
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from PIL import Image
from io import BytesIO

# 导入模型定义 (确保路径正确，models/darknet.py 存在)
from models.darknet import Darknet

# --- 1. 初始化 FastAPI 应用 ---
app = FastAPI(title="Strawberry 6D Pose Estimation API")

# --- 2. 配置 CORS 跨域 ---
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:5173",
    "http://127.0.0.1:8080",
    "http://127.0.0.1:5173",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. 加载模型与权重 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "darknet_strawberry.pth"

import os
print(f"Loading model on {DEVICE}...")
model = Darknet().to(DEVICE)

checkpoint_path = 'darknet_strawberry_checkpoint.pth'
if os.path.exists(checkpoint_path):
    print(f"!!! 正在加载模型权重: {checkpoint_path} !!!")
    try:
        # 必须使用 strict=False 以防万一，但要捕获是否成功
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE), strict=False)
        print("✅ 权重加载成功！")
    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
else:
    print("❌ 找不到权重文件，模型正在使用随机初始化的垃圾参数！！")

model.eval()

# --- 4. 定义图像预处理 ---
transform = transforms.Compose([
    transforms.Resize((600, 800)),
    transforms.ToTensor(),
])

# --- 5. 相机内参 ---
FX = 400.32
FY = 400.32
CX = 400.0
CY = 300.0

K = np.array([
    [FX, 0, CX],
    [0, FY, CY],
    [0, 0, 1]
], dtype=np.float32)

DIST_COEFFS = np.zeros((4, 1))

# --- 6. 辅助函数: NMS ---
def nms(predictions, dist_threshold=40.0):
    """
    基于欧氏距离的 NMS (Non-Maximum Suppression)
    dist_threshold: 两个目标中心点距离小于此值视为重叠
    """
    if not predictions:
        return []
    
    # 按置信度从高到低排序
    predictions.sort(key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    while predictions:
        best = predictions.pop(0)
        keep.append(best)
        
        # 移除与当前 best 距离过近的目标
        filtered_preds = []
        for p in predictions:
            # 计算欧氏距离
            dist = math.hypot(best['center_2d'][0] - p['center_2d'][0], 
                              best['center_2d'][1] - p['center_2d'][1])
            if dist > dist_threshold:
                filtered_preds.append(p)
        predictions = filtered_preds
        
    return keep

# --- 7. 核心推理函数 (抽离以便复用) ---
def run_inference(image):
    """
    执行模型推理，返回检测到的候选目标列表
    """
    # 预处理 [1, 3, 600, 800]
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # 加入验证打印
    print(f"--- Input Tensor Info ---")
    print(f"Shape: {image_tensor.shape}")
    print(f"Max pixel: {image_tensor.max().item():.4f}, Min pixel: {image_tensor.min().item():.4f}")
    print(f"Mean pixel: {image_tensor.mean().item():.4f}")
    print(f"-------------------------")
    
    # 模型推理 [1, 22, 18, 25]
    with torch.no_grad():
        output = model(image_tensor)
        # 不要急着转 numpy，先在 Tensor 上做 Sigmoid
        
    # 1. 强制 Sigmoid 激活置信度
    # output shape: [1, 22, 18, 25]
    conf_logits = output[0, 21, :, :] # [18, 25]
    conf_scores = torch.sigmoid(conf_logits).cpu().numpy() # [18, 25]
    
    output = output.cpu().numpy()[0] # [22, 18, 25]

    # 2. 提高阈值
    CONF_THRESHOLD = 0.60
    candidates = []
    
    # 找到所有置信度 > 阈值的网格索引
    valid_indices = np.where(conf_scores > CONF_THRESHOLD)
    
    # --- DEBUG INFO ---
    print(f"==== 当前图像的最高置信度: {conf_scores.max().item():.4f} ====")
    if len(valid_indices[0]) == 0:
        print(f"DEBUG: No targets found above threshold {CONF_THRESHOLD}.")
    
    for i in range(len(valid_indices[0])):
        gy, gx = valid_indices[0][i], valid_indices[1][i]
        
        # 提取该网格的 22 维向量
        vec = output[:, gy, gx]
        confidence = float(conf_scores[gy, gx])
        
        # 解析参数
        center_2d = vec[0:2] # u, v
        vertex_offsets = vec[2:18]
        dims_3d = vec[18:21] # w, h, l
        
        # 坐标防越界保护
        u_curr, v_curr = float(center_2d[0]), float(center_2d[1])
        
        # --- DEBUG MODE: 打印详细信息 ---
        if confidence > 0.6:
            print(f"🔥 发现及格目标: Grid({gy},{gx}), Score={confidence:.4f}, Raw_U={u_curr:.2f}, Raw_V={v_curr:.2f}")
        
        # ---------------------------------------------

        # 强制尺寸为正数
        w, h, l = np.abs(dims_3d[0]), np.abs(dims_3d[1]), np.abs(dims_3d[2])
        w, h, l = max(w, 0.01), max(h, 0.01), max(l, 0.01)

        # --- 3D 物理尺寸与深度安检 ---
        w_final, h_final, l_final = w * 1000, h * 1000, l * 1000
        if not (10 < w_final < 150 and 10 < h_final < 150 and 10 < l_final < 150):
            continue
        
        # PnP 解算
        points_2d = []
        for k in range(8):
            u = u_curr + vertex_offsets[2*k]
            v = v_curr + vertex_offsets[2*k+1]
            points_2d.append([u, v])
        points_2d.append([u_curr, v_curr]) # 中心点
        points_2d = np.array(points_2d, dtype=np.float32)

        # 3D 物体坐标系
        x_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2, 0]
        y_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2, 0]
        z_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2, 0]
        points_3d = np.array([x_corners, y_corners, z_corners], dtype=np.float32).T

        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(points_3d, points_2d, K, DIST_COEFFS, flags=cv2.SOLVEPNP_EPNP)
        
        if success:
            x_pos, y_pos, z_pos = tvec.flatten()
            z_final = z_pos * 1000
            
            # --- 深度必须合理 ---
            if not (50 < z_final < 2000):
                continue

            # 候选目标
            candidate = {
                "confidence": confidence,
                "center_2d": [u_curr, v_curr],
                "points_2d": points_2d.tolist(), # 保存所有顶点用于绘图
                "position": {
                    "x": float(x_pos * 1000), # mm
                    "y": float(y_pos * 1000),
                    "z": float(z_final)
                },
                "dimensions": {
                    "l": float(l_final), # mm
                    "w": float(w_final),
                    "h": float(h_final)
                }
            }
            candidates.append(candidate)

    # 3. NMS 过滤 (dist_threshold=60)
    final_results = nms(candidates, dist_threshold=60.0)
    return final_results

# --- 8. API 接口: 实时视频流预测 (保持兼容) ---
@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """
    Grid-based Multi-Object Detection & Pose Estimation
    返回 JSON 数组，包含所有检测到的目标。
    """
    try:
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        
        final_results = run_inference(image)
        
        # 为了减少传输数据量，实时接口可以移除 points_2d 详细数据，只保留 center_2d
        for res in final_results:
            if 'points_2d' in res:
                del res['points_2d']
        
        return final_results

    except Exception as e:
        print(f"Prediction Error: {e}")
        return [{"error": str(e)}]

# --- 9. API 接口: 单图静态分析 (新增) ---
@app.post("/api/analyze_image")
async def analyze_image(file: UploadFile = File(...)):
    """
    单图上传分析接口。
    返回: { "targets": [...], "result_image": "base64_string" }
    """
    try:
        # 读取图片
        image_data = await file.read()
        pil_image = Image.open(BytesIO(image_data)).convert("RGB")
        
        # 执行推理
        targets = run_inference(pil_image)
        
        # 转换 OpenCV 格式以便绘图 (RGB -> BGR)
        # 注意：推理时我们 resize 到了 800x600，所以绘图也要在 800x600 上进行
        # 或者将坐标映射回原图。为了简单，我们将原图 resize 到 800x600 返回。
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # --- FIX: 强制将底图 Resize 到与推理一致的尺寸 (800x600) ---
        # 否则预测的坐标 (基于800x600) 画在原图上会错位
        cv_image = cv2.resize(cv_image, (800, 600))
        # --------------------------------------------------------
        
        # 绘制 3D 边界框
        for target in targets:
            # 4. 安全的 OpenCV 画框: 坐标转 Int
            pts = np.array(target['points_2d'], dtype=np.int32)
            vertices = pts[:8]
            center = pts[8]
            
            # --- FIX: 画图防越界保护 (限制在画布范围内) ---
            # 虽然 cv2.line 支持越界坐标，但为了防止极端值溢出 int32，这里做个软截断
            center_plot = (
                int(max(-10000, min(10000, center[0]))),
                int(max(-10000, min(10000, center[1])))
            )
            # -------------------------------------------
            
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7)
            ]
            
            color = (0, 255, 0) # Green
            for start, end in edges:
                pt1 = tuple(vertices[start])
                pt2 = tuple(vertices[end])
                # 同样的软截断保护
                pt1_safe = (int(max(-10000, min(10000, pt1[0]))), int(max(-10000, min(10000, pt1[1]))))
                pt2_safe = (int(max(-10000, min(10000, pt2[0]))), int(max(-10000, min(10000, pt2[1]))))
                cv2.line(cv_image, pt1_safe, pt2_safe, color, 2)
            
            cv2.circle(cv_image, center_plot, 4, (0, 0, 255), -1)
            
            conf_text = f"{target['confidence']:.2f}"
            cv2.putText(cv_image, conf_text, (center_plot[0]+5, center_plot[1]-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 编码结果图像为 Base64
        _, buffer = cv2.imencode('.jpg', cv_image)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        base64_string = f"data:image/jpeg;base64,{jpg_as_text}"
        
        # 清理 targets 中的 points_2d 以精简 JSON
        for t in targets:
            if 'points_2d' in t:
                del t['points_2d']
        
        return {
            "targets": targets,
            "result_image": base64_string
        }

    except Exception as e:
        print(f"Analysis Error: {e}")
        return {"error": str(e)}

@app.get("/")
def read_root():
    return {"message": "Strawberry Grid-based Pose Estimation API is Running!"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
