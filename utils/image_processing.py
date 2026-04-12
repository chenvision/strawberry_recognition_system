import cv2
import torch
import numpy as np
import math

# --- 相机内参 ---
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

def run_inference(image, model, transform, device):
    """
    Kaggle-Compatible Inference: 适配 kaggle_train.py 的绝对像素回归逻辑
    """
    # 1. 预处理
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 2. 模型推理 [1, 22, 18, 25]
    with torch.no_grad():
        output = model(image_tensor)
        
    # 3. 置信度解码 (Channel 21)
    conf_logits = output[0, 21, :, :]
    conf_scores = torch.sigmoid(conf_logits).cpu().numpy()
    
    # 调试打印: 找出全图最高置信度
    max_conf = conf_scores.max()
    print(f"[DEBUG] Global Max Confidence: {max_conf:.4f}")
    
    output_np = output.cpu().numpy()[0] # [22, 18, 25]

    # 设置一个较低的初始阈值
    CONF_THRESHOLD = 0.20
    candidates = []
    
    # 找到所有置信度 > 阈值的网格
    valid_indices = np.where(conf_scores > CONF_THRESHOLD)
    print(f"[DEBUG] Found {len(valid_indices[0])} candidates with confidence > {CONF_THRESHOLD}")
    
    for i in range(len(valid_indices[0])):
        gy, gx = valid_indices[0][i], valid_indices[1][i]
        vec = output_np[:, gy, gx]
        confidence = float(conf_scores[gy, gx])
        
        # A. 中心点解码 (kaggle_train.py 存储的是绝对像素坐标)
        u_center, v_center = vec[0], vec[1]
        
        # B. 顶点解码 (kaggle_train.py 存储的是相对于中心点的绝对像素偏移)
        vertex_offsets = vec[2:18]
        pts_2d = []
        for k in range(8):
            du_abs = vertex_offsets[2*k]
            dv_abs = vertex_offsets[2*k+1]
            pts_2d.append([u_center + du_abs, v_center + dv_abs])
        
        # 第 9 个点是中心点
        pts_2d.append([u_center, v_center])
        pts_2d_np = np.array(pts_2d, dtype=np.float32)

        # C. 尺寸解码 (mm) - 使用固定的默认值进行诊断
        # w, h, l = vec[18], vec[19], vec[20]
        # w, h, l = abs(w), abs(h), abs(l)
        w, h, l = 30.0, 30.0, 30.0

        # D. PnP 位姿解算
        # 3D 物体坐标系下的 8 个顶点 (与 kaggle_train.py 映射逻辑一致)
        pts_3d = np.array([
            [w/2, h/2, l/2], [-w/2, h/2, l/2], [-w/2, -h/2, l/2], [w/2, -h/2, l/2],
            [w/2, h/2, -l/2], [-w/2, h/2, -l/2], [-w/2, -h/2, -l/2], [w/2, -h/2, -l/2]
        ], dtype=np.float32)
        
        # 选取 2D 的前 8 个点
        success, rvec, tvec = cv2.solvePnP(pts_3d, pts_2d_np[:8], K, DIST_COEFFS, flags=cv2.SOLVEPNP_EPNP)
        print(f"[DEBUG] Candidate {i}: solvePnP success = {success}")
        
        if success:
            x_mm, y_mm, z_mm = tvec.flatten()
            print(f"[DEBUG] Candidate {i}: z_mm = {z_mm:.2f} mm")
            
            # 过滤异常深度 (草莓通常在 10cm 到 1.5m 之间)
            if 100 < z_mm < 1500:
                candidates.append({
                    "confidence": confidence,
                    "center_2d": [float(u_center), float(v_center)],
                    "points_2d": pts_2d_np.tolist(),
                    "position": {"x": float(x_mm), "y": float(y_mm), "z": float(z_mm)},
                    "dimensions": {"l": float(l), "w": float(w), "h": float(h)}
                })

    # 5. NMS 过滤
    final_results = nms(candidates, dist_threshold=60.0)
    return final_results
