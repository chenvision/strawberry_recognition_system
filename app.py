import cv2
import torch
import numpy as np
import uvicorn
import base64
import math
import csv
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from torchvision import transforms
from PIL import Image
from io import BytesIO

# 瀵煎叆妯″瀷瀹氫箟 (纭繚璺緞姝ｇ‘锛宮odels/darknet.py 瀛樺湪)
from models.darknet import Darknet

# --- 1. 鍒濆鍖?FastAPI 搴旂敤 ---
app = FastAPI(title="Strawberry 6D Pose Estimation API")

# --- 2. 閰嶇疆 CORS 璺ㄥ煙 ---
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

PROJECT_ROOT = Path(__file__).resolve().parent
DEMO_FRAME_DIR = PROJECT_ROOT / "data" / "Straw6D_Raw" / "colored_maps"
DEMO_BOX_DIR = PROJECT_ROOT / "data" / "Straw6D_Raw" / "boxes"

if DEMO_FRAME_DIR.exists():
    app.mount("/demo-data", StaticFiles(directory=str(DEMO_FRAME_DIR)), name="demo-data")

# --- 3. 鍔犺浇妯″瀷涓庢潈閲?---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "darknet_strawberry.pth"

import os
print(f"Loading model on {DEVICE}...")
model = Darknet().to(DEVICE)

checkpoint_path = 'darknet_strawberry_checkpoint.pth'
if os.path.exists(checkpoint_path):
    print(f"Loading model weights from: {checkpoint_path}")
    try:
        # 蹇呴』浣跨敤 strict=False 浠ラ槻涓囦竴锛屼絾瑕佹崟鑾锋槸鍚︽垚鍔?
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE), strict=False)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Model weight loading failed: {e}")
else:
    print("Model weights not found, using randomly initialized parameters.")

model.eval()

# --- 4. 瀹氫箟鍥惧儚棰勫鐞?---
transform = transforms.Compose([
    transforms.Resize((600, 800)),
    transforms.ToTensor(),
])

# --- 5. 鐩告満鍐呭弬 ---
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
BOX_DRAW_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]
DEMO_COLORS = [
    (0, 255, 0),
    (0, 200, 255),
    (255, 160, 0),
    (255, 0, 180),
    (80, 220, 255),
]

# --- 6. 杈呭姪鍑芥暟: NMS ---
def nms(predictions, dist_threshold=40.0):
    """
    鍩轰簬娆ф皬璺濈鐨?NMS (Non-Maximum Suppression)
    dist_threshold: 涓や釜鐩爣涓績鐐硅窛绂诲皬浜庢鍊艰涓洪噸鍙?
    """
    if not predictions:
        return []
    
    # 鎸夌疆淇″害浠庨珮鍒颁綆鎺掑簭
    predictions.sort(key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    while predictions:
        best = predictions.pop(0)
        keep.append(best)
        
        # 绉婚櫎涓庡綋鍓?best 璺濈杩囪繎鐨勭洰鏍?
        filtered_preds = []
        for p in predictions:
            # 璁＄畻娆ф皬璺濈
            dist = math.hypot(best['center_2d'][0] - p['center_2d'][0], 
                              best['center_2d'][1] - p['center_2d'][1])
            if dist > dist_threshold:
                filtered_preds.append(p)
        predictions = filtered_preds
        
    return keep


def euler_to_rotation(theta):
    rx, ry, rz = theta

    rot_x = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ], dtype=np.float32)
    rot_y = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ], dtype=np.float32)
    rot_z = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ], dtype=np.float32)

    return rot_z @ rot_y @ rot_x


def clamp_point(point, limit=10000):
    return (
        int(max(-limit, min(limit, point[0]))),
        int(max(-limit, min(limit, point[1]))),
    )


def draw_box_overlay(cv_image, points_2d, color, label_text, axis_2d=None):
    pts = np.asarray(points_2d, dtype=np.float32)
    vertices = pts[:8]
    center = pts[8]
    center_plot = clamp_point(center)

    # 1. 绘制立体框的 12 条边
    for start, end in BOX_DRAW_EDGES:
        pt1 = clamp_point(vertices[start])
        pt2 = clamp_point(vertices[end])
        cv2.line(cv_image, pt1, pt2, color, 2)

    # 2. 绘制 RGB 三色坐标轴 (如果存在)
    if axis_2d is not None:
        # axis_2d 顺序: [X_tip, Y_tip, Z_tip, Origin]
        origin = clamp_point(axis_2d[3])
        x_tip = clamp_point(axis_2d[0])
        y_tip = clamp_point(axis_2d[1])
        z_tip = clamp_point(axis_2d[2])
        
        # OpenCV 是 BGR: Red(0,0,255), Green(0,255,0), Blue(255,0,0)
        cv2.line(cv_image, origin, x_tip, (0, 0, 255), 3) # X - 红
        cv2.line(cv_image, origin, y_tip, (0, 255, 0), 3) # Y - 绿
        cv2.line(cv_image, origin, z_tip, (255, 0, 0), 3) # Z - 蓝

    # 3. 移除原本的中心点圆圈和文字标签，保持画面纯净
    # cv2.circle(cv_image, center_plot, 4, (0, 0, 255), -1)
    # cv2.putText(
    #     cv_image,
    #     label_text,
    #     (center_plot[0] + 6, center_plot[1] - 6),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.5,
    #     (255, 255, 255),
    #     1,
    #     cv2.LINE_AA,
    # )


def load_demo_targets(frame_name):
    frame_name = Path(frame_name).name
    csv_path = DEMO_BOX_DIR / f"{Path(frame_name).stem}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Annotation file not found for demo frame: {frame_name}")

    targets = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            x = float(row["x"])
            y = float(row["y"])
            z = float(row["z"])
            w = abs(float(row["w"]))
            h = abs(float(row["h"]))
            l = abs(float(row["l"]))
            roll = float(row["roll"])
            pitch = float(row["pitch"])
            yaw = float(row["yaw"])

            rotation = euler_to_rotation([roll, pitch, yaw])
            x_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2, 0]
            y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2, 0]
            z_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, 0]
            corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
            corners_3d = rotation @ corners
            corners_3d += np.array([x, y, z], dtype=np.float32).reshape(3, 1)
            corners_3d = corners_3d.transpose(1, 0)

            pts_2d = []
            valid_projection = True
            for point in corners_3d:
                depth = -float(point[2])
                if depth <= 1e-6:
                    valid_projection = False
                    break
                u = FX * float(point[0]) / depth + CX
                v = FY * float(-point[1]) / depth + CY
                pts_2d.append([u, v])

            if not valid_projection:
                continue

            pts_array = np.asarray(pts_2d, dtype=np.float32)
            if not np.any(
                (pts_array[:, 0] >= -100) & (pts_array[:, 0] <= 900) &
                (pts_array[:, 1] >= -100) & (pts_array[:, 1] <= 700)
            ):
                continue

            targets.append({
                "label": int(float(row["label"])),
                "confidence": 1.0,
                "center_2d": [float(pts_array[8, 0]), float(pts_array[8, 1])],
                "points_2d": pts_array.tolist(),
                "position": {
                    "x": float(x * 1000),
                    "y": float(y * 1000),
                    "z": float(abs(z) * 1000),
                },
                "dimensions": {
                    "l": float(l * 1000),
                    "w": float(w * 1000),
                    "h": float(h * 1000),
                }
            })

    targets.sort(key=lambda target: target["position"]["z"])
    return targets


def encode_image_to_base64(cv_image):
    success, buffer = cv2.imencode(".jpg", cv_image)
    if not success:
        raise RuntimeError("Failed to encode result image.")
    jpg_as_text = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{jpg_as_text}"


def strip_points_2d(targets):
    cleaned_targets = []
    for target in targets:
        cleaned_target = dict(target)
        # 保持 points_2d 以便前端绘制（如果需要的话，实时接口现在不再删除它）
        cleaned_targets.append(cleaned_target)
    return cleaned_targets


def build_demo_frame_result(frame_name):
    frame_name = Path(frame_name).name
    image_path = DEMO_FRAME_DIR / frame_name
    if not image_path.exists():
        raise FileNotFoundError(f"Demo frame not found: {frame_name}")

    cv_image = cv2.imread(str(image_path))
    if cv_image is None:
        raise RuntimeError(f"Failed to load demo frame: {frame_name}")

    targets = load_demo_targets(frame_name)
    for index, target in enumerate(targets):
        color = DEMO_COLORS[index % len(DEMO_COLORS)]
        draw_box_overlay(cv_image, target["points_2d"], color, f"GT #{index + 1}")

    return {
        "frame": frame_name,
        "mode": "ground_truth_replay",
        "targets": strip_points_2d(targets),
        "result_image": encode_image_to_base64(cv_image),
    }

# --- 7. 鏍稿績鎺ㄧ悊鍑芥暟 (鎶界浠ヤ究澶嶇敤) ---
def run_inference(image):
    """
    鎵ц妯″瀷鎺ㄧ悊锛岃繑鍥炴娴嬪埌鐨勫€欓€夌洰鏍囧垪琛?
    """
    # 棰勫鐞?[1, 3, 600, 800]
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # 鍔犲叆楠岃瘉鎵撳嵃
    print(f"--- Input Tensor Info ---")
    print(f"Shape: {image_tensor.shape}")
    print(f"Max pixel: {image_tensor.max().item():.4f}, Min pixel: {image_tensor.min().item():.4f}")
    print(f"Mean pixel: {image_tensor.mean().item():.4f}")
    print(f"-------------------------")
    
    # 妯″瀷鎺ㄧ悊 [1, 22, 18, 25]
    with torch.no_grad():
        output = model(image_tensor)
        # 涓嶈鎬ョ潃杞?numpy锛屽厛鍦?Tensor 涓婂仛 Sigmoid
        
    # 1. 寮哄埗 Sigmoid 婵€娲荤疆淇″害
    # output shape: [1, 22, 18, 25]
    conf_logits = output[0, 21, :, :] # [18, 25]
    conf_scores = torch.sigmoid(conf_logits).cpu().numpy() # [18, 25]
    
    output = output.cpu().numpy()[0] # [22, 18, 25]

    # 2. 鎻愰珮闃堝€?
    # 2. 降低阈值并引入 Fallback 机制
    CONF_THRESHOLD = 0.35
    candidates = []
    valid_indices = np.where(conf_scores > CONF_THRESHOLD)
    for j in range(len(valid_indices[0])):
        gy, gx = valid_indices[0][j], valid_indices[1][j]
        vec = output[:, gy, gx]
        confidence = float(conf_scores[gy, gx])
        u_curr, v_curr = float(vec[0]), float(vec[1])
        vertex_offsets = vec[2:18]
        dims_3d = vec[18:21]
        w, h, l = np.abs(dims_3d[0]), np.abs(dims_3d[1]), np.abs(dims_3d[2])
        w, h, l = max(w, 0.01), max(h, 0.01), max(l, 0.01)
        w_f, h_f, l_f = w * 1000, h * 1000, l * 1000
        if not (5 < w_f < 200 and 5 < h_f < 200 and 5 < l_f < 200): continue
        pts_2d = np.array([[u_curr + vertex_offsets[2*k], v_curr + vertex_offsets[2*k+1]] for k in range(8)] + [[u_curr, v_curr]], dtype=np.float32)
        pts_3d = np.array([[w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2,0],[h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2,0],[l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2,0]], dtype=np.float32).T
        success, rvec, tvec = cv2.solvePnP(pts_3d, pts_2d, K, DIST_COEFFS, flags=cv2.SOLVEPNP_EPNP)
        if not success:
            z_fb = 0.5; tvec = np.array([[ (u_curr-K[0,2])*z_fb/K[0,0] ], [ (v_curr-K[1,2])*z_fb/K[1,1] ], [z_fb]], dtype=np.float32)
            rvec = np.zeros((3,1), dtype=np.float32); success = True
        if success:
            x_p, y_p, z_p = tvec.flatten(); z_f = z_p * 1000
            if not (50 < z_f < 3000): continue
            ax_len = min(w,h,l)*0.8; ax_3d = np.array([[ax_len,0,0],[0,ax_len,0],[0,0,ax_len],[0,0,0]], dtype=np.float32)
            ax_2d, _ = cv2.projectPoints(ax_3d, rvec, tvec, K, DIST_COEFFS)
            candidates.append({"confidence": confidence, "center_2d": [u_curr, v_curr], "points_2d": pts_2d.tolist(), "axis_2d": ax_2d.reshape(-1,2).tolist(), "position": {"x":float(x_p*1000),"y":float(y_p*1000),"z":float(z_f)}, "dimensions": {"l":float(l_f),"w":float(w_f),"h":float(h_f)}})
    # 3. NMS 过滤 (优化后的阈值)
    final_results = nms(candidates, dist_threshold=35.0)
    return final_results

# --- 8. API 鎺ュ彛: 瀹炴椂瑙嗛娴侀娴?(淇濇寔鍏煎) ---
@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """
    Grid-based Multi-Object Detection & Pose Estimation
    杩斿洖 JSON 鏁扮粍锛屽寘鍚墍鏈夋娴嬪埌鐨勭洰鏍囥€?
    """
    try:
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        
        final_results = run_inference(image)
        return final_results

    except Exception as e:
        print(f"Prediction Error: {e}")
        return [{"error": str(e)}]

# --- 9. API 鎺ュ彛: 鍗曞浘闈欐€佸垎鏋?(鏂板) ---
@app.post("/api/analyze_image")
async def analyze_image(file: UploadFile = File(...)):
    """
    鍗曞浘涓婁紶鍒嗘瀽鎺ュ彛銆?
    杩斿洖: { "targets": [...], "result_image": "base64_string" }
    """
    try:
        # 璇诲彇鍥剧墖
        image_data = await file.read()
        pil_image = Image.open(BytesIO(image_data)).convert("RGB")
        
        # 鎵ц鎺ㄧ悊
        targets = run_inference(pil_image)
        
        # 杞崲 OpenCV 鏍煎紡浠ヤ究缁樺浘 (RGB -> BGR)
        # 娉ㄦ剰锛氭帹鐞嗘椂鎴戜滑 resize 鍒颁簡 800x600锛屾墍浠ョ粯鍥句篃瑕佸湪 800x600 涓婅繘琛?
        # 鎴栬€呭皢鍧愭爣鏄犲皠鍥炲師鍥俱€備负浜嗙畝鍗曪紝鎴戜滑灏嗗師鍥?resize 鍒?800x600 杩斿洖銆?
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # --- FIX: 寮哄埗灏嗗簳鍥?Resize 鍒颁笌鎺ㄧ悊涓€鑷寸殑灏哄 (800x600) ---
        # 鍚﹀垯棰勬祴鐨勫潗鏍?(鍩轰簬800x600) 鐢诲湪鍘熷浘涓婁細閿欎綅
        cv_image = cv2.resize(cv_image, (800, 600))
        # --------------------------------------------------------
        
        # --- 恢复后端精准绘图 (作为底色) ---
        for target in targets:
            draw_box_overlay(
                cv_image, 
                target["points_2d"], 
                (0, 255, 0), # 绿框
                f"{target['confidence']:.2f}",
                axis_2d=target.get("axis_2d")
            )

        return {
            "targets": strip_points_2d(targets),
            "result_image": encode_image_to_base64(cv_image)
        }


    except Exception as e:
        print(f"Analysis Error: {e}")
        return {"error": str(e)}

@app.get("/api/demo/frames")
def demo_frames(limit: int = 120):
    if not DEMO_FRAME_DIR.exists():
        return {"frames": [], "count": 0}

    image_suffixes = {".png", ".jpg", ".jpeg"}
    frames = sorted(
        path for path in DEMO_FRAME_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in image_suffixes
        and (DEMO_BOX_DIR / f"{path.stem}.csv").exists()
    )[:max(limit, 0)]

    return {
        "count": len(frames),
        "frames": [
            {
                "name": frame.name,
                "url": f"/demo-data/{frame.name}",
            }
            for frame in frames
        ],
    }


@app.get("/api/demo/analyze_frame")
def demo_analyze_frame(name: str):
    try:
        return build_demo_frame_result(name)
    except Exception as e:
        print(f"Demo Analysis Error: {e}")
        return {"error": str(e)}

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.get("/")
def read_root():
    return {"message": "Strawberry Grid-based Pose Estimation API is Running!"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

