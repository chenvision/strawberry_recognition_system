import cv2
import torch
import numpy as np
import uvicorn
import base64
import math
import time
import asyncio
import csv
from pathlib import Path
from typing import List, Optional, Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from torchvision import transforms
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

# 导入模型定义
from models.darknet import Darknet

# --- 1. Pydantic 数据模型 ---
class Vector3(BaseModel):
    x: float
    y: float
    z: float

class EulerAngles(BaseModel):
    roll: float
    pitch: float
    yaw: float

class Dimensions(BaseModel):
    l: float
    w: float
    h: float

class StrawberryPose(BaseModel):
    confidence: float
    center_2d: List[float]
    points_2d: List[List[float]]
    axis_2d: Optional[List[List[float]]] = None
    position: Vector3
    orientation: EulerAngles # 增加朝向字段
    dimensions: Dimensions

class InferenceResponse(BaseModel):
    targets: List[StrawberryPose]
    processing_time_ms: float
    frame_id: Optional[int] = None

class AnalyzeResponse(BaseModel):
    targets: List[StrawberryPose]
    result_image: str # Base64

# --- 2. 初始化 FastAPI ---
app = FastAPI(title="Strawberry 6D Pose Full-Duplex API")

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.middleware("http")
async def log_requests(request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"Unhandled error during {request.method} {request.url}: {e}", exc_info=True)
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=500, content={"error": str(e)})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEMO_FRAME_DIR = PROJECT_ROOT / "data" / "Straw6D_Raw" / "colored_maps"
DEMO_BOX_DIR = PROJECT_ROOT / "data" / "Straw6D_Raw" / "boxes"

if DEMO_FRAME_DIR.exists():
    app.mount("/demo-data", StaticFiles(directory=str(DEMO_FRAME_DIR)), name="demo-data")

# --- 3. 模型加载与配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "darknet_strawberry_checkpoint.pth"
executor = ThreadPoolExecutor(max_workers=2)

print(f"Loading model on {DEVICE}...")
model = Darknet().to(DEVICE)
if Path(MODEL_PATH).exists():
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    print("Weights loaded.")
model.eval()

transform = transforms.Compose([
    transforms.Resize((600, 800)),
    transforms.ToTensor(),
])

# 相机内参
FX, FY = 400.32, 400.32
CX, CY = 400.0, 300.0
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float32)
DIST_COEFFS = np.zeros((4, 1))

BOX_DRAW_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

DEMO_COLORS = [(0, 255, 0), (0, 200, 255), (255, 160, 0), (255, 0, 180), (80, 220, 255)]

# --- 4. 辅助函数 ---
def euler_to_rotation(theta):
    rx, ry, rz = theta
    rot_x = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]], dtype=np.float32)
    rot_y = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]], dtype=np.float32)
    rot_z = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]], dtype=np.float32)
    return rot_z @ rot_y @ rot_x

def rotation_to_euler(rmat):
    sy = math.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(rmat[2, 1], rmat[2, 2])
        y = math.atan2(-rmat[2, 0], sy)
        z = math.atan2(rmat[1, 0], rmat[0, 0])
    else:
        x = math.atan2(-rmat[1, 2], rmat[1, 1])
        y = math.atan2(-rmat[2, 0], sy)
        z = 0
    return x, y, z

def clamp_point(point, limit=10000):
    return (int(max(-limit, min(limit, point[0]))), int(max(-limit, min(limit, point[1]))))

def draw_box_overlay(cv_image, points_2d, color, axis_2d=None):
    pts = np.asarray(points_2d, dtype=np.float32)
    vertices = pts[:8]
    for start, end in BOX_DRAW_EDGES:
        cv2.line(cv_image, clamp_point(vertices[start]), clamp_point(vertices[end]), color, 2)
    if axis_2d is not None:
        origin = clamp_point(axis_2d[3]); x_tip = clamp_point(axis_2d[0]); y_tip = clamp_point(axis_2d[1]); z_tip = clamp_point(axis_2d[2])
        cv2.line(cv_image, origin, x_tip, (0, 0, 255), 3)
        cv2.line(cv_image, origin, y_tip, (0, 255, 0), 3)
        cv2.line(cv_image, origin, z_tip, (255, 0, 0), 3)

def encode_image_to_base64(cv_image):
    _, buffer = cv2.imencode(".jpg", cv_image)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

def load_demo_targets(frame_name):
    frame_name = Path(frame_name).name
    csv_path = DEMO_BOX_DIR / f"{Path(frame_name).stem}.csv"
    if not csv_path.exists(): raise FileNotFoundError(f"Annotation not found: {frame_name}")
    targets = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            x, y, z = float(row["x"]), float(row["y"]), float(row["z"])
            w, h, l = abs(float(row["w"])), abs(float(row["h"])), abs(float(row["l"]))
            roll, pitch, yaw = float(row["roll"]), float(row["pitch"]), float(row["yaw"])
            rotation = euler_to_rotation([roll, pitch, yaw])
            corners_3d = rotation @ np.array([[w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2,0],[h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2,0],[l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2,0]], dtype=np.float32)
            corners_3d += np.array([x, y, z], dtype=np.float32).reshape(3, 1)
            pts_2d = []
            for point in corners_3d.T:
                depth = -float(point[2])
                if depth <= 1e-6: continue
                pts_2d.append([FX * float(point[0]) / depth + CX, FY * float(-point[1]) / depth + CY])
            if len(pts_2d) < 9: continue
            targets.append({
                "confidence": 1.0, "center_2d": pts_2d[8], "points_2d": pts_2d,
                "position": {"x": x*1000, "y": y*1000, "z": abs(z)*1000},
                "orientation": {"roll": math.degrees(roll), "pitch": math.degrees(pitch), "yaw": math.degrees(yaw)},
                "dimensions": {"l": l*1000, "w": w*1000, "h": h*1000}
            })
    return targets

# --- 5. 核心推理逻辑 ---
def sync_run_inference(pil_image: Image.Image) -> List[dict]:
    image_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(image_tensor)
    conf_scores = torch.sigmoid(output[0, 21, :, :]).cpu().numpy()
    output_np = output.cpu().numpy()[0]

    candidates = []
    valid_indices = np.where(conf_scores > 0.35)
    for gy, gx in zip(*valid_indices):
        vec = output_np[:, gy, gx]
        u_curr, v_curr = float(vec[0]), float(vec[1])
        vertex_offsets = vec[2:18]; dims_3d = vec[18:21]
        w, h, l = np.abs(dims_3d[0]), np.abs(dims_3d[1]), np.abs(dims_3d[2])
        w_f, h_f, l_f = w * 1000, h * 1000, l * 1000
        if not (5 < w_f < 200 and 5 < h_f < 200 and 5 < l_f < 200): continue
        pts_2d = [[u_curr + vertex_offsets[2*k], v_curr + vertex_offsets[2*k+1]] for k in range(8)]
        pts_2d.append([u_curr, v_curr])
        pts_2d = np.array(pts_2d, dtype=np.float32)
        pts_3d = np.array([[w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2,0],[h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2,0],[l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2,0]], dtype=np.float32).T
        success, rvec, tvec = cv2.solvePnP(pts_3d, pts_2d, K, DIST_COEFFS, flags=cv2.SOLVEPNP_EPNP)
        if success:
            x_p, y_p, z_p = tvec.flatten(); z_f = z_p * 1000
            if not (50 < z_f < 3000): continue
            
            # 计算欧拉角
            rmat, _ = cv2.Rodrigues(rvec)
            rx, ry, rz = rotation_to_euler(rmat)
            
            ax_len = min(w,h,l)*0.8; ax_3d = np.array([[ax_len,0,0],[0,ax_len,0],[0,0,ax_len],[0,0,0]], dtype=np.float32)
            ax_2d, _ = cv2.projectPoints(ax_3d, rvec, tvec, K, DIST_COEFFS)
            candidates.append({
                "confidence": float(conf_scores[gy, gx]), "center_2d": [u_curr, v_curr], "points_2d": pts_2d.tolist(),
                "axis_2d": ax_2d.reshape(-1,2).tolist(), 
                "position": {"x": float(x_p*1000), "y": float(y_p*1000), "z": float(z_f)},
                "orientation": {"roll": math.degrees(rx), "pitch": math.degrees(ry), "yaw": math.degrees(rz)},
                "dimensions": {"l": float(l_f), "w": float(w_f), "h": float(h_f)}
            })
    candidates.sort(key=lambda x: x['confidence'], reverse=True)
    final = []
    for c in candidates:
        if any(math.hypot(c['center_2d'][0]-f['center_2d'][0], c['center_2d'][1]-f['center_2d'][1]) < 35 for f in final): continue
        final.append(c)
    return final

# --- 6. WebSocket 端点 ---
@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()
    processing = False
    try:
        while True:
            data = await websocket.receive()
            if "bytes" in data: img_bytes = data["bytes"]
            elif "text" in data:
                try: img_bytes = base64.b64decode(data["text"].split(",")[1])
                except: continue
            else: continue

            if processing: continue
            processing = True
            start_time = time.time()
            try:
                pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
                loop = asyncio.get_event_loop()
                targets = await loop.run_in_executor(executor, sync_run_inference, pil_img)
                response = InferenceResponse(targets=targets, processing_time_ms=(time.time()-start_time)*1000)
                await websocket.send_text(response.json())
            except Exception as e: await websocket.send_json({"error": str(e)})
            finally: processing = False
    except WebSocketDisconnect: pass

# --- 7. HTTP API 端点 ---
@app.post("/api/predict", response_model=List[StrawberryPose])
async def predict_api(file: UploadFile = File(...)):
    img_bytes = await file.read(); pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
    loop = asyncio.get_event_loop(); results = await loop.run_in_executor(executor, sync_run_inference, pil_img)
    return results

@app.post("/api/analyze_image", response_model=AnalyzeResponse)
async def analyze_image_api(file: UploadFile = File(...)):
    img_bytes = await file.read(); pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
    loop = asyncio.get_event_loop(); targets = await loop.run_in_executor(executor, sync_run_inference, pil_img)
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR); cv_img = cv2.resize(cv_img, (800, 600))
    for t in targets: draw_box_overlay(cv_img, t["points_2d"], (0, 255, 0), axis_2d=t.get("axis_2d"))
    return AnalyzeResponse(targets=targets, result_image=encode_image_to_base64(cv_img))

@app.get("/api/demo/frames")
def list_demo_frames(limit: int = 200):
    if not DEMO_FRAME_DIR.exists(): return {"frames": []}
    image_suffixes = {".png", ".jpg", ".jpeg"}
    frames = []
    for p in DEMO_FRAME_DIR.iterdir():
        if p.suffix.lower() in image_suffixes:
            csv_path = DEMO_BOX_DIR / f"{p.stem}.csv"
            if csv_path.exists():
                frames.append(p.name)
    
    frames.sort()
    return {"frames": [{"name": f, "url": f"/demo-data/{f}"} for f in frames[:limit]]}

@app.get("/api/demo/analyze_frame")
def demo_analyze_frame(name: str):
    try:
        image_path = DEMO_FRAME_DIR / name
        if not image_path.exists(): return {"error": "Frame not found"}
        cv_img = cv2.imread(str(image_path))
        targets = load_demo_targets(name)
        for i, t in enumerate(targets): draw_box_overlay(cv_img, t["points_2d"], DEMO_COLORS[i % len(DEMO_COLORS)])
        return {"frame": name, "mode": "ground_truth_replay", "targets": targets, "result_image": encode_image_to_base64(cv_img)}
    except Exception as e: return {"error": str(e)}

@app.get("/api/health")
def health(): return {"status": "ok", "device": str(DEVICE)}

@app.get("/")
def root(): return {"message": "Strawberry 6D Pose Async API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
