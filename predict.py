"""
离线预测脚本 - 对单张图片执行推理并保存可视化结果。
用法: python predict.py --image <图片路径> [--output <输出路径>]
"""
import argparse
import math
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from models.darknet import Darknet

# --- 相机内参（与 app.py 保持一致）---
FX, FY = 400.32, 400.32
CX, CY = 400.0, 300.0
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float32)
DIST_COEFFS = np.zeros((4, 1))

BOX_EDGES = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7),
]

transform = transforms.Compose([
    transforms.Resize((600, 800)),
    transforms.ToTensor(),
])


def nms(predictions, dist_threshold=60.0):
    if not predictions:
        return []
    predictions.sort(key=lambda x: x['confidence'], reverse=True)
    keep = []
    while predictions:
        best = predictions.pop(0)
        keep.append(best)
        predictions = [
            p for p in predictions
            if math.hypot(best['center_2d'][0] - p['center_2d'][0],
                          best['center_2d'][1] - p['center_2d'][1]) > dist_threshold
        ]
    return keep


def run_inference(model, image_pil, device, conf_threshold=0.60):
    image_tensor = transform(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)  # [1, 22, 18, 25]

    conf_scores = torch.sigmoid(output[0, 21, :, :]).cpu().numpy()
    output_np   = output.cpu().numpy()[0]

    candidates = []
    for gy, gx in zip(*np.where(conf_scores > conf_threshold)):
        vec        = output_np[:, gy, gx]
        confidence = float(conf_scores[gy, gx])

        u_c, v_c       = float(vec[0]), float(vec[1])
        vertex_offsets = vec[2:18]
        w, h, l        = np.abs(vec[18]), np.abs(vec[19]), np.abs(vec[20])
        w, h, l        = max(w, 0.01), max(h, 0.01), max(l, 0.01)

        # 物理尺寸合理性过滤（10~150 mm）
        if not (10 < w*1000 < 150 and 10 < h*1000 < 150 and 10 < l*1000 < 150):
            continue

        # 构建 2D 关键点
        points_2d = [[u_c + vertex_offsets[2*k], v_c + vertex_offsets[2*k+1]] for k in range(8)]
        points_2d.append([u_c, v_c])
        points_2d = np.array(points_2d, dtype=np.float32)

        # 3D 参考模型
        x_c = [w/2,-w/2,-w/2, w/2, w/2,-w/2,-w/2, w/2, 0]
        y_c = [h/2, h/2, h/2, h/2,-h/2,-h/2,-h/2,-h/2, 0]
        z_c = [l/2, l/2,-l/2,-l/2, l/2, l/2,-l/2,-l/2, 0]
        points_3d = np.array([x_c, y_c, z_c], dtype=np.float32).T

        success, rvec, tvec = cv2.solvePnP(points_3d, points_2d, K, DIST_COEFFS,
                                           flags=cv2.SOLVEPNP_EPNP)
        if not success:
            continue

        x_pos, y_pos, z_pos = tvec.flatten()
        z_mm = z_pos * 1000
        if not (50 < z_mm < 2000):
            continue

        candidates.append({
            "confidence": confidence,
            "center_2d":  [u_c, v_c],
            "points_2d":  points_2d.tolist(),
            "position":   {"x": float(x_pos*1000), "y": float(y_pos*1000), "z": float(z_mm)},
            "dimensions": {"l": float(l*1000), "w": float(w*1000), "h": float(h*1000)},
        })

    return nms(candidates)


def draw_results(cv_image, targets):
    for i, t in enumerate(targets):
        pts = np.array(t["points_2d"], dtype=np.float32)
        color = [(0,255,0),(0,200,255),(255,160,0),(255,0,180),(80,220,255)][i % 5]

        for s, e in BOX_EDGES:
            p1 = (int(np.clip(pts[s,0],-10000,10000)), int(np.clip(pts[s,1],-10000,10000)))
            p2 = (int(np.clip(pts[e,0],-10000,10000)), int(np.clip(pts[e,1],-10000,10000)))
            cv2.line(cv_image, p1, p2, color, 2)

        cx, cy = int(pts[8,0]), int(pts[8,1])
        cv2.circle(cv_image, (cx, cy), 4, (0,0,255), -1)

        pos = t["position"]
        dim = t["dimensions"]
        label = (f"#{i+1} conf:{t['confidence']:.2f} "
                 f"z:{pos['z']:.0f}mm "
                 f"l:{dim['l']:.0f} w:{dim['w']:.0f} h:{dim['h']:.0f}mm")
        cv2.putText(cv_image, label, (cx+6, cy-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    return cv_image


def main():
    parser = argparse.ArgumentParser(description="草莓 6D 位姿离线预测")
    parser.add_argument("--image",  required=True, help="输入图片路径")
    parser.add_argument("--output", default="result.jpg", help="输出图片路径")
    parser.add_argument("--weights", default="darknet_strawberry_checkpoint.pth", help="权重文件路径")
    parser.add_argument("--conf",   type=float, default=0.60, help="置信度阈值")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 加载模型
    model = Darknet().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device), strict=False)
    model.eval()
    print(f"Weights loaded from: {args.weights}")

    # 读取图片
    pil_image = Image.open(args.image).convert("RGB")
    cv_image  = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    cv_image  = cv2.resize(cv_image, (800, 600))

    # 推理
    targets = run_inference(model, pil_image, device, conf_threshold=args.conf)
    print(f"检测到 {len(targets)} 个草莓目标")

    for i, t in enumerate(targets):
        pos = t["position"]
        dim = t["dimensions"]
        print(f"  #{i+1}: conf={t['confidence']:.3f} | "
              f"位置 x={pos['x']:.1f} y={pos['y']:.1f} z={pos['z']:.1f} mm | "
              f"尺寸 l={dim['l']:.1f} w={dim['w']:.1f} h={dim['h']:.1f} mm")

    # 可视化并保存
    result_img = draw_results(cv_image, targets)
    cv2.imwrite(args.output, result_img)
    print(f"结果已保存至: {args.output}")


if __name__ == "__main__":
    main()
