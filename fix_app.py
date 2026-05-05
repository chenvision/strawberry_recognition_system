
import sys

with open('app.py', 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

new_lines = []
skip = False
for i, line in enumerate(lines):
    if 'CONF_THRESHOLD = 0.60' in line:
        new_lines.append('    # 2. 降低阈值并引入 Fallback 机制\n')
        new_lines.append('    CONF_THRESHOLD = 0.35\n')
        new_lines.append('    candidates = []\n')
        new_lines.append('    valid_indices = np.where(conf_scores > CONF_THRESHOLD)\n')
        new_lines.append('    for j in range(len(valid_indices[0])):\n')
        new_lines.append('        gy, gx = valid_indices[0][j], valid_indices[1][j]\n')
        new_lines.append('        vec = output[:, gy, gx]\n')
        new_lines.append('        confidence = float(conf_scores[gy, gx])\n')
        new_lines.append('        u_curr, v_curr = float(vec[0]), float(vec[1])\n')
        new_lines.append('        vertex_offsets = vec[2:18]\n')
        new_lines.append('        dims_3d = vec[18:21]\n')
        new_lines.append('        w, h, l = np.abs(dims_3d[0]), np.abs(dims_3d[1]), np.abs(dims_3d[2])\n')
        new_lines.append('        w, h, l = max(w, 0.01), max(h, 0.01), max(l, 0.01)\n')
        new_lines.append('        w_f, h_f, l_f = w * 1000, h * 1000, l * 1000\n')
        new_lines.append('        if not (5 < w_f < 200 and 5 < h_f < 200 and 5 < l_f < 200): continue\n')
        new_lines.append('        pts_2d = np.array([[u_curr + vertex_offsets[2*k], v_curr + vertex_offsets[2*k+1]] for k in range(8)] + [[u_curr, v_curr]], dtype=np.float32)\n')
        new_lines.append('        pts_3d = np.array([[w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2,0],[h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2,0],[l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2,0]], dtype=np.float32).T\n')
        new_lines.append('        success, rvec, tvec = cv2.solvePnP(pts_3d, pts_2d, K, DIST_COEFFS, flags=cv2.SOLVEPNP_EPNP)\n')
        new_lines.append('        if not success:\n')
        new_lines.append('            z_fb = 0.5; tvec = np.array([[ (u_curr-K[0,2])*z_fb/K[0,0] ], [ (v_curr-K[1,2])*z_fb/K[1,1] ], [z_fb]], dtype=np.float32)\n')
        new_lines.append('            rvec = np.zeros((3,1), dtype=np.float32); success = True\n')
        new_lines.append('        if success:\n')
        new_lines.append('            x_p, y_p, z_p = tvec.flatten(); z_f = z_p * 1000\n')
        new_lines.append('            if not (50 < z_f < 3000): continue\n')
        new_lines.append('            ax_len = min(w,h,l)*0.8; ax_3d = np.array([[ax_len,0,0],[0,ax_len,0],[0,0,ax_len],[0,0,0]], dtype=np.float32)\n')
        new_lines.append('            ax_2d, _ = cv2.projectPoints(ax_3d, rvec, tvec, K, DIST_COEFFS)\n')
        new_lines.append('            candidates.append({"confidence": confidence, "center_2d": [u_curr, v_curr], "points_2d": pts_2d.tolist(), "axis_2d": ax_2d.reshape(-1,2).tolist(), "position": {"x":float(x_p*1000),"y":float(y_p*1000),"z":float(z_f)}, "dimensions": {"l":float(l_f),"w":float(w_f),"h":float(h_f)}})\n')
        skip = True
    elif 'final_results = nms(candidates, dist_threshold=60.0)' in line:
        new_lines.append('    # 3. NMS 过滤 (优化后的阈值)\n')
        new_lines.append('    final_results = nms(candidates, dist_threshold=35.0)\n')
        skip = False
    elif not skip:
        new_lines.append(line)

with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print("File updated successfully")
