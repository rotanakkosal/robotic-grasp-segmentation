#!/usr/bin/env python3
"""
Generate the 3-panel suction grasp visualization for ALL 5 objects in T-Less frame 10.
Each image shows: Depth Map → Top Surface Flatness → Result (RED center + GREEN grasp)
All values computed live from real depth data.
"""
import cv2
import numpy as np
import os
import json
import glob

OUT = "/home/kosal/cbnu_project/AI/picking-arm-robot/uoais/output_explanation"
os.makedirs(OUT, exist_ok=True)

WHITE = (255, 255, 255)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 150, 50)
YELLOW = (0, 230, 255)
GRAY = (180, 180, 180)
ORANGE = (0, 140, 255)
CYAN = (255, 255, 0)
BG = (30, 30, 30)


def put_text(img, text, pos, scale=0.55, color=WHITE, thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def put_text_center(img, text, y, scale=0.6, color=WHITE, thickness=1):
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = (img.shape[1] - tw) // 2
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def generate_suction_for_object(rgb, depth_mm, masks, obj_idx, out_path):
    """Generate a 3-panel suction grasp image for one object."""
    mask = masks[obj_idx]
    mask_uint8 = (mask > 0).astype(np.uint8)
    H_img, W_img = rgb.shape[:2]

    # Shape info for title
    contours_full, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt_full = max(contours_full, key=cv2.contourArea)
    contour_area = cv2.contourArea(cnt_full)
    perimeter = cv2.arcLength(cnt_full, True)
    circularity = (4 * np.pi * contour_area) / (perimeter * perimeter) if perimeter > 0 else 0
    rect = cv2.minAreaRect(cnt_full)
    w_r, h_r = rect[1]
    aspect_ratio = max(w_r, h_r) / min(w_r, h_r) if min(w_r, h_r) > 0 else 1.0
    is_circular = circularity > 0.7

    if is_circular:
        shape_label = f"circular, circ={circularity:.2f}"
    elif aspect_ratio > 2.0:
        shape_label = f"elongated, AR={aspect_ratio:.1f}"
    else:
        shape_label = f"compact, circ={circularity:.2f}"

    # Crop around object
    coords = np.where(mask_uint8 > 0)
    y_min, y_max = coords[0].min() - 20, coords[0].max() + 20
    x_min, x_max = coords[1].min() - 20, coords[1].max() + 20
    y_min, x_min = max(0, y_min), max(0, x_min)
    y_max, x_max = min(H_img, y_max), min(W_img, x_max)

    crop_mask = mask_uint8[y_min:y_max, x_min:x_max]
    crop_depth = depth_mm[y_min:y_max, x_min:x_max]
    crop_rgb = rgb[y_min:y_max, x_min:x_max]

    cnts, _ = cv2.findContours(crop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- Compute geometric center (adaptive) ---
    dt = cv2.distanceTransform(crop_mask, cv2.DIST_L2, 5)
    max_dt_val = dt.max()
    _, _, _, dt_loc = cv2.minMaxLoc(dt)
    dtx, dty = dt_loc

    M = cv2.moments(crop_mask)
    mx_c = M["m10"] / M["m00"]
    my_c = M["m01"] / M["m00"]
    mx_i = int(np.clip(round(mx_c), 0, crop_mask.shape[1] - 1))
    my_i = int(np.clip(round(my_c), 0, crop_mask.shape[0] - 1))

    if is_circular:
        cx_r, cy_r = dtx, dty
    else:
        inside = crop_mask[my_i, mx_i] > 0
        edge_d = dt[my_i, mx_i] if inside else 0.0
        min_clearance = max_dt_val * 0.20
        if inside and edge_d >= min_clearance:
            cx_r, cy_r = mx_i, my_i
        elif inside:
            edge_ratio = edge_d / max_dt_val if max_dt_val > 0 else 0
            dt_weight = min(1.0 - (edge_ratio / 0.20), 0.7)
            cx_r = int((1.0 - dt_weight) * mx_c + dt_weight * dtx)
            cy_r = int((1.0 - dt_weight) * my_c + dt_weight * dty)
        else:
            cx_r, cy_r = dtx, dty

    # --- Compute depth-based suction grasp ---
    dep_on_mask = crop_depth.copy()
    dep_on_mask[crop_mask == 0] = 0
    valid = dep_on_mask[dep_on_mask > 0]
    if len(valid) > 0:
        d_min, d_max = valid.min(), valid.max()
    else:
        d_min, d_max = 0, 0
    depth_range = d_max - d_min

    # Top surface (closest to camera)
    top_thresh = d_min + depth_range * 0.20
    top_surface = (dep_on_mask > 0) & (dep_on_mask <= top_thresh)
    top_count = int(np.sum(top_surface))

    # Surface normals from depth gradients
    dx = cv2.Sobel(crop_depth, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(crop_depth, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(dx**2 + dy**2)

    # Flatness = inverse of gradient magnitude on top surface
    flatness = np.zeros_like(gradient_mag)
    if np.any(top_surface) and gradient_mag[top_surface].max() > 0:
        flatness[top_surface] = 1.0 - np.clip(
            gradient_mag[top_surface] / gradient_mag[top_surface].max(), 0, 1
        )

    # Edge clearance on top surface
    dt_top = cv2.distanceTransform(top_surface.astype(np.uint8), cv2.DIST_L2, 5)
    dt_top_max = dt_top.max()
    dt_top_norm = dt_top / max(dt_top_max, 1)

    # Center proximity (prefer center of crop for balanced grip)
    ch, cw = crop_mask.shape[:2]
    yy, xx = np.mgrid[:ch, :cw]
    bbox_cx = (coords[1].min() - x_min + coords[1].max() - x_min) / 2
    bbox_cy = (coords[0].min() - y_min + coords[0].max() - y_min) / 2
    center_dist = np.sqrt((xx - bbox_cx)**2 + (yy - bbox_cy)**2)
    max_cd = center_dist.max()
    center_prox = 1.0 - np.clip(center_dist / max(max_cd, 1), 0, 1)

    # Combined score
    score = np.zeros_like(flatness)
    if np.any(top_surface):
        score[top_surface] = (
            0.40 * flatness[top_surface] +
            0.30 * dt_top_norm[top_surface] +
            0.30 * center_prox[top_surface]
        )

    if np.any(score > 0):
        best_idx = np.unravel_index(np.argmax(score), score.shape)
        gx, gy = best_idx[1], best_idx[0]
        best_score = score[best_idx]
        grasp_flatness = flatness[best_idx]
        grasp_edge = dt_top_norm[best_idx]
        grasp_center = center_prox[best_idx]
    else:
        gx, gy = cx_r, cy_r
        best_score = 0
        grasp_flatness = 0
        grasp_edge = 0
        grasp_center = 0

    # Global coords
    global_cx, global_cy = cx_r + x_min, cy_r + y_min
    global_gx, global_gy = gx + x_min, gy + y_min
    dist_cg = np.sqrt((cx_r - gx)**2 + (cy_r - gy)**2)

    # Depth at grasp point
    depth_at_grasp = crop_depth[gy, gx] if crop_mask[gy, gx] > 0 else 0

    # --- Build 3-panel image ---
    panel_w, panel_h = 280, 220
    canvas_w = panel_w * 3 + 80
    canvas_h = panel_h + 165
    img = np.full((canvas_h, canvas_w, 3), BG, dtype=np.uint8)

    title = f"SUCTION GRASP: Object #{obj_idx} ({shape_label})"
    put_text_center(img, title, 25, 0.55, YELLOW, 2)
    put_text_center(img, f"Depth range: {d_min:.0f}mm - {d_max:.0f}mm  (range={depth_range:.0f}mm)", 48, 0.4, GRAY)

    def resize_p(src):
        return cv2.resize(src, (panel_w, panel_h), interpolation=cv2.INTER_LINEAR)

    sy1 = 60

    # ── PANEL 1: Depth Map ──
    dep_norm = np.zeros_like(dep_on_mask, dtype=np.uint8)
    if len(valid) > 0 and (d_max - d_min) > 0:
        valid_mask = dep_on_mask > 0
        dep_norm[valid_mask] = (
            (dep_on_mask[valid_mask] - d_min) / (d_max - d_min) * 255
        ).astype(np.uint8)

    p1 = cv2.applyColorMap(dep_norm, cv2.COLORMAP_VIRIDIS)
    p1[crop_mask == 0] = np.array(BG)
    cv2.drawContours(p1, cnts, -1, WHITE, 1)
    # Mark top surface boundary
    if np.any(top_surface):
        top_cnts, _ = cv2.findContours(top_surface.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(p1, top_cnts, -1, YELLOW, 1)

    p1 = resize_p(p1)
    sx1 = 20
    img[sy1:sy1 + panel_h, sx1:sx1 + panel_w] = p1
    put_text(img, "1. Depth Map", (sx1, sy1 + panel_h + 15), 0.48, WHITE, 1)
    put_text(img, f"   BRIGHT={d_min:.0f}mm (top)  DARK={d_max:.0f}mm (far)", (sx1, sy1 + panel_h + 32), 0.33, CYAN)
    put_text(img, f"   Yellow outline = top surface (<{top_thresh:.0f}mm)", (sx1, sy1 + panel_h + 48), 0.33, YELLOW)
    put_text(img, f"   Top surface: {top_count} pixels", (sx1, sy1 + panel_h + 63), 0.33, GRAY)

    # ── PANEL 2: Flatness Heatmap ──
    flat_vis = np.zeros_like(dep_norm)
    if np.any(top_surface):
        flat_vals = (flatness[top_surface] * 255).astype(np.uint8)
        flat_vis[top_surface] = flat_vals

    p2 = cv2.applyColorMap(flat_vis, cv2.COLORMAP_HOT)
    p2[crop_mask == 0] = np.array(BG)
    p2[~top_surface & (crop_mask > 0)] = np.array([40, 40, 40])
    cv2.drawContours(p2, cnts, -1, WHITE, 1)

    # Mark the best grasp point on flatness map
    # Scale gx, gy to panel coords
    scale_x = panel_w / crop_mask.shape[1]
    scale_y = panel_h / crop_mask.shape[0]
    gx_p = int(gx * scale_x)
    gy_p = int(gy * scale_y)
    cv2.drawMarker(p2, (gx, gy), GREEN, cv2.MARKER_CROSS, 15, 2)

    p2 = resize_p(p2)
    sx2 = 20 + panel_w + 20
    img[sy1:sy1 + panel_h, sx2:sx2 + panel_w] = p2
    put_text(img, "2. Flatness + Scoring", (sx2, sy1 + panel_h + 15), 0.48, WHITE, 1)
    put_text(img, f"   HOT = flat surface (good suction seal)", (sx2, sy1 + panel_h + 32), 0.33, ORANGE)
    put_text(img, f"   DARK = sides/slopes (bad seal)", (sx2, sy1 + panel_h + 48), 0.33, GRAY)
    put_text(img, f"   Score = 40%flat + 30%edge + 30%center", (sx2, sy1 + panel_h + 63), 0.33, CYAN)

    # ── PANEL 3: Final Result ──
    p3 = crop_rgb.copy()
    p3[crop_mask > 0] = (p3[crop_mask > 0] * 0.5 + np.array([100, 180, 100]) * 0.5).astype(np.uint8)
    cv2.drawContours(p3, cnts, -1, WHITE, 1)

    # Draw RED dot (geometric center)
    cv2.circle(p3, (cx_r, cy_r), 7, RED, -1)
    cv2.circle(p3, (cx_r, cy_r), 7, WHITE, 2)
    put_text(p3, "CENTER", (cx_r - 30, cy_r - 12), 0.3, RED, 1)

    # Draw GREEN cross (suction grasp)
    cv2.drawMarker(p3, (gx, gy), GREEN, cv2.MARKER_CROSS, 22, 2)
    put_text(p3, "GRASP", (gx + 12, gy - 3), 0.3, GREEN, 1)

    # White line if different
    if dist_cg > 3:
        cv2.line(p3, (cx_r, cy_r), (gx, gy), WHITE, 1, cv2.LINE_AA)
        mid_x = (cx_r + gx) // 2
        mid_y = (cy_r + gy) // 2
        put_text(p3, f"{dist_cg:.0f}px", (mid_x + 5, mid_y - 5), 0.3, YELLOW)

    p3 = resize_p(p3)
    sx3 = 20 + (panel_w + 20) * 2
    img[sy1:sy1 + panel_h, sx3:sx3 + panel_w] = p3
    put_text(img, "3. RESULT: Two Points", (sx3, sy1 + panel_h + 15), 0.48, WHITE, 1)
    put_text(img, f"   RED center = ({global_cx}, {global_cy})", (sx3, sy1 + panel_h + 32), 0.33, RED)
    put_text(img, f"   GREEN grasp = ({global_gx}, {global_gy}) d={depth_at_grasp:.0f}mm", (sx3, sy1 + panel_h + 48), 0.33, GREEN)
    put_text(img, f"   Offset = {dist_cg:.1f}px  Score = {best_score:.2f}", (sx3, sy1 + panel_h + 63), 0.33, YELLOW)

    # Score breakdown at bottom
    put_text(img, f"Best grasp score breakdown:  flatness={grasp_flatness:.2f}  edge={grasp_edge:.2f}  center={grasp_center:.2f}",
             (20, canvas_h - 10), 0.38, GRAY)

    cv2.imwrite(out_path, img)
    print(f"  Saved: {out_path}")

    return {
        'obj_idx': obj_idx,
        'shape': shape_label,
        'center': (global_cx, global_cy),
        'grasp': (global_gx, global_gy),
        'offset': dist_cg,
        'score': best_score,
        'depth': depth_at_grasp,
        'depth_range': depth_range,
    }


def main():
    scene_dir = "/home/kosal/cbnu_project/AI/picking-arm-robot/dataset/t_less-data/000003"
    frame_id = 10
    fid = f"{frame_id:06d}"

    print(f"Loading T-Less frame {frame_id}...")
    rgb = cv2.imread(os.path.join(scene_dir, "rgb", f"{fid}.png"))
    depth_raw = cv2.imread(os.path.join(scene_dir, "depth", f"{fid}.png"), cv2.IMREAD_UNCHANGED)
    with open(os.path.join(scene_dir, "scene_camera.json")) as f:
        cam = json.load(f)[str(frame_id)]
    depth_mm = depth_raw.astype(np.float32) * cam["depth_scale"]

    mask_files = sorted(glob.glob(os.path.join(scene_dir, "mask", f"{fid}_*.png")))
    masks = [cv2.imread(mf, cv2.IMREAD_GRAYSCALE) for mf in mask_files]

    print(f"Found {len(masks)} objects\n")

    results = []
    for obj_idx in range(len(masks)):
        mask = masks[obj_idx]
        if np.sum(mask > 0) < 100:
            print(f"  Object #{obj_idx}: too small, skipping")
            continue
        out_path = os.path.join(OUT, f"07_suction_grasp_obj{obj_idx}.png")
        result = generate_suction_for_object(rgb, depth_mm, masks, obj_idx, out_path)
        results.append(result)

    # Summary
    print(f"\n{'='*80}")
    print(f"  SUCTION GRASP SUMMARY: All Objects in Frame 10")
    print(f"{'='*80}")
    print(f"  {'Obj':>3} | {'Shape':>25} | {'Center':>12} | {'Grasp':>12} | {'Offset':>7} | {'Score':>5} | {'Depth':>7}")
    print(f"  {'-'*3}-+-{'-'*25}-+-{'-'*12}-+-{'-'*12}-+-{'-'*7}-+-{'-'*5}-+-{'-'*7}")
    for r in results:
        print(f"  #{r['obj_idx']:>2} | {r['shape']:>25} | ({r['center'][0]:>3},{r['center'][1]:>3}) "
              f"| ({r['grasp'][0]:>3},{r['grasp'][1]:>3}) | {r['offset']:>5.1f}px | {r['score']:>5.2f} | {r['depth']:>5.0f}mm")

    print(f"\nAll images saved to: {OUT}/")


if __name__ == "__main__":
    main()
