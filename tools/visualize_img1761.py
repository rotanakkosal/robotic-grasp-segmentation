#!/usr/bin/env python3
"""
Run UOAIS segmentation on IMG_1761.png, save masks, then generate
ALL visualization pipelines (2D, 3D/dummy, DT, shape metrics).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np
import torch
import glob

# ============================================================
# STEP 1: Run UOAIS segmentation
# ============================================================
print("=" * 70)
print("  STEP 1: Running UOAIS segmentation on IMG_1761.png")
print("=" * 70)

IMG_PATH = "/home/kosal/cbnu_project/AI/picking-arm-robot/uoais/sample_data/arm-robot-Dataset/IMG_1761.png"
OUT_DIR = "/home/kosal/cbnu_project/AI/picking-arm-robot/uoais/output_img1761"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "masks"), exist_ok=True)

rgb_orig = cv2.imread(IMG_PATH)
# Resize if too large (UOAIS expects ~640x480)
max_dim = 1280
h_orig, w_orig = rgb_orig.shape[:2]
if max(h_orig, w_orig) > max_dim:
    scale = max_dim / max(h_orig, w_orig)
    rgb = cv2.resize(rgb_orig, (int(w_orig * scale), int(h_orig * scale)))
else:
    rgb = rgb_orig.copy()
H, W = rgb.shape[:2]
print(f"  Original: {w_orig}x{h_orig} -> Resized: {W}x{H}")

# Create dummy depth (no real depth for phone photos)
depth_mm = np.ones((H, W), dtype=np.float32) * 800.0

# Use the existing run_with_centroids pipeline
from run_with_centroids import process_single_image
from adet.utils.post_process import DefaultPredictor
from adet.config import get_cfg

uoais_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cfg = get_cfg()
cfg.merge_from_file(os.path.join(uoais_root, "configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml"))
cfg.defrost()
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3

print("  Loading model...")
predictor = DefaultPredictor(cfg)

print("  Running inference via process_single_image...")
results = process_single_image(
    rgb_img=rgb,
    depth_img_raw=depth_mm,
    predictor=predictor,
    cfg=cfg,
    fg_model=None,
    centroid_method="suction",
    camera_intrinsics={'fx': 591.0, 'fy': 591.0, 'cx': W / 2, 'cy': H / 2},
    use_amodal=True,
    clean_vis=True,
    suction_cup_radius=15
)

# Extract masks from results
pred_masks = results['masks']
pred_visible = results['visible_masks']
pred_boxes = results['boxes']
scores_arr = results.get('scores', np.ones(len(pred_masks)))
if hasattr(scores_arr, 'cpu'):
    scores_arr = scores_arr.cpu().numpy()

# Resize rgb to match mask size (model resizes internally)
model_W, model_H = cfg.INPUT.IMG_SIZE
rgb = cv2.resize(rgb, (model_W, model_H))
H, W = rgb.shape[:2]
depth_mm = np.ones((H, W), dtype=np.float32) * 800.0

print(f"  Detected {len(pred_masks)} objects")

# Save masks and filter
masks = []
for i in range(len(pred_masks)):
    mask = (pred_masks[i] * 255).astype(np.uint8)
    mask_area = np.sum(mask > 0)
    if mask_area > 500:
        mask_path = os.path.join(OUT_DIR, "masks", f"mask_{len(masks):02d}.png")
        cv2.imwrite(mask_path, mask)
        masks.append(mask)
        score_val = scores_arr[i] if i < len(scores_arr) else 0
        print(f"    Object #{len(masks)-1}: area={mask_area} score={score_val:.2f}")

print(f"  Saved {len(masks)} masks to {OUT_DIR}/masks/")

# ============================================================
# STEP 2: Import visualization tools
# ============================================================
from centroid_utils import (
    compute_centroid_adaptive,
    compute_suction_grasp_point,
    analyze_mask_shape,
    compute_all_centroids,
    draw_centroids,
)

WHITE = (255, 255, 255)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 150, 50)
YELLOW = (0, 230, 255)
GRAY = (180, 180, 180)
ORANGE = (0, 140, 255)
CYAN = (255, 255, 0)
BG = (30, 30, 30)

OBJ_COLORS = [
    (100, 150, 255), (100, 255, 150), (255, 150, 100),
    (200, 100, 255), (100, 255, 255), (255, 200, 100),
    (150, 100, 255), (100, 200, 200), (200, 200, 100),
    (255, 100, 150),
]


def put_text(img, text, pos, scale=0.55, color=WHITE, thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def put_text_center(img, text, y, scale=0.6, color=WHITE, thickness=1):
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = (img.shape[1] - tw) // 2
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


# ============================================================
# STEP 3: Generate 2D Pipeline for each object
# ============================================================
print("\n" + "=" * 70)
print("  STEP 2: Generating 2D Pipeline visualizations")
print("=" * 70)

for obj_idx, mask in enumerate(masks):
    mask_uint8 = (mask > 0).astype(np.uint8)
    area = int(np.sum(mask_uint8))

    # Shape metrics
    contours_full, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_full:
        continue
    cnt_full = max(contours_full, key=cv2.contourArea)
    contour_area = cv2.contourArea(cnt_full)
    perimeter = cv2.arcLength(cnt_full, True)
    circularity = (4 * np.pi * contour_area) / (perimeter * perimeter) if perimeter > 0 else 0
    rect = cv2.minAreaRect(cnt_full)
    w_r, h_r = rect[1]
    aspect_ratio = max(w_r, h_r) / min(w_r, h_r) if min(w_r, h_r) > 0 else 1.0
    hull = cv2.convexHull(cnt_full)
    hull_area = cv2.contourArea(hull)
    solidity = contour_area / hull_area if hull_area > 0 else 1.0
    is_circular = circularity > 0.7

    if is_circular:
        shape_label = f"circular, circ={circularity:.2f}"
    elif aspect_ratio > 2.0:
        shape_label = f"elongated, AR={aspect_ratio:.1f}"
    else:
        shape_label = f"compact, circ={circularity:.2f}"

    # Crop
    coords = np.where(mask_uint8 > 0)
    y_min, y_max = max(0, coords[0].min() - 20), min(H, coords[0].max() + 20)
    x_min, x_max = max(0, coords[1].min() - 20), min(W, coords[1].max() + 20)

    crop_rgb = rgb[y_min:y_max, x_min:x_max].copy()
    crop_mask = mask_uint8[y_min:y_max, x_min:x_max]

    cnts, _ = cv2.findContours(crop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt_crop = max(cnts, key=cv2.contourArea) if cnts else None

    dt = cv2.distanceTransform(crop_mask, cv2.DIST_L2, 5)
    max_dt = dt.max()
    _, _, _, dt_loc = cv2.minMaxLoc(dt)
    dtx, dty = dt_loc

    M = cv2.moments(crop_mask)
    mx_c = M["m10"] / M["m00"]
    my_c = M["m01"] / M["m00"]
    mx_i = int(np.clip(round(mx_c), 0, crop_mask.shape[1] - 1))
    my_i = int(np.clip(round(my_c), 0, crop_mask.shape[0] - 1))

    inside = crop_mask[my_i, mx_i] > 0
    edge_d = dt[my_i, mx_i] if inside else 0.0
    min_clearance = max_dt * 0.20

    if is_circular:
        final_cx, final_cy = dtx, dty
        safety_text = "N/A (circular -> DT)"
    elif inside and edge_d >= min_clearance:
        final_cx, final_cy = mx_i, my_i
        safety_text = f"SAFE ({edge_d:.0f}px >= {min_clearance:.0f}px)"
    elif inside:
        edge_ratio = edge_d / max_dt if max_dt > 0 else 0
        dt_weight = min(1.0 - (edge_ratio / 0.20), 0.7)
        final_cx = int((1.0 - dt_weight) * mx_c + dt_weight * dtx)
        final_cy = int((1.0 - dt_weight) * my_c + dt_weight * dty)
        safety_text = f"BLEND ({edge_d:.0f}px < {min_clearance:.0f}px)"
    else:
        final_cx, final_cy = dtx, dty
        safety_text = "OUTSIDE -> DT fallback"

    global_cx = final_cx + x_min
    global_cy = final_cy + y_min

    # Build 6-panel image
    panel_w, panel_h = 250, 200
    canvas_w = panel_w * 3 + 80
    canvas_h = panel_h * 2 + 165
    img_out = np.full((canvas_h, canvas_w, 3), BG, dtype=np.uint8)

    title = f"2D PIPELINE: IMG_1761 Object #{obj_idx} ({shape_label})"
    put_text_center(img_out, title, 25, 0.48, YELLOW, 2)

    def resize_p(src):
        return cv2.resize(src, (panel_w, panel_h), interpolation=cv2.INTER_LINEAR)

    sy = 55
    gap = 20

    # Panel 1: Mask overlay
    p1 = crop_rgb.copy()
    color = OBJ_COLORS[obj_idx % len(OBJ_COLORS)]
    p1[crop_mask > 0] = (p1[crop_mask > 0] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
    cv2.drawContours(p1, cnts, -1, WHITE, 1)
    p1 = resize_p(p1)
    sx1 = 20
    img_out[sy:sy + panel_h, sx1:sx1 + panel_w] = p1
    put_text(img_out, "1. Segmented Mask", (sx1, sy + panel_h + 15), 0.43, WHITE, 1)
    put_text(img_out, f"   area={area}px", (sx1, sy + panel_h + 32), 0.33, GRAY)

    # Panel 2: Shape analysis
    p2 = np.full((crop_mask.shape[0], crop_mask.shape[1], 3), BG, dtype=np.uint8)
    p2[crop_mask > 0] = [80, 80, 80]
    cv2.drawContours(p2, cnts, -1, WHITE, 1)
    if cnt_crop is not None:
        rect_crop = cv2.minAreaRect(cnt_crop)
        box = cv2.boxPoints(rect_crop).astype(np.int32)
        cv2.polylines(p2, [box], True, YELLOW, 2)
        hull_crop = cv2.convexHull(cnt_crop)
        cv2.polylines(p2, [hull_crop], True, CYAN, 1)
    p2 = resize_p(p2)
    sx2 = sx1 + panel_w + gap
    img_out[sy:sy + panel_h, sx2:sx2 + panel_w] = p2
    put_text(img_out, "2. Shape Analysis", (sx2, sy + panel_h + 15), 0.43, WHITE, 1)
    circ_color = GREEN if is_circular else ORANGE
    put_text(img_out, f"   circ={circularity:.3f} AR={aspect_ratio:.2f} sol={solidity:.3f}", (sx2, sy + panel_h + 32), 0.30, GRAY)
    if is_circular:
        put_text(img_out, f"   CIRCULAR -> DT", (sx2, sy + panel_h + 48), 0.35, GREEN)
    elif aspect_ratio > 2.0:
        put_text(img_out, f"   ELONGATED -> Moments", (sx2, sy + panel_h + 48), 0.35, ORANGE)
    else:
        put_text(img_out, f"   COMPACT -> Moments", (sx2, sy + panel_h + 48), 0.35, ORANGE)

    # Panel 3: DT heatmap
    dt_norm = (dt / max(max_dt, 1) * 255).astype(np.uint8)
    p3 = cv2.applyColorMap(dt_norm, cv2.COLORMAP_JET)
    p3[crop_mask == 0] = np.array(BG)
    cv2.circle(p3, dt_loc, 5, WHITE, 2)
    p3 = resize_p(p3)
    sx3 = sx2 + panel_w + gap
    img_out[sy:sy + panel_h, sx3:sx3 + panel_w] = p3
    put_text(img_out, "3. Distance Transform", (sx3, sy + panel_h + 15), 0.43, WHITE, 1)
    if is_circular:
        put_text(img_out, f"   DT center = geometric center", (sx3, sy + panel_h + 32), 0.33, GREEN)
    else:
        put_text(img_out, f"   DT biased to widest part", (sx3, sy + panel_h + 32), 0.33, RED)

    # Row 2
    sy2 = sy + panel_h + 63

    # Panel 4: Method + Safety
    p4 = crop_rgb.copy()
    p4[crop_mask > 0] = (p4[crop_mask > 0] * 0.6 + np.array([80, 80, 80]) * 0.4).astype(np.uint8)
    cv2.drawContours(p4, cnts, -1, WHITE, 1)
    if is_circular:
        cv2.circle(p4, (dtx, dty), 7, GREEN, -1)
        cv2.circle(p4, (dtx, dty), 7, WHITE, 2)
    else:
        cv2.circle(p4, (mx_i, my_i), max(int(edge_d), 2), (0, 0, 200), 1)
        cv2.circle(p4, (mx_i, my_i), 5, RED, -1)
        cv2.circle(p4, (dtx, dty), 4, BLUE, -1)
    p4 = resize_p(p4)
    sx4 = 20
    img_out[sy2:sy2 + panel_h, sx4:sx4 + panel_w] = p4
    if is_circular:
        put_text(img_out, "4. Circular -> DT directly", (sx4, sy2 + panel_h + 15), 0.40, GREEN, 1)
    else:
        put_text(img_out, "4. Moments + Safety Check", (sx4, sy2 + panel_h + 15), 0.43, WHITE, 1)
    put_text(img_out, f"   {safety_text}", (sx4, sy2 + panel_h + 32), 0.33, circ_color)

    # Panel 5: Final result
    p5 = crop_rgb.copy()
    p5[crop_mask > 0] = (p5[crop_mask > 0] * 0.5 + np.array([100, 180, 100]) * 0.5).astype(np.uint8)
    cv2.drawContours(p5, cnts, -1, WHITE, 1)
    if not is_circular:
        cv2.circle(p5, (mx_i, my_i), 4, RED, -1)
        cv2.circle(p5, (dtx, dty), 4, BLUE, -1)
        if abs(mx_i - final_cx) + abs(my_i - final_cy) > 3:
            cv2.arrowedLine(p5, (mx_i, my_i), (final_cx, final_cy), YELLOW, 1, tipLength=0.15)
    cv2.circle(p5, (final_cx, final_cy), 7, GREEN, -1)
    cv2.circle(p5, (final_cx, final_cy), 7, WHITE, 2)
    p5 = resize_p(p5)
    sx5 = sx4 + panel_w + gap
    img_out[sy2:sy2 + panel_h, sx5:sx5 + panel_w] = p5
    put_text(img_out, f"5. FINAL RED DOT at ({global_cx},{global_cy})", (sx5, sy2 + panel_h + 15), 0.40, GREEN, 1)

    # Panel 6: Full image
    p6 = rgb.copy()
    for mi, m in enumerate(masks):
        mb = m > 0
        c = OBJ_COLORS[mi % len(OBJ_COLORS)]
        p6[mb] = (p6[mb] * 0.5 + np.array(c) * 0.5).astype(np.uint8)
    cv2.drawContours(p6, contours_full, -1, YELLOW, 2)
    cv2.circle(p6, (global_cx, global_cy), 10, RED, -1)
    cv2.circle(p6, (global_cx, global_cy), 10, WHITE, 2)
    cv2.putText(p6, f"#{obj_idx}", (global_cx + 12, global_cy - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2, cv2.LINE_AA)

    p6_h = panel_h
    p6_w = int(p6.shape[1] * p6_h / p6.shape[0])
    p6 = cv2.resize(p6, (p6_w, p6_h))
    sx6 = sx5 + panel_w + gap
    available = canvas_w - sx6
    if p6_w > available:
        p6 = p6[:, :available]
        p6_w = available
    img_out[sy2:sy2 + panel_h, sx6:sx6 + p6_w] = p6
    put_text(img_out, "6. Full Image", (sx6, sy2 + panel_h + 15), 0.43, WHITE, 1)

    out_path = os.path.join(OUT_DIR, f"2d_pipeline_obj{obj_idx}.png")
    cv2.imwrite(out_path, img_out)
    print(f"  Saved: {out_path}")
    print(f"    #{obj_idx}: {shape_label} -> center=({global_cx},{global_cy}) safety={safety_text}")


# ============================================================
# STEP 4: Generate DT explanation for each object
# ============================================================
print("\n" + "=" * 70)
print("  STEP 3: Generating DT comparison for each object")
print("=" * 70)

for obj_idx, mask in enumerate(masks):
    mask_uint8 = (mask > 0).astype(np.uint8)
    contours_full, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_full:
        continue
    cnt_full = max(contours_full, key=cv2.contourArea)
    contour_area = cv2.contourArea(cnt_full)
    perimeter = cv2.arcLength(cnt_full, True)
    circularity = (4 * np.pi * contour_area) / (perimeter * perimeter) if perimeter > 0 else 0

    coords = np.where(mask_uint8 > 0)
    y_min, y_max = max(0, coords[0].min() - 15), min(H, coords[0].max() + 15)
    x_min, x_max = max(0, coords[1].min() - 15), min(W, coords[1].max() + 15)
    crop_mask = mask_uint8[y_min:y_max, x_min:x_max]
    crop_rgb_c = rgb[y_min:y_max, x_min:x_max].copy()

    cnts_c, _ = cv2.findContours(crop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dt = cv2.distanceTransform(crop_mask, cv2.DIST_L2, 5)
    max_dt = dt.max()
    _, _, _, dt_loc = cv2.minMaxLoc(dt)
    M = cv2.moments(crop_mask)
    mom_x, mom_y = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    dist_err = np.sqrt((dt_loc[0] - mom_x)**2 + (dt_loc[1] - mom_y)**2)

    ch, cw = crop_mask.shape
    panel_w_dt = min(max(cw + 40, 300), 400)
    panel_h_dt = min(max(ch + 40, 200), 300)
    canvas_w_dt = panel_w_dt * 2 + 60
    canvas_h_dt = panel_h_dt + 100
    img_dt = np.full((canvas_h_dt, canvas_w_dt, 3), BG, dtype=np.uint8)

    is_circ = circularity > 0.7
    status = "CORRECT (circular)" if is_circ else f"ERROR = {dist_err:.0f}px!"
    put_text_center(img_dt, f"DT vs Moments: Object #{obj_idx} (circ={circularity:.2f}) -> {status}", 22, 0.42, YELLOW, 1)

    def resize_dt(src):
        return cv2.resize(src, (panel_w_dt, panel_h_dt), interpolation=cv2.INTER_LINEAR)

    # Left: DT heatmap
    dt_norm = (dt / max(max_dt, 1) * 255).astype(np.uint8)
    p_l = cv2.applyColorMap(dt_norm, cv2.COLORMAP_JET)
    p_l[crop_mask == 0] = np.array(BG)
    cv2.circle(p_l, dt_loc, 5, WHITE, -1)
    p_l = resize_dt(p_l)
    sx_l = 20
    sy_dt = 35
    img_dt[sy_dt:sy_dt + panel_h_dt, sx_l:sx_l + panel_w_dt] = p_l
    put_text(img_dt, "Distance Transform", (sx_l, sy_dt + panel_h_dt + 15), 0.4, WHITE)
    put_text(img_dt, f"DT max at ({dt_loc[0]+x_min},{dt_loc[1]+y_min})", (sx_l, sy_dt + panel_h_dt + 32), 0.33, RED)

    # Right: RGB with both points + inscribed circles
    p_r = crop_rgb_c.copy()
    p_r[crop_mask > 0] = (p_r[crop_mask > 0] * 0.4 + np.array([60, 60, 60]) * 0.6).astype(np.uint8)
    cv2.drawContours(p_r, cnts_c, -1, WHITE, 1)
    # DT inscribed circle
    cv2.circle(p_r, dt_loc, int(max_dt), WHITE, 1, cv2.LINE_AA)
    cv2.circle(p_r, dt_loc, 5, RED, -1)
    cv2.circle(p_r, dt_loc, 5, WHITE, 2)
    # Moments
    cv2.circle(p_r, (mom_x, mom_y), 5, GREEN, -1)
    cv2.circle(p_r, (mom_x, mom_y), 5, WHITE, 2)
    if dist_err > 5:
        cv2.arrowedLine(p_r, dt_loc, (mom_x, mom_y), YELLOW, 2, tipLength=0.12)
    p_r = resize_dt(p_r)
    sx_r = sx_l + panel_w_dt + 20
    img_dt[sy_dt:sy_dt + panel_h_dt, sx_r:sx_r + panel_w_dt] = p_r
    put_text(img_dt, "DT(RED) vs Moments(GREEN)", (sx_r, sy_dt + panel_h_dt + 15), 0.4, WHITE)
    err_color = GREEN if dist_err < 10 else RED
    put_text(img_dt, f"Error = {dist_err:.0f}px  r={max_dt:.0f}px", (sx_r, sy_dt + panel_h_dt + 32), 0.33, err_color)

    out_path = os.path.join(OUT_DIR, f"dt_comparison_obj{obj_idx}.png")
    cv2.imwrite(out_path, img_dt)
    print(f"  Saved: {out_path}  (DT err={dist_err:.0f}px)")


# ============================================================
# STEP 5: Shape metrics overview (all objects side by side)
# ============================================================
print("\n" + "=" * 70)
print("  STEP 4: Generating shape metrics overview")
print("=" * 70)

n_obj = len(masks)
panel_sz = 150
canvas_w_sm = panel_sz * min(n_obj, 6) + (min(n_obj, 6) + 1) * 15
canvas_h_sm = panel_sz + 160
img_sm = np.full((canvas_h_sm, canvas_w_sm, 3), BG, dtype=np.uint8)
put_text_center(img_sm, "Shape Metrics: All Objects in IMG_1761", 22, 0.5, YELLOW, 1)

for obj_idx, mask in enumerate(masks):
    if obj_idx >= 6:
        break
    mask_uint8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        continue
    cnt = max(contours, key=cv2.contourArea)
    c_area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    circ = (4 * np.pi * c_area) / (peri * peri) if peri > 0 else 0
    rect = cv2.minAreaRect(cnt)
    w_r, h_r = rect[1]
    ar = max(w_r, h_r) / min(w_r, h_r) if min(w_r, h_r) > 0 else 1.0
    hull = cv2.convexHull(cnt)
    h_area = cv2.contourArea(hull)
    sol = c_area / h_area if h_area > 0 else 1.0
    is_circ = circ > 0.7

    coords = np.where(mask_uint8 > 0)
    y1, y2 = max(0, coords[0].min() - 5), min(H, coords[0].max() + 5)
    x1, x2 = max(0, coords[1].min() - 5), min(W, coords[1].max() + 5)
    crop = rgb[y1:y2, x1:x2].copy()
    crop_m = mask_uint8[y1:y2, x1:x2]
    c_color = OBJ_COLORS[obj_idx % len(OBJ_COLORS)]
    crop[crop_m > 0] = (crop[crop_m > 0] * 0.5 + np.array(c_color) * 0.5).astype(np.uint8)
    crop_cnts, _ = cv2.findContours(crop_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(crop, crop_cnts, -1, WHITE, 1)
    crop = cv2.resize(crop, (panel_sz, panel_sz))

    sx = 15 + obj_idx * (panel_sz + 15)
    sy_sm = 40
    img_sm[sy_sm:sy_sm + panel_sz, sx:sx + panel_sz] = crop

    circ_color = GREEN if is_circ else ORANGE
    put_text(img_sm, f"#{obj_idx}", (sx, sy_sm + panel_sz + 15), 0.4, WHITE, 1)
    put_text(img_sm, f"circ={circ:.2f}", (sx, sy_sm + panel_sz + 32), 0.33, circ_color)
    put_text(img_sm, f"AR={ar:.2f}", (sx, sy_sm + panel_sz + 47), 0.33, RED if ar > 2.0 else WHITE)
    put_text(img_sm, f"sol={sol:.2f}", (sx, sy_sm + panel_sz + 62), 0.33, ORANGE if sol < 0.85 else WHITE)
    shape_txt = "CIRCULAR" if is_circ else ("ELONG" if ar > 2.0 else "COMPACT")
    put_text(img_sm, shape_txt, (sx, sy_sm + panel_sz + 80), 0.35, circ_color, 1)

out_path = os.path.join(OUT_DIR, "shape_metrics_overview.png")
cv2.imwrite(out_path, img_sm)
print(f"  Saved: {out_path}")


# ============================================================
# STEP 6: Full image with all centroids
# ============================================================
print("\n" + "=" * 70)
print("  STEP 5: Generating full centroid visualization")
print("=" * 70)

vis_full = rgb.copy()
np.random.seed(42)

for obj_idx, mask in enumerate(masks):
    mask_uint8 = (mask > 0).astype(np.uint8)
    mask_bool = mask > 0
    color = OBJ_COLORS[obj_idx % len(OBJ_COLORS)]
    vis_full[mask_bool] = (vis_full[mask_bool] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
    cnts, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis_full, cnts, -1, WHITE, 1)

    center = compute_centroid_adaptive(mask)
    if center:
        cx, cy = int(center[0]), int(center[1])
        cv2.circle(vis_full, (cx, cy), 8, RED, -1)
        cv2.circle(vis_full, (cx, cy), 8, WHITE, 2)
        cv2.putText(vis_full, f"#{obj_idx}", (cx + 10, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 2, cv2.LINE_AA)

cv2.putText(vis_full, "RED = Center (2D) | No depth -> GREEN=RED", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
out_path = os.path.join(OUT_DIR, "full_centroids.png")
cv2.imwrite(out_path, vis_full)
print(f"  Saved: {out_path}")


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("  ALL DONE!")
print("=" * 70)
print(f"  Output directory: {OUT_DIR}/")
print(f"  Files generated:")
print(f"    masks/mask_XX.png          - Segmentation masks ({len(masks)} objects)")
print(f"    2d_pipeline_objX.png       - 2D centroid pipeline per object")
print(f"    dt_comparison_objX.png     - DT vs Moments comparison per object")
print(f"    shape_metrics_overview.png - All shapes side by side")
print(f"    full_centroids.png         - Full image with all RED dots")
print(f"\n  NOTE: No real depth -> GREEN cross = RED dot (same position)")
print(f"  With L515 camera, GREEN would shift to flattest suction spot")
