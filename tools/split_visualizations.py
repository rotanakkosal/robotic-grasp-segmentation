#!/usr/bin/env python3
"""
Split all visualizations into per-object folders with individual images.
Each object gets its own folder with separate panel images.

Output structure:
  output_img1761_split/
    obj_00/
      01_mask_overlay.png        - Segmented mask with color overlay
      02_shape_analysis.png      - Bounding rect + convex hull
      03_dt_heatmap.png          - Distance Transform heatmap
      04_moments_safety.png      - Moments + safety check visualization
      05_final_center.png        - Final centroid result
      06_full_image.png          - Full image with this object highlighted
      07_dt_vs_moments.png       - DT vs Moments side comparison
      08_crop_rgb.png            - Clean RGB crop (no overlay)
      09_mask_binary.png         - Binary mask crop
    obj_01/
      ...
    shape_metrics_overview.png
    full_centroids.png
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np

# Load masks saved by the previous run
MASK_DIR = "/home/kosal/cbnu_project/AI/picking-arm-robot/uoais/output_img1761/masks"
IMG_PATH = "/home/kosal/cbnu_project/AI/picking-arm-robot/uoais/sample_data/arm-robot-Dataset/IMG_1761.png"
OUT_DIR = "/home/kosal/cbnu_project/AI/picking-arm-robot/uoais/output_img1761_split"

# Working resolution for masks / overlays
MODEL_W, MODEL_H = 640, 480

os.makedirs(OUT_DIR, exist_ok=True)

# Load and resize RGB to match saved masks
rgb_orig = cv2.imread(IMG_PATH)
rgb = cv2.resize(rgb_orig, (MODEL_W, MODEL_H))
H, W = rgb.shape[:2]
depth_mm = np.ones((H, W), dtype=np.float32) * 800.0

# Load masks
masks = []
i = 0
while True:
    path = os.path.join(MASK_DIR, f"mask_{i:02d}.png")
    if not os.path.exists(path):
        break
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    masks.append(m)
    i += 1

print(f"Loaded {len(masks)} masks from {MASK_DIR}")
print(f"RGB: {W}x{H}, Output: {OUT_DIR}/\n")

from centroid_utils import compute_centroid_adaptive

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
    (255, 100, 150), (180, 130, 255),
]


for obj_idx, mask in enumerate(masks):
    obj_dir = os.path.join(OUT_DIR, f"obj_{obj_idx:02d}")
    os.makedirs(obj_dir, exist_ok=True)

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
        shape_label = f"circular (circ={circularity:.2f})"
    elif aspect_ratio > 2.0:
        shape_label = f"elongated (AR={aspect_ratio:.1f})"
    else:
        shape_label = f"compact (circ={circularity:.2f})"

    # Crop region with padding
    coords = np.where(mask_uint8 > 0)
    y_min, y_max = max(0, coords[0].min() - 20), min(H, coords[0].max() + 20)
    x_min, x_max = max(0, coords[1].min() - 20), min(W, coords[1].max() + 20)

    crop_rgb = rgb[y_min:y_max, x_min:x_max].copy()
    crop_mask = mask_uint8[y_min:y_max, x_min:x_max]

    cnts, _ = cv2.findContours(crop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt_crop = max(cnts, key=cv2.contourArea) if cnts else None

    # DT
    dt = cv2.distanceTransform(crop_mask, cv2.DIST_L2, 5)
    max_dt = dt.max()
    _, _, _, dt_loc = cv2.minMaxLoc(dt)
    dtx, dty = dt_loc

    # Moments
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

    color = OBJ_COLORS[obj_idx % len(OBJ_COLORS)]

    # ── 08: Clean RGB crop (no overlay) ──
    cv2.imwrite(os.path.join(obj_dir, "08_crop_rgb.png"), crop_rgb)

    # ── 09: Binary mask crop ──
    cv2.imwrite(os.path.join(obj_dir, "09_mask_binary.png"), crop_mask * 255)

    # ── 01: Mask overlay ──
    p1 = crop_rgb.copy()
    p1[crop_mask > 0] = (p1[crop_mask > 0] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
    cv2.drawContours(p1, cnts, -1, WHITE, 1)
    cv2.imwrite(os.path.join(obj_dir, "01_mask_overlay.png"), p1)

    # ── 02: Shape analysis ──
    p2 = np.full((crop_mask.shape[0], crop_mask.shape[1], 3), BG, dtype=np.uint8)
    p2[crop_mask > 0] = [80, 80, 80]
    cv2.drawContours(p2, cnts, -1, WHITE, 1)
    if cnt_crop is not None:
        rect_crop = cv2.minAreaRect(cnt_crop)
        box = cv2.boxPoints(rect_crop).astype(np.int32)
        cv2.polylines(p2, [box], True, YELLOW, 2)
        hull_crop = cv2.convexHull(cnt_crop)
        cv2.polylines(p2, [hull_crop], True, CYAN, 1)
    cv2.imwrite(os.path.join(obj_dir, "02_shape_analysis.png"), p2)

    # ── 03: DT heatmap ──
    dt_norm = (dt / max(max_dt, 1) * 255).astype(np.uint8)
    p3 = cv2.applyColorMap(dt_norm, cv2.COLORMAP_JET)
    p3[crop_mask == 0] = np.array(BG)
    cv2.circle(p3, dt_loc, 5, WHITE, 2)
    cv2.imwrite(os.path.join(obj_dir, "03_dt_heatmap.png"), p3)

    # ── 04: Moments + Safety ──
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
    cv2.imwrite(os.path.join(obj_dir, "04_moments_safety.png"), p4)

    # ── 05: Final center ──
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
    cv2.imwrite(os.path.join(obj_dir, "05_final_center.png"), p5)

    # ── 06: Full image with this object highlighted ──
    p6 = rgb.copy()
    # Dim all other objects
    for mi, m in enumerate(masks):
        mb = (m > 0)
        c = OBJ_COLORS[mi % len(OBJ_COLORS)]
        if mi == obj_idx:
            p6[mb] = (p6[mb] * 0.4 + np.array(c) * 0.6).astype(np.uint8)
        else:
            p6[mb] = (p6[mb] * 0.7 + np.array([80, 80, 80]) * 0.3).astype(np.uint8)
    cnts_full, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(p6, cnts_full, -1, YELLOW, 2)
    cv2.circle(p6, (global_cx, global_cy), 10, RED, -1)
    cv2.circle(p6, (global_cx, global_cy), 10, WHITE, 2)
    cv2.imwrite(os.path.join(obj_dir, "06_full_image.png"), p6)

    # ── 07: DT vs Moments comparison ──
    dist_err = np.sqrt((dtx - mx_i)**2 + (dty - my_i)**2)

    # Left: DT heatmap with DT point
    dt_left = cv2.applyColorMap(dt_norm, cv2.COLORMAP_JET)
    dt_left[crop_mask == 0] = np.array(BG)
    cv2.circle(dt_left, dt_loc, 5, WHITE, -1)
    cv2.imwrite(os.path.join(obj_dir, "07a_dt_heatmap_point.png"), dt_left)

    # Right: RGB with both points + inscribed circle
    dt_right = crop_rgb.copy()
    dt_right[crop_mask > 0] = (dt_right[crop_mask > 0] * 0.4 + np.array([60, 60, 60]) * 0.6).astype(np.uint8)
    cv2.drawContours(dt_right, cnts, -1, WHITE, 1)
    cv2.circle(dt_right, dt_loc, int(max_dt), WHITE, 1, cv2.LINE_AA)
    cv2.circle(dt_right, dt_loc, 5, RED, -1)
    cv2.circle(dt_right, dt_loc, 5, WHITE, 2)
    cv2.circle(dt_right, (mx_i, my_i), 5, GREEN, -1)
    cv2.circle(dt_right, (mx_i, my_i), 5, WHITE, 2)
    if dist_err > 5:
        cv2.arrowedLine(dt_right, dt_loc, (mx_i, my_i), YELLOW, 2, tipLength=0.12)
    cv2.imwrite(os.path.join(obj_dir, "07b_dt_vs_moments.png"), dt_right)

    # ── info.txt: metadata for this object ──
    with open(os.path.join(obj_dir, "info.txt"), "w") as f:
        f.write(f"Object #{obj_idx}\n")
        f.write(f"Shape: {shape_label}\n")
        f.write(f"Area: {area} pixels\n")
        f.write(f"Circularity: {circularity:.3f}\n")
        f.write(f"Aspect Ratio: {aspect_ratio:.2f}\n")
        f.write(f"Solidity: {solidity:.3f}\n")
        f.write(f"Is Circular: {is_circular}\n")
        f.write(f"Method: {'DT (circular)' if is_circular else 'Moments + Safety'}\n")
        f.write(f"Safety: {safety_text}\n")
        f.write(f"DT max at: ({dtx + x_min}, {dty + y_min})\n")
        f.write(f"Moments at: ({mx_i + x_min}, {my_i + y_min})\n")
        f.write(f"Final center: ({global_cx}, {global_cy})\n")
        f.write(f"DT vs Moments error: {dist_err:.1f}px\n")
        f.write(f"DT radius: {max_dt:.1f}px\n")
        f.write(f"Crop region: x=[{x_min},{x_max}] y=[{y_min},{y_max}]\n")

    print(f"  obj_{obj_idx:02d}/  {shape_label:30s}  center=({global_cx},{global_cy})  [{9} files]")


# ── Shape metrics overview (all objects) ──
n_obj = len(masks)
cols = min(n_obj, 6)
rows = (n_obj + cols - 1) // cols
panel_sz = 150
canvas_w = panel_sz * cols + (cols + 1) * 15
canvas_h = panel_sz * rows + rows * 95 + 40
img_sm = np.full((canvas_h, canvas_w, 3), BG, dtype=np.uint8)

for obj_idx, mask in enumerate(masks):
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
    hull_cnt = cv2.convexHull(cnt)
    h_area = cv2.contourArea(hull_cnt)
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

    row = obj_idx // cols
    col = obj_idx % cols
    sx = 15 + col * (panel_sz + 15)
    sy = 30 + row * (panel_sz + 95)
    img_sm[sy:sy + panel_sz, sx:sx + panel_sz] = crop

    circ_color = GREEN if is_circ else ORANGE
    ty = sy + panel_sz + 15
    cv2.putText(img_sm, f"#{obj_idx}", (sx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1, cv2.LINE_AA)
    cv2.putText(img_sm, f"circ={circ:.2f}", (sx, ty + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.33, circ_color, 1, cv2.LINE_AA)
    cv2.putText(img_sm, f"AR={ar:.2f}", (sx, ty + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.33, RED if ar > 2.0 else WHITE, 1, cv2.LINE_AA)
    cv2.putText(img_sm, f"sol={sol:.2f}", (sx, ty + 47), cv2.FONT_HERSHEY_SIMPLEX, 0.33, ORANGE if sol < 0.85 else WHITE, 1, cv2.LINE_AA)
    shape_txt = "CIRCULAR" if is_circ else ("ELONGATED" if ar > 2.0 else "COMPACT")
    cv2.putText(img_sm, shape_txt, (sx, ty + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.35, circ_color, 1, cv2.LINE_AA)

cv2.imwrite(os.path.join(OUT_DIR, "shape_metrics_overview.png"), img_sm)
print(f"\n  shape_metrics_overview.png")

# ── Full centroids image ──
vis_full = rgb.copy()
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

cv2.imwrite(os.path.join(OUT_DIR, "full_centroids.png"), vis_full)
print(f"  full_centroids.png")

print(f"\n{'='*60}")
print(f"  DONE! Output: {OUT_DIR}/")
print(f"{'='*60}")
print(f"\n  Per-object folders (obj_00 to obj_{len(masks)-1:02d}):")
print(f"    01_mask_overlay.png     - Segmented mask with color")
print(f"    02_shape_analysis.png   - Bounding rect (yellow) + convex hull (cyan)")
print(f"    03_dt_heatmap.png       - Distance Transform heatmap")
print(f"    04_moments_safety.png   - Moments/DT point + safety circle")
print(f"    05_final_center.png     - Final centroid (green dot)")
print(f"    06_full_image.png       - Full image, this object highlighted")
print(f"    07a_dt_heatmap_point.png - DT heatmap with max point")
print(f"    07b_dt_vs_moments.png   - Both points + inscribed circle + arrow")
print(f"    08_crop_rgb.png         - Clean RGB crop (no overlay)")
print(f"    09_mask_binary.png      - Binary mask crop")
print(f"    info.txt                - All metrics & coordinates")
