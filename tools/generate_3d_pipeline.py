#!/usr/bin/env python3
"""
Generate 6-panel 3D DEPTH pipeline visualization for ALL 5 objects in T-Less frame 10.
Shows: RGB+Mask → Depth Map → Top Surface → Surface Normals → Scoring → Final GREEN cross
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


def generate_3d_pipeline(rgb, depth_mm, masks, obj_idx, out_path):
    """Generate a 6-panel 3D depth pipeline image for one object."""
    mask = masks[obj_idx]
    mask_uint8 = (mask > 0).astype(np.uint8)
    H_img, W_img = rgb.shape[:2]
    area = int(np.sum(mask_uint8))

    # Shape info for title
    contours_full, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt_full = max(contours_full, key=cv2.contourArea)
    contour_area = cv2.contourArea(cnt_full)
    perimeter = cv2.arcLength(cnt_full, True)
    circularity = (4 * np.pi * contour_area) / (perimeter * perimeter) if perimeter > 0 else 0
    rect = cv2.minAreaRect(cnt_full)
    w_r, h_r = rect[1]
    aspect_ratio = max(w_r, h_r) / min(w_r, h_r) if min(w_r, h_r) > 0 else 1.0

    if circularity > 0.7:
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

    crop_rgb = rgb[y_min:y_max, x_min:x_max].copy()
    crop_mask = mask_uint8[y_min:y_max, x_min:x_max]
    crop_depth = depth_mm[y_min:y_max, x_min:x_max]
    ch, cw = crop_mask.shape

    cnts, _ = cv2.findContours(crop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ─── Depth stats ───
    dep_on_mask = crop_depth.copy()
    dep_on_mask[crop_mask == 0] = 0
    valid = dep_on_mask[dep_on_mask > 0]
    d_min, d_max = valid.min(), valid.max()
    depth_range = d_max - d_min

    # ─── Step 1: Top surface ───
    top_thresh = d_min + depth_range * 0.20
    top_surface = (dep_on_mask > 0) & (dep_on_mask <= top_thresh)
    top_count = int(np.sum(top_surface))

    # ─── Step 2: Surface normals ───
    # Smooth depth slightly to reduce noise
    depth_smooth = cv2.GaussianBlur(crop_depth, (5, 5), 0)
    dx = cv2.Sobel(depth_smooth, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(depth_smooth, cv2.CV_64F, 0, 1, ksize=3)
    # Normal vector = (-dz/dx, -dz/dy, 1), normalized
    norm_z = np.ones_like(dx)
    magnitude = np.sqrt(dx**2 + dy**2 + norm_z**2)
    nx = -dx / magnitude
    ny = -dy / magnitude
    nz = norm_z / magnitude
    # Gradient magnitude (how tilted the surface is)
    gradient_mag = np.sqrt(dx**2 + dy**2)

    # ─── Step 3: Flatness ───
    flatness = np.zeros_like(gradient_mag)
    if np.any(top_surface) and gradient_mag[top_surface].max() > 0:
        flatness[top_surface] = 1.0 - np.clip(
            gradient_mag[top_surface] / gradient_mag[top_surface].max(), 0, 1
        )

    # ─── Step 4: Edge clearance on top surface ───
    dt_top = cv2.distanceTransform(top_surface.astype(np.uint8), cv2.DIST_L2, 5)
    dt_top_max = max(dt_top.max(), 1)
    dt_top_norm = dt_top / dt_top_max

    # ─── Step 5: Center proximity ───
    mask_coords = np.where(crop_mask > 0)
    bbox_cx = (mask_coords[1].min() + mask_coords[1].max()) / 2
    bbox_cy = (mask_coords[0].min() + mask_coords[0].max()) / 2
    yy, xx = np.mgrid[:ch, :cw]
    center_dist = np.sqrt((xx - bbox_cx)**2 + (yy - bbox_cy)**2)
    center_prox = 1.0 - np.clip(center_dist / max(center_dist.max(), 1), 0, 1)

    # ─── Step 6: Combined score ───
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
        s_flat = flatness[best_idx]
        s_edge = dt_top_norm[best_idx]
        s_center = center_prox[best_idx]
    else:
        gx, gy = cw // 2, ch // 2
        best_score, s_flat, s_edge, s_center = 0, 0, 0, 0

    # Also compute RED dot (geometric center) for comparison
    dt_full = cv2.distanceTransform(crop_mask, cv2.DIST_L2, 5)
    max_dt_full = dt_full.max()
    _, _, _, dt_full_loc = cv2.minMaxLoc(dt_full)
    M = cv2.moments(crop_mask)
    mx_c = M["m10"] / M["m00"]
    my_c = M["m01"] / M["m00"]
    mx_i = int(np.clip(round(mx_c), 0, cw - 1))
    my_i = int(np.clip(round(my_c), 0, ch - 1))

    is_circular = circularity > 0.7
    if is_circular:
        cx_r, cy_r = dt_full_loc
    else:
        inside = crop_mask[my_i, mx_i] > 0
        edge_d = dt_full[my_i, mx_i] if inside else 0.0
        min_clearance = max_dt_full * 0.20
        if inside and edge_d >= min_clearance:
            cx_r, cy_r = mx_i, my_i
        elif inside:
            edge_ratio = edge_d / max_dt_full if max_dt_full > 0 else 0
            dt_weight = min(1.0 - (edge_ratio / 0.20), 0.7)
            cx_r = int((1.0 - dt_weight) * mx_c + dt_weight * dt_full_loc[0])
            cy_r = int((1.0 - dt_weight) * my_c + dt_weight * dt_full_loc[1])
        else:
            cx_r, cy_r = dt_full_loc

    global_cx, global_cy = cx_r + x_min, cy_r + y_min
    global_gx, global_gy = gx + x_min, gy + y_min
    depth_at_grasp = crop_depth[gy, gx] if crop_mask[gy, gx] > 0 else 0
    dist_cg = np.sqrt((cx_r - gx)**2 + (cy_r - gy)**2)

    # ─── Build 6-panel image ───
    panel_w, panel_h = 250, 200
    canvas_w = panel_w * 3 + 80
    canvas_h = panel_h * 2 + 165
    img = np.full((canvas_h, canvas_w, 3), BG, dtype=np.uint8)

    title = f"3D PIPELINE: Object #{obj_idx} ({shape_label})"
    put_text_center(img, title, 25, 0.55, GREEN, 2)
    put_text_center(img, f"Depth: {d_min:.0f}-{d_max:.0f}mm  range={depth_range:.0f}mm  Uses: RGB + DEPTH", 48, 0.38, GRAY)

    def resize_p(src):
        return cv2.resize(src, (panel_w, panel_h), interpolation=cv2.INTER_LINEAR)

    sy1 = 60
    gap = 20

    # ── PANEL 1: RGB + Depth overlay ──
    # Show depth as color overlay on RGB
    dep_norm_full = np.zeros((ch, cw), dtype=np.uint8)
    valid_m = dep_on_mask > 0
    if np.any(valid_m):
        dep_norm_full[valid_m] = ((dep_on_mask[valid_m] - d_min) / max(depth_range, 1) * 255).astype(np.uint8)
    dep_color = cv2.applyColorMap(dep_norm_full, cv2.COLORMAP_VIRIDIS)

    p1 = crop_rgb.copy()
    # Blend depth color onto RGB where mask is active
    p1[crop_mask > 0] = (p1[crop_mask > 0] * 0.4 + dep_color[crop_mask > 0] * 0.6).astype(np.uint8)
    cv2.drawContours(p1, cnts, -1, WHITE, 1)
    p1 = resize_p(p1)
    sx1 = 20
    img[sy1:sy1 + panel_h, sx1:sx1 + panel_w] = p1
    put_text(img, "1. RGB + Depth Map", (sx1, sy1 + panel_h + 15), 0.43, WHITE, 1)
    put_text(img, f"   BRIGHT={d_min:.0f}mm (close/top)", (sx1, sy1 + panel_h + 32), 0.33, CYAN)
    put_text(img, f"   DARK={d_max:.0f}mm (far/bottom)", (sx1, sy1 + panel_h + 47), 0.33, (100, 50, 200))

    # ── PANEL 2: Top surface extraction ──
    p2 = crop_rgb.copy()
    p2[crop_mask > 0] = (p2[crop_mask > 0] * 0.3).astype(np.uint8)  # darken everything
    # Highlight top surface in bright
    p2[top_surface] = (crop_rgb[top_surface] * 0.5 + np.array([0, 255, 255]) * 0.5).astype(np.uint8)
    cv2.drawContours(p2, cnts, -1, WHITE, 1)
    # Draw top surface contour
    if np.any(top_surface):
        top_cnts, _ = cv2.findContours(top_surface.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(p2, top_cnts, -1, YELLOW, 2)
    p2 = resize_p(p2)
    sx2 = sx1 + panel_w + gap
    img[sy1:sy1 + panel_h, sx2:sx2 + panel_w] = p2
    put_text(img, "2. Top Surface (20%)", (sx2, sy1 + panel_h + 15), 0.43, WHITE, 1)
    put_text(img, f"   depth < {top_thresh:.0f}mm", (sx2, sy1 + panel_h + 32), 0.33, YELLOW)
    put_text(img, f"   {top_count} pixels (BRIGHT area)", (sx2, sy1 + panel_h + 47), 0.33, CYAN)

    # ── PANEL 3: Surface Normals ──
    # Visualize normals as RGB: nx→R, ny→G, nz→B
    normal_vis = np.full((ch, cw, 3), BG, dtype=np.uint8)
    if np.any(top_surface):
        # Map normal components [-1,1] to [0,255]
        normal_vis[top_surface, 2] = ((nx[top_surface] + 1) * 0.5 * 255).astype(np.uint8)  # R
        normal_vis[top_surface, 1] = ((ny[top_surface] + 1) * 0.5 * 255).astype(np.uint8)  # G
        normal_vis[top_surface, 0] = ((nz[top_surface]) * 255).astype(np.uint8)             # B
    cv2.drawContours(normal_vis, cnts, -1, (80, 80, 80), 1)
    if np.any(top_surface):
        cv2.drawContours(normal_vis, top_cnts, -1, WHITE, 1)
    p3 = resize_p(normal_vis)
    sx3 = sx2 + panel_w + gap
    img[sy1:sy1 + panel_h, sx3:sx3 + panel_w] = p3
    put_text(img, "3. Surface Normals", (sx3, sy1 + panel_h + 15), 0.43, WHITE, 1)
    put_text(img, f"   Sobel on depth -> dx, dy", (sx3, sy1 + panel_h + 32), 0.33, GRAY)
    put_text(img, f"   Uniform color = FLAT", (sx3, sy1 + panel_h + 47), 0.33, GREEN)

    # ── Row 2 ──
    sy2 = sy1 + panel_h + 63

    # ── PANEL 4: Flatness heatmap ──
    flat_vis = np.zeros((ch, cw), dtype=np.uint8)
    if np.any(top_surface):
        flat_vis[top_surface] = (flatness[top_surface] * 255).astype(np.uint8)
    p4 = cv2.applyColorMap(flat_vis, cv2.COLORMAP_HOT)
    p4[crop_mask == 0] = np.array(BG)
    p4[~top_surface & (crop_mask > 0)] = np.array([30, 30, 30])
    cv2.drawContours(p4, cnts, -1, (80, 80, 80), 1)
    p4 = resize_p(p4)
    sx4 = 20
    img[sy2:sy2 + panel_h, sx4:sx4 + panel_w] = p4
    put_text(img, "4. Flatness (40% weight)", (sx4, sy2 + panel_h + 15), 0.43, WHITE, 1)
    put_text(img, f"   HOT = flat (good seal)", (sx4, sy2 + panel_h + 32), 0.33, ORANGE)
    put_text(img, f"   DARK = tilted (bad seal)", (sx4, sy2 + panel_h + 47), 0.33, GRAY)

    # ── PANEL 5: Combined score ──
    score_vis = np.zeros((ch, cw), dtype=np.uint8)
    if np.any(score > 0):
        score_norm = score / max(score.max(), 1)
        score_vis[top_surface] = (score_norm[top_surface] * 255).astype(np.uint8)
    p5 = cv2.applyColorMap(score_vis, cv2.COLORMAP_INFERNO)
    p5[crop_mask == 0] = np.array(BG)
    p5[~top_surface & (crop_mask > 0)] = np.array([30, 30, 30])
    cv2.drawContours(p5, cnts, -1, (80, 80, 80), 1)
    # Mark the best point
    cv2.drawMarker(p5, (gx, gy), GREEN, cv2.MARKER_CROSS, 15, 2)
    p5 = resize_p(p5)
    sx5 = sx4 + panel_w + gap
    img[sy2:sy2 + panel_h, sx5:sx5 + panel_w] = p5
    put_text(img, "5. Score = 40%flat+30%edge+30%ctr", (sx5, sy2 + panel_h + 15), 0.38, WHITE, 1)
    put_text(img, f"   BRIGHT = best grasp spot", (sx5, sy2 + panel_h + 32), 0.33, YELLOW)
    put_text(img, f"   GREEN cross = maximum", (sx5, sy2 + panel_h + 47), 0.33, GREEN)

    # ── PANEL 6: Final result with both points ──
    p6 = crop_rgb.copy()
    p6[crop_mask > 0] = (p6[crop_mask > 0] * 0.5 + np.array([80, 130, 80]) * 0.5).astype(np.uint8)
    cv2.drawContours(p6, cnts, -1, WHITE, 1)

    # RED dot (2D geometric center)
    cv2.circle(p6, (cx_r, cy_r), 7, RED, -1)
    cv2.circle(p6, (cx_r, cy_r), 7, WHITE, 2)
    put_text(p6, "CENTER", (cx_r - 32, cy_r - 12), 0.3, RED, 1)
    put_text(p6, "(2D only)", (cx_r - 28, cy_r + 20), 0.25, RED)

    # GREEN cross (3D suction grasp)
    cv2.drawMarker(p6, (gx, gy), GREEN, cv2.MARKER_CROSS, 22, 2)
    put_text(p6, "GRASP", (gx + 12, gy - 5), 0.3, GREEN, 1)
    put_text(p6, "(3D depth)", (gx + 12, gy + 12), 0.25, GREEN)

    # White line if different
    if dist_cg > 3:
        cv2.line(p6, (cx_r, cy_r), (gx, gy), WHITE, 1, cv2.LINE_AA)
        mid_x = (cx_r + gx) // 2
        mid_y = (cy_r + gy) // 2
        put_text(p6, f"{dist_cg:.0f}px", (mid_x + 5, mid_y - 8), 0.3, YELLOW, 1)

    p6 = resize_p(p6)
    sx6 = sx5 + panel_w + gap
    img[sy2:sy2 + panel_h, sx6:sx6 + panel_w] = p6
    put_text(img, "6. RESULT: 2D vs 3D", (sx6, sy2 + panel_h + 15), 0.43, WHITE, 1)
    put_text(img, f"   RED=({global_cx},{global_cy}) 2D only", (sx6, sy2 + panel_h + 32), 0.33, RED)
    put_text(img, f"   GREEN=({global_gx},{global_gy}) d={depth_at_grasp:.0f}mm", (sx6, sy2 + panel_h + 47), 0.33, GREEN)

    # Bottom summary
    put_text(img, f"Score={best_score:.2f}  (flat={s_flat:.2f} edge={s_edge:.2f} center={s_center:.2f})  "
             f"Offset RED->GREEN = {dist_cg:.1f}px",
             (20, canvas_h - 10), 0.38, YELLOW)

    cv2.imwrite(out_path, img)
    print(f"  Saved: {out_path}")

    return {
        'obj_idx': obj_idx,
        'shape': shape_label,
        'depth_range': depth_range,
        'top_pixels': top_count,
        'center_2d': (global_cx, global_cy),
        'grasp_3d': (global_gx, global_gy),
        'offset': dist_cg,
        'score': best_score,
        'depth_at_grasp': depth_at_grasp,
        'flat': s_flat,
        'edge': s_edge,
        'ctr': s_center,
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
        out_path = os.path.join(OUT, f"15_3d_pipeline_obj{obj_idx}.png")
        result = generate_3d_pipeline(rgb, depth_mm, masks, obj_idx, out_path)
        results.append(result)

    # Summary
    print(f"\n{'='*90}")
    print(f"  3D PIPELINE SUMMARY: All Objects in Frame 10")
    print(f"{'='*90}")
    print(f"  {'Obj':>3} | {'Shape':>25} | {'Depth Range':>11} | {'Top Px':>7} | {'2D Center':>12} | {'3D Grasp':>12} | {'Offset':>7} | {'Score':>5}")
    print(f"  {'-'*3}-+-{'-'*25}-+-{'-'*11}-+-{'-'*7}-+-{'-'*12}-+-{'-'*12}-+-{'-'*7}-+-{'-'*5}")
    for r in results:
        print(f"  #{r['obj_idx']:>2} | {r['shape']:>25} | {r['depth_range']:>7.0f}mm   | {r['top_pixels']:>7} "
              f"| ({r['center_2d'][0]:>3},{r['center_2d'][1]:>3}) "
              f"| ({r['grasp_3d'][0]:>3},{r['grasp_3d'][1]:>3}) "
              f"| {r['offset']:>5.1f}px | {r['score']:>5.2f}")

    print(f"\n  Key insight: 3D depth shifts the grasp point to the FLATTEST spot on the TOP surface.")
    print(f"  Without depth, GREEN = RED (same point). With depth, they can differ by up to {max(r['offset'] for r in results):.0f}px.")
    print(f"\nAll images saved to: {OUT}/")


if __name__ == "__main__":
    main()
