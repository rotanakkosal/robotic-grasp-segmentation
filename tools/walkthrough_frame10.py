#!/usr/bin/env python3
"""
EXACT step-by-step walkthrough of finding the central point
using T-Less frame 10 as a real example.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import json
import glob
from skimage.morphology import medial_axis

# ============================================================
# STEP 0: Load the data (after segmentation)
# ============================================================
scene_dir = "/home/kosal/cbnu_project/AI/picking-arm-robot/dataset/t_less-data/000003"
frame_id = 10
fid = f"{frame_id:06d}"

rgb = cv2.imread(os.path.join(scene_dir, "rgb", f"{fid}.png"))
depth_raw = cv2.imread(os.path.join(scene_dir, "depth", f"{fid}.png"), cv2.IMREAD_UNCHANGED)
with open(os.path.join(scene_dir, "scene_camera.json")) as f:
    cam = json.load(f)[str(frame_id)]
depth_mm = depth_raw.astype(np.float32) * cam["depth_scale"]

mask_files = sorted(glob.glob(os.path.join(scene_dir, "mask", f"{fid}_*.png")))
masks = [cv2.imread(mf, cv2.IMREAD_GRAYSCALE) for mf in mask_files]

H, W = rgb.shape[:2]
print(f"Image: {W}x{H}")
print(f"Depth range: {depth_mm[depth_mm>0].min():.0f} - {depth_mm[depth_mm>0].max():.0f} mm")
print(f"Objects detected: {len(masks)}")
print()

# Process each object
for obj_idx, mask in enumerate(masks):
    mask_uint8 = (mask > 0).astype(np.uint8)
    area = np.sum(mask_uint8)
    if area < 100:
        continue

    coords = np.where(mask_uint8 > 0)
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()

    print(f"{'='*70}")
    print(f"  OBJECT #{obj_idx}")
    print(f"  Mask area: {area} pixels")
    print(f"  Bounding box: x=[{x_min},{x_max}] y=[{y_min},{y_max}]")
    print(f"{'='*70}")

    # ============================================================
    # STEP 1: ANALYZE THE SHAPE
    # ============================================================
    print(f"\n  STEP 1: Analyze shape")
    print(f"  ─────────────────────")

    # Find contour
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)

    # Circularity
    circularity = (4 * np.pi * contour_area) / (perimeter * perimeter) if perimeter > 0 else 0
    print(f"    circularity = 4π × {contour_area:.0f} / {perimeter:.0f}² = {circularity:.3f}")
    print(f"    (1.0 = perfect circle, lower = more irregular)")

    # Aspect ratio
    rect = cv2.minAreaRect(largest)
    w_rect, h_rect = rect[1]
    if h_rect > 0 and w_rect > 0:
        aspect_ratio = max(w_rect, h_rect) / min(w_rect, h_rect)
    else:
        aspect_ratio = 1.0
    print(f"    aspect_ratio = {max(w_rect,h_rect):.0f} / {min(w_rect,h_rect):.0f} = {aspect_ratio:.2f}")

    # Solidity
    hull = cv2.convexHull(largest)
    hull_area = cv2.contourArea(hull)
    solidity = contour_area / hull_area if hull_area > 0 else 1.0
    print(f"    solidity = {contour_area:.0f} / {hull_area:.0f} = {solidity:.3f}")

    is_circular = circularity > 0.7
    is_elongated = aspect_ratio > 2.0

    print(f"\n    is_circular?  {circularity:.3f} > 0.7 → {'YES ●' if is_circular else 'NO'}")
    print(f"    is_elongated? {aspect_ratio:.2f} > 2.0 → {'YES ─' if is_elongated else 'NO'}")

    # ============================================================
    # STEP 2: CHOOSE METHOD & COMPUTE CENTER
    # ============================================================
    print(f"\n  STEP 2: Compute geometric center")
    print(f"  ──────────────────────────────────")

    if is_circular:
        print(f"    Shape is CIRCULAR → use Distance Transform")
        print()

        # Distance transform
        dt = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
        _, max_val, _, max_loc = cv2.minMaxLoc(dt)
        cx, cy = float(max_loc[0]), float(max_loc[1])

        print(f"    Distance Transform:")
        print(f"      Max inscribed circle radius = {max_val:.1f} pixels")
        print(f"      Center of that circle = ({cx:.0f}, {cy:.0f})")
        print(f"    ✓ Result: RED DOT at ({cx:.0f}, {cy:.0f})")

    else:
        print(f"    Shape is NOT circular → use Moments (center of mass)")
        print()

        # Moments
        M = cv2.moments(mask_uint8)
        mx = M["m10"] / M["m00"]
        my = M["m01"] / M["m00"]

        print(f"    Moments calculation:")
        print(f"      Total mass (m00) = {M['m00']:.0f} pixels")
        print(f"      Sum of x positions (m10) = {M['m10']:.0f}")
        print(f"      Sum of y positions (m01) = {M['m01']:.0f}")
        print(f"      center_x = {M['m10']:.0f} / {M['m00']:.0f} = {mx:.1f}")
        print(f"      center_y = {M['m01']:.0f} / {M['m00']:.0f} = {my:.1f}")

        # STEP 3: Safety check
        print(f"\n  STEP 3: Safety check")
        print(f"  ─────────────────────")

        dt = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
        max_dt = np.max(dt)

        mx_int = int(round(np.clip(mx, 0, W-1)))
        my_int = int(round(np.clip(my, 0, H-1)))

        inside = mask_uint8[my_int, mx_int] > 0
        edge_dist = dt[my_int, mx_int] if inside else 0.0
        min_clearance = max_dt * 0.20

        print(f"    Is ({mx:.0f},{my:.0f}) inside the mask? → {'YES' if inside else 'NO'}")
        print(f"    Distance from nearest edge = {edge_dist:.1f} px")
        print(f"    Min required clearance = {max_dt:.1f} × 20% = {min_clearance:.1f} px")

        if inside and edge_dist >= min_clearance:
            print(f"    {edge_dist:.1f} >= {min_clearance:.1f} → SAFE! Use moments directly")
            cx, cy = mx, my
        elif inside:
            print(f"    {edge_dist:.1f} < {min_clearance:.1f} → Too close to edge!")
            _, _, _, dt_max_loc = cv2.minMaxLoc(dt)
            dt_cx, dt_cy = float(dt_max_loc[0]), float(dt_max_loc[1])
            edge_ratio = edge_dist / max_dt
            dt_weight = min(1.0 - (edge_ratio / 0.20), 0.7)
            cx = (1.0 - dt_weight) * mx + dt_weight * dt_cx
            cy = (1.0 - dt_weight) * my + dt_weight * dt_cy
            print(f"    Blending {(1-dt_weight)*100:.0f}% moments + {dt_weight*100:.0f}% DT max")
            print(f"    Moments=({mx:.0f},{my:.0f}) + DT max=({dt_cx:.0f},{dt_cy:.0f})")
        else:
            print(f"    Point is OUTSIDE mask! Finding nearest mask point...")
            _, _, _, dt_max_loc = cv2.minMaxLoc(dt)
            dt_cx, dt_cy = float(dt_max_loc[0]), float(dt_max_loc[1])
            distances = (coords[1] - mx)**2 + (coords[0] - my)**2
            nearest_idx = np.argmin(distances)
            nearest_x = float(coords[1][nearest_idx])
            nearest_y = float(coords[0][nearest_idx])
            cx = 0.5 * nearest_x + 0.5 * dt_cx
            cy = 0.5 * nearest_y + 0.5 * dt_cy
            print(f"    Nearest mask point=({nearest_x:.0f},{nearest_y:.0f})")
            print(f"    DT max=({dt_cx:.0f},{dt_cy:.0f})")
            print(f"    Blend 50/50")

        print(f"\n    ✓ Result: RED DOT at ({cx:.0f}, {cy:.0f})")

    print()

# ============================================================
# Create visualization
# ============================================================
print("\nGenerating visualization...")

vis = rgb.copy()
np.random.seed(42)

for obj_idx, mask in enumerate(masks):
    mask_bool = mask > 0
    if np.sum(mask_bool) < 100:
        continue

    # Draw mask overlay
    color = np.random.randint(80, 220, 3).tolist()
    overlay = vis.copy()
    overlay[mask_bool] = (overlay[mask_bool] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
    vis = overlay
    contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (255, 255, 255), 1)

    # Import and compute
    from centroid_utils import compute_centroid_adaptive, compute_suction_grasp_point

    center = compute_centroid_adaptive(mask)
    grasp = compute_suction_grasp_point(mask, depth_mm, suction_cup_radius=15)

    # Draw RED dot (geometric center)
    if center:
        cx, cy = int(center[0]), int(center[1])
        cv2.circle(vis, (cx, cy), 7, (0, 0, 255), -1)
        cv2.circle(vis, (cx, cy), 7, (255, 255, 255), 2)

    # Draw GREEN cross (suction grasp)
    if grasp:
        gx, gy = int(grasp[0]), int(grasp[1])
        cv2.drawMarker(vis, (gx, gy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

    # Connect with white line if different
    if center and grasp:
        dist = np.sqrt((center[0]-grasp[0])**2 + (center[1]-grasp[1])**2)
        if dist > 5:
            cv2.line(vis, (int(center[0]), int(center[1])), (int(grasp[0]), int(grasp[1])), (255,255,255), 1)

    # Label
    if center:
        cv2.putText(vis, f"#{obj_idx}", (int(center[0])+10, int(center[1])-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

cv2.putText(vis, "RED = Center | GREEN = Grasp", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
cv2.imwrite("./output_tless_dual/walkthrough_frame10.png", vis)
print("Saved: ./output_tless_dual/walkthrough_frame10.png")
