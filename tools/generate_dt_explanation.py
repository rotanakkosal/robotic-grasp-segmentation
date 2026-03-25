#!/usr/bin/env python3
"""
Generate step-by-step visualization of how Distance Transform works.
Uses real T-Less frame 10 objects + a simple synthetic example.
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


# ============================================================
# 1. Simple grid example — show DT values as numbers
# ============================================================
def vis_dt_grid():
    """Show DT on a simple grid with actual distance values visible."""
    cell = 40  # pixel size of each grid cell
    # Create a simple shape: 11x7 grid with an oval-ish mask
    rows, cols = 9, 13
    mask_grid = np.zeros((rows, cols), dtype=np.uint8)
    # Draw a rough oval
    for r in range(rows):
        for c in range(cols):
            # Ellipse equation
            cy, cx = rows // 2, cols // 2
            if ((r - cy) / 3.5)**2 + ((c - cx) / 5.5)**2 <= 1.0:
                mask_grid[r, c] = 1

    # Compute DT on a scaled-up version for accuracy
    scale = 10
    mask_big = np.kron(mask_grid, np.ones((scale, scale), dtype=np.uint8))
    dt_big = cv2.distanceTransform(mask_big, cv2.DIST_L2, 5)
    # Sample DT values at cell centers
    dt_values = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            if mask_grid[r, c]:
                dt_values[r, c] = dt_big[r * scale + scale // 2, c * scale + scale // 2] / scale

    max_dt = dt_values.max()

    W = cols * cell + 80
    H = rows * cell + 120
    img = np.full((H, W, 3), BG, dtype=np.uint8)
    put_text_center(img, "Distance Transform: Each number = distance to nearest edge", 22, 0.45, YELLOW, 1)

    ox, oy = 40, 40  # offset

    for r in range(rows):
        for c in range(cols):
            x1, y1 = ox + c * cell, oy + r * cell
            x2, y2 = x1 + cell, y1 + cell

            if mask_grid[r, c]:
                val = dt_values[r, c]
                # Color intensity based on distance
                t = val / max(max_dt, 1)
                # Blue (edge) → Red (center) colormap
                b = int(255 * (1 - t))
                g = int(80 * t)
                red = int(255 * t)
                cv2.rectangle(img, (x1 + 1, y1 + 1), (x2 - 1, y2 - 1), (b, g, red), -1)
                # Write the distance value
                txt = f"{val:.1f}"
                txt_color = WHITE if t < 0.5 else YELLOW
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
                cv2.putText(img, txt, (x1 + (cell - tw) // 2, y1 + (cell + th) // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, txt_color, 1, cv2.LINE_AA)
                # Mark the max
                if abs(val - max_dt) < 0.05:
                    cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, 2)
            else:
                cv2.rectangle(img, (x1 + 1, y1 + 1), (x2 - 1, y2 - 1), (20, 20, 20), -1)
                put_text(img, "0", (x1 + cell // 2 - 5, y1 + cell // 2 + 5), 0.3, (60, 60, 60))

            # Grid lines
            cv2.rectangle(img, (x1, y1), (x2, y2), (50, 50, 50), 1)

    put_text(img, f"MAX = {max_dt:.1f} (green box) = center of largest inscribed circle",
             (ox, oy + rows * cell + 20), 0.4, GREEN)
    put_text(img, "BLUE = near edge (small distance)    RED = far from edge (large distance)",
             (ox - 20, oy + rows * cell + 40), 0.35, GRAY)
    put_text(img, "Each cell shows: how far (in cells) is the nearest background pixel?",
             (ox - 20, oy + rows * cell + 58), 0.35, GRAY)

    cv2.imwrite(os.path.join(OUT, "10_dt_grid_example.png"), img)
    print("Saved 10_dt_grid_example.png")


# ============================================================
# 2. DT on a circle — show inscribed circle
# ============================================================
def vis_dt_circle():
    """DT on a circle shape with inscribed circle visualization."""
    size = 300
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (150, 150), 110, 255, -1)

    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, max_val, _, max_loc = cv2.minMaxLoc(dt)

    W, H = 750, 380
    img = np.full((H, W, 3), BG, dtype=np.uint8)
    put_text_center(img, "DT on CIRCLE: Max = center of inscribed circle", 25, 0.55, YELLOW, 2)

    # Panel 1: Binary mask
    p1 = np.full((size, size, 3), BG, dtype=np.uint8)
    p1[mask > 0] = [80, 80, 80]
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(p1, cnts, -1, WHITE, 2)
    put_text(p1, "Binary Mask", (90, 20), 0.5, WHITE, 1)

    sx1, sy1 = 20, 55
    p1s = cv2.resize(p1, (220, 220))
    img[sy1:sy1 + 220, sx1:sx1 + 220] = p1s

    # Panel 2: DT heatmap
    dt_norm = (dt / max(max_val, 1) * 255).astype(np.uint8)
    p2 = cv2.applyColorMap(dt_norm, cv2.COLORMAP_JET)
    p2[mask == 0] = np.array(BG)
    cv2.drawContours(p2, cnts, -1, WHITE, 1)
    # Draw concentric distance rings
    for ring_val in [20, 40, 60, 80, 100]:
        ring_mask = (dt >= ring_val - 0.5) & (dt <= ring_val + 0.5) & (mask > 0)
        p2[ring_mask] = [200, 200, 200]
    cv2.circle(p2, max_loc, 5, WHITE, -1)
    put_text(p2, "Distance Transform", (55, 20), 0.5, WHITE, 1)
    put_text(p2, "rings = equal distance", (60, size - 15), 0.35, GRAY)

    sx2 = 260
    p2s = cv2.resize(p2, (220, 220))
    img[sy1:sy1 + 220, sx2:sx2 + 220] = p2s

    # Panel 3: Result with inscribed circle
    p3 = np.full((size, size, 3), BG, dtype=np.uint8)
    p3[mask > 0] = [60, 80, 60]
    cv2.drawContours(p3, cnts, -1, WHITE, 2)
    # Inscribed circle
    cv2.circle(p3, max_loc, int(max_val), GREEN, 2, cv2.LINE_AA)
    cv2.circle(p3, max_loc, 6, RED, -1)
    cv2.circle(p3, max_loc, 6, WHITE, 2)
    # Radius line
    edge_x = max_loc[0] + int(max_val)
    cv2.arrowedLine(p3, max_loc, (edge_x, max_loc[1]), YELLOW, 2, tipLength=0.1)
    put_text(p3, f"r={max_val:.0f}px", (max_loc[0] + 10, max_loc[1] - 10), 0.4, YELLOW)
    put_text(p3, "Inscribed Circle", (65, 20), 0.5, WHITE, 1)
    put_text(p3, f"Center ({max_loc[0]},{max_loc[1]})", (80, size - 15), 0.4, GREEN)

    sx3 = 500
    p3s = cv2.resize(p3, (220, 220))
    img[sy1:sy1 + 220, sx3:sx3 + 220] = p3s

    # Arrows between panels
    cv2.arrowedLine(img, (sx1 + 220 + 5, sy1 + 110), (sx2 - 5, sy1 + 110), GRAY, 2, tipLength=0.15)
    cv2.arrowedLine(img, (sx2 + 220 + 5, sy1 + 110), (sx3 - 5, sy1 + 110), GRAY, 2, tipLength=0.15)

    put_text(img, "For a CIRCLE: the inscribed circle center = geometric center = DT max", (20, H - 60), 0.42, GREEN, 1)
    put_text(img, f"DT max = {max_val:.0f}px at ({max_loc[0]}, {max_loc[1]})  =  exact center of the circle", (20, H - 35), 0.42, WHITE)
    put_text(img, "Every direction from center to edge is the SAME distance -> DT works perfectly!", (20, H - 12), 0.38, GRAY)

    cv2.imwrite(os.path.join(OUT, "11_dt_on_circle.png"), img)
    print("Saved 11_dt_on_circle.png")


# ============================================================
# 3. DT on bottle shape — show WHY it fails
# ============================================================
def vis_dt_bottle():
    """DT on a bottle-like shape showing inscribed circles at different positions."""
    size_h, size_w = 250, 450
    mask = np.zeros((size_h, size_w), dtype=np.uint8)
    # Bottle: fat left end, thin right body
    cv2.circle(mask, (80, 125), 90, 255, -1)
    cv2.ellipse(mask, (250, 125), (200, 45), 0, 0, 360, 255, -1)

    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, max_val, _, max_loc = cv2.minMaxLoc(dt)

    # Moments center
    M = cv2.moments(mask)
    mom_x = int(M["m10"] / M["m00"])
    mom_y = int(M["m01"] / M["m00"])

    W, H = 900, 650
    img = np.full((H, W, 3), BG, dtype=np.uint8)
    put_text_center(img, "DT on BOTTLE SHAPE: Why it gives WRONG center", 28, 0.6, YELLOW, 2)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Panel 1: DT heatmap
    dt_norm = (dt / max(max_val, 1) * 255).astype(np.uint8)
    p1 = cv2.applyColorMap(dt_norm, cv2.COLORMAP_JET)
    p1[mask == 0] = np.array(BG)
    cv2.drawContours(p1, cnts, -1, WHITE, 1)
    # Mark DT max
    cv2.circle(p1, max_loc, 6, WHITE, -1)

    sx1, sy1 = 20, 60
    img[sy1:sy1 + size_h, sx1:sx1 + size_w] = p1
    put_text(img, "Distance Transform Heatmap", (sx1 + 100, sy1 - 8), 0.5, WHITE, 1)
    put_text(img, f"DT max = {max_val:.0f}px at ({max_loc[0]},{max_loc[1]})", (sx1, sy1 + size_h + 18), 0.42, WHITE)
    put_text(img, "HOT = far from edge,  BLUE = near edge", (sx1, sy1 + size_h + 38), 0.38, GRAY)

    # Panel 2: Show inscribed circles at different positions
    p2 = np.full((size_h, size_w, 3), BG, dtype=np.uint8)
    p2[mask > 0] = [50, 60, 50]
    cv2.drawContours(p2, cnts, -1, WHITE, 1)

    # Draw inscribed circle at DT max (fat end)
    cv2.circle(p2, max_loc, int(max_val), RED, 2, cv2.LINE_AA)
    cv2.circle(p2, max_loc, 6, RED, -1)
    cv2.circle(p2, max_loc, 6, WHITE, 2)
    put_text(p2, f"r={max_val:.0f}", (max_loc[0] + 8, int(max_loc[1] - max_val + 15)), 0.35, RED)

    # Draw inscribed circle at moments center (true center)
    mom_dt = dt[mom_y, mom_x]
    cv2.circle(p2, (mom_x, mom_y), int(mom_dt), GREEN, 2, cv2.LINE_AA)
    cv2.circle(p2, (mom_x, mom_y), 6, GREEN, -1)
    cv2.circle(p2, (mom_x, mom_y), 6, WHITE, 2)
    put_text(p2, f"r={mom_dt:.0f}", (mom_x + 8, mom_y - int(mom_dt) + 15), 0.35, GREEN)

    # Draw inscribed circle at a mid point
    mid_x = (max_loc[0] + mom_x) // 2
    mid_y = 125
    mid_dt = dt[mid_y, mid_x]
    cv2.circle(p2, (mid_x, mid_y), int(mid_dt), CYAN, 1, cv2.LINE_AA)
    cv2.circle(p2, (mid_x, mid_y), 4, CYAN, -1)
    put_text(p2, f"r={mid_dt:.0f}", (mid_x + 5, mid_y - int(mid_dt) + 12), 0.3, CYAN)

    sx2 = 20
    sy2 = sy1 + size_h + 55
    img[sy2:sy2 + size_h, sx2:sx2 + size_w] = p2
    put_text(img, "Inscribed Circles at Different Positions", (sx2 + 80, sy2 - 8), 0.5, WHITE, 1)

    # Legend on the right
    lx = sx2 + size_w + 30
    ly = sy1 + 20

    put_text(img, "The KEY problem:", (lx, ly), 0.5, YELLOW, 1)
    ly += 35
    put_text(img, "DT picks the point where the", (lx, ly), 0.42, WHITE)
    ly += 20
    put_text(img, "LARGEST circle fits inside.", (lx, ly), 0.42, WHITE)
    ly += 35

    put_text(img, f"RED circle (DT max):", (lx, ly), 0.42, RED, 1)
    ly += 20
    put_text(img, f"  r = {max_val:.0f}px at fat end", (lx, ly), 0.4, RED)
    ly += 20
    put_text(img, f"  Biggest circle fits here", (lx, ly), 0.38, GRAY)
    ly += 30

    put_text(img, f"GREEN circle (true center):", (lx, ly), 0.42, GREEN, 1)
    ly += 20
    put_text(img, f"  r = {mom_dt:.0f}px at moments center", (lx, ly), 0.4, GREEN)
    ly += 20
    put_text(img, f"  Smaller circle = lower DT", (lx, ly), 0.38, GRAY)
    ly += 30

    dist = np.sqrt((max_loc[0] - mom_x)**2 + (max_loc[1] - mom_y)**2)
    put_text(img, f"Error = {dist:.0f}px!", (lx, ly), 0.5, ORANGE, 2)
    ly += 25
    put_text(img, f"DT picks fat end, NOT center", (lx, ly), 0.42, ORANGE)

    ly += 40
    put_text(img, "This is why we use Moments", (lx, ly), 0.42, GREEN, 1)
    ly += 20
    put_text(img, "for non-circular shapes.", (lx, ly), 0.42, GREEN, 1)

    cv2.imwrite(os.path.join(OUT, "12_dt_on_bottle.png"), img)
    print("Saved 12_dt_on_bottle.png")


# ============================================================
# 4. DT on real T-Less objects — all 5
# ============================================================
def vis_dt_real_objects():
    """Show DT heatmap + inscribed circle + DT max for all 5 real T-Less objects."""
    scene_dir = "/home/kosal/cbnu_project/AI/picking-arm-robot/dataset/t_less-data/000003"
    frame_id = 10
    fid = f"{frame_id:06d}"

    rgb = cv2.imread(os.path.join(scene_dir, "rgb", f"{fid}.png"))
    mask_files = sorted(glob.glob(os.path.join(scene_dir, "mask", f"{fid}_*.png")))
    masks = [cv2.imread(mf, cv2.IMREAD_GRAYSCALE) for mf in mask_files]

    H_img, W_img = rgb.shape[:2]

    panel_w, panel_h = 180, 160
    n_obj = len(masks)
    canvas_w = panel_w * n_obj + (n_obj + 1) * 15
    canvas_h = panel_h * 2 + 130
    img = np.full((canvas_h, canvas_w, 3), BG, dtype=np.uint8)
    put_text_center(img, "Distance Transform on ALL 5 Real T-Less Objects", 25, 0.55, YELLOW, 2)
    put_text_center(img, "Row 1: DT heatmap    Row 2: Inscribed circle (WHITE) + DT max (RED)", 48, 0.38, GRAY)

    for obj_idx, mask in enumerate(masks):
        mask_uint8 = (mask > 0).astype(np.uint8)
        if np.sum(mask_uint8) < 100:
            continue

        # Crop
        coords = np.where(mask_uint8 > 0)
        y_min, y_max = coords[0].min() - 10, coords[0].max() + 10
        x_min, x_max = coords[1].min() - 10, coords[1].max() + 10
        y_min, x_min = max(0, y_min), max(0, x_min)
        y_max, x_max = min(H_img, y_max), min(W_img, x_max)

        crop_mask = mask_uint8[y_min:y_max, x_min:x_max]
        crop_rgb = rgb[y_min:y_max, x_min:x_max].copy()

        dt = cv2.distanceTransform(crop_mask, cv2.DIST_L2, 5)
        max_dt = dt.max()
        _, _, _, dt_loc = cv2.minMaxLoc(dt)
        dtx, dty = dt_loc

        # Moments
        M_obj = cv2.moments(crop_mask)
        mom_x = int(M_obj["m10"] / M_obj["m00"])
        mom_y = int(M_obj["m01"] / M_obj["m00"])

        cnts, _ = cv2.findContours(crop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(cnts, key=cv2.contourArea)
        c_area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        circ = (4 * np.pi * c_area) / (peri * peri) if peri > 0 else 0
        is_circ = circ > 0.7

        sx = 15 + obj_idx * (panel_w + 15)

        # Row 1: DT heatmap
        dt_norm = (dt / max(max_dt, 1) * 255).astype(np.uint8)
        p_dt = cv2.applyColorMap(dt_norm, cv2.COLORMAP_JET)
        p_dt[crop_mask == 0] = np.array(BG)
        cv2.drawContours(p_dt, cnts, -1, WHITE, 1)
        cv2.circle(p_dt, dt_loc, 4, WHITE, -1)
        p_dt = cv2.resize(p_dt, (panel_w, panel_h))

        sy_r1 = 60
        img[sy_r1:sy_r1 + panel_h, sx:sx + panel_w] = p_dt

        # Row 2: Inscribed circle visualization
        p_ic = crop_rgb.copy()
        p_ic[crop_mask > 0] = (p_ic[crop_mask > 0] * 0.4 + np.array([60, 60, 60]) * 0.6).astype(np.uint8)
        cv2.drawContours(p_ic, cnts, -1, WHITE, 1)
        # Inscribed circle at DT max
        cv2.circle(p_ic, (dtx, dty), int(max_dt), WHITE, 1, cv2.LINE_AA)
        cv2.circle(p_ic, (dtx, dty), 5, RED, -1)
        cv2.circle(p_ic, (dtx, dty), 5, WHITE, 2)
        # Moments center
        cv2.circle(p_ic, (mom_x, mom_y), 5, GREEN, -1)
        # Distance line
        dist_dm = np.sqrt((dtx - mom_x)**2 + (dty - mom_y)**2)
        if dist_dm > 5:
            cv2.line(p_ic, (dtx, dty), (mom_x, mom_y), YELLOW, 1)
        p_ic = cv2.resize(p_ic, (panel_w, panel_h))

        sy_r2 = sy_r1 + panel_h + 5
        img[sy_r2:sy_r2 + panel_h, sx:sx + panel_w] = p_ic

        # Labels
        circ_color = GREEN if is_circ else ORANGE
        shape_txt = "CIRCULAR" if is_circ else "not circ"
        put_text(img, f"#{obj_idx} circ={circ:.2f}", (sx, sy_r2 + panel_h + 15), 0.35, circ_color)
        put_text(img, f"{shape_txt}", (sx, sy_r2 + panel_h + 30), 0.33, circ_color)
        put_text(img, f"r={max_dt:.0f}px", (sx, sy_r2 + panel_h + 45), 0.33, WHITE)
        err_txt = f"err={dist_dm:.0f}px"
        err_color = GREEN if dist_dm < 10 else RED
        put_text(img, err_txt, (sx + 80, sy_r2 + panel_h + 45), 0.33, err_color)

    put_text(img, "RED = DT max    GREEN = true center (moments)    WHITE circle = inscribed circle at DT max",
             (15, canvas_h - 10), 0.35, GRAY)

    cv2.imwrite(os.path.join(OUT, "13_dt_real_objects.png"), img)
    print("Saved 13_dt_real_objects.png")


# ============================================================
# 5. Step-by-step DT process visualization
# ============================================================
def vis_dt_steps():
    """Show the DT computation process step by step on a real object."""
    scene_dir = "/home/kosal/cbnu_project/AI/picking-arm-robot/dataset/t_less-data/000003"
    frame_id = 10
    fid = f"{frame_id:06d}"

    rgb = cv2.imread(os.path.join(scene_dir, "rgb", f"{fid}.png"))
    mask_files = sorted(glob.glob(os.path.join(scene_dir, "mask", f"{fid}_*.png")))

    # Use object #4 (circular) and #1 (elongated) side by side
    mask_circ = cv2.imread(mask_files[4], cv2.IMREAD_GRAYSCALE)
    mask_elong = cv2.imread(mask_files[1], cv2.IMREAD_GRAYSCALE)
    H_img, W_img = rgb.shape[:2]

    def crop_object(mask_raw):
        m = (mask_raw > 0).astype(np.uint8)
        coords = np.where(m > 0)
        y1, y2 = max(0, coords[0].min() - 10), min(H_img, coords[0].max() + 10)
        x1, x2 = max(0, coords[1].min() - 10), min(W_img, coords[1].max() + 10)
        return m[y1:y2, x1:x2], rgb[y1:y2, x1:x2].copy()

    mask_c, rgb_c = crop_object(mask_circ)
    mask_e, rgb_e = crop_object(mask_elong)

    panel_w, panel_h = 200, 180
    canvas_w = panel_w * 4 + 100
    canvas_h = panel_h * 2 + 170
    img = np.full((canvas_h, canvas_w, 3), BG, dtype=np.uint8)
    put_text_center(img, "DT Step-by-Step: CIRCULAR vs ELONGATED (real T-Less objects)", 25, 0.5, YELLOW, 2)

    def resize_p(src):
        return cv2.resize(src, (panel_w, panel_h), interpolation=cv2.INTER_LINEAR)

    def process_row(mask_crop, rgb_crop, row_y, label, label_color):
        cnts, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dt = cv2.distanceTransform(mask_crop, cv2.DIST_L2, 5)
        max_dt = dt.max()
        _, _, _, dt_loc = cv2.minMaxLoc(dt)
        M = cv2.moments(mask_crop)
        mom_x, mom_y = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        dist_err = np.sqrt((dt_loc[0] - mom_x)**2 + (dt_loc[1] - mom_y)**2)

        # Step A: Binary mask
        pA = np.full((*mask_crop.shape, 3), BG, dtype=np.uint8)
        pA[mask_crop > 0] = [200, 200, 200]
        cv2.drawContours(pA, cnts, -1, WHITE, 1)
        pA = resize_p(pA)
        sx = 20
        img[row_y:row_y + panel_h, sx:sx + panel_w] = pA
        put_text(img, "A. Binary Mask", (sx, row_y - 5), 0.38, WHITE)

        # Step B: Edge detection (pixels with distance=1)
        edge_mask = (dt > 0) & (dt <= 2)
        pB = np.full((*mask_crop.shape, 3), BG, dtype=np.uint8)
        pB[mask_crop > 0] = [40, 40, 40]
        pB[edge_mask] = [0, 0, 255]  # Red = edge pixels
        pB = resize_p(pB)
        sx2 = 20 + panel_w + 20
        img[row_y:row_y + panel_h, sx2:sx2 + panel_w] = pB
        put_text(img, "B. Find Edges (red)", (sx2, row_y - 5), 0.38, RED)

        # Step C: DT heatmap
        dt_norm = (dt / max(max_dt, 1) * 255).astype(np.uint8)
        pC = cv2.applyColorMap(dt_norm, cv2.COLORMAP_JET)
        pC[mask_crop == 0] = np.array(BG)
        # Draw distance rings
        for ring in np.arange(10, max_dt, max_dt / 5):
            ring_m = (dt >= ring - 0.5) & (dt <= ring + 0.5)
            pC[ring_m] = [180, 180, 180]
        pC = resize_p(pC)
        sx3 = 20 + (panel_w + 20) * 2
        img[row_y:row_y + panel_h, sx3:sx3 + panel_w] = pC
        put_text(img, "C. Measure Distance", (sx3, row_y - 5), 0.38, WHITE)

        # Step D: Result — inscribed circle
        pD = rgb_crop.copy()
        pD[mask_crop > 0] = (pD[mask_crop > 0] * 0.4 + np.array([60, 60, 60]) * 0.6).astype(np.uint8)
        cv2.drawContours(pD, cnts, -1, WHITE, 1)
        # Inscribed circle
        cv2.circle(pD, dt_loc, int(max_dt), WHITE, 2, cv2.LINE_AA)
        cv2.circle(pD, dt_loc, 6, RED, -1)
        cv2.circle(pD, dt_loc, 6, WHITE, 2)
        # Moments
        cv2.circle(pD, (mom_x, mom_y), 5, GREEN, -1)
        if dist_err > 5:
            cv2.arrowedLine(pD, dt_loc, (mom_x, mom_y), YELLOW, 2, tipLength=0.15)
        pD = resize_p(pD)
        sx4 = 20 + (panel_w + 20) * 3
        img[row_y:row_y + panel_h, sx4:sx4 + panel_w] = pD
        put_text(img, "D. Find Maximum", (sx4, row_y - 5), 0.38, WHITE)

        # Row label
        put_text(img, label, (canvas_w - 170, row_y + panel_h // 2), 0.42, label_color, 1)
        put_text(img, f"r={max_dt:.0f}px", (canvas_w - 170, row_y + panel_h // 2 + 20), 0.35, WHITE)
        err_color = GREEN if dist_err < 10 else RED
        put_text(img, f"err={dist_err:.0f}px", (canvas_w - 170, row_y + panel_h // 2 + 38), 0.35, err_color)

        # Arrows
        for i in range(3):
            ax1 = 20 + (panel_w + 20) * i + panel_w + 3
            ax2 = 20 + (panel_w + 20) * (i + 1) - 3
            cv2.arrowedLine(img, (ax1, row_y + panel_h // 2), (ax2, row_y + panel_h // 2),
                            GRAY, 2, tipLength=0.15)

    # Row 1: Circular
    process_row(mask_c, rgb_c, 65, "CIRCULAR", GREEN)
    # Row 2: Elongated
    process_row(mask_e, rgb_e, 65 + panel_h + 50, "ELONGATED", ORANGE)

    put_text(img, "CIRCULAR: DT max = true center (error ~0px)  |  ELONGATED: DT max = fat end (error ~100px)",
             (20, canvas_h - 30), 0.4, WHITE)
    put_text(img, "RED = DT max    GREEN = moments (true center)    YELLOW arrow = error distance",
             (20, canvas_h - 8), 0.35, GRAY)

    cv2.imwrite(os.path.join(OUT, "14_dt_step_by_step.png"), img)
    print("Saved 14_dt_step_by_step.png")


if __name__ == "__main__":
    print("Generating DT explanation visuals...\n")
    vis_dt_grid()
    vis_dt_circle()
    vis_dt_bottle()
    vis_dt_real_objects()
    vis_dt_steps()
    print(f"\nAll saved to: {OUT}/")
