#!/usr/bin/env python3
"""
Generate educational visualization images for the centroid workflow explanation.
Each image illustrates one concept in the pipeline.
"""
import cv2
import numpy as np
import os
import json
import glob

OUT = "/home/kosal/cbnu_project/AI/picking-arm-robot/uoais/output_explanation"
os.makedirs(OUT, exist_ok=True)

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 150, 50)
YELLOW = (0, 230, 255)
GRAY = (180, 180, 180)
DARK_GRAY = (80, 80, 80)
ORANGE = (0, 140, 255)
CYAN = (255, 255, 0)
PINK = (180, 105, 255)
BG = (30, 30, 30)


def put_text(img, text, pos, scale=0.55, color=WHITE, thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def put_text_center(img, text, y, scale=0.6, color=WHITE, thickness=1):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = (img.shape[1] - tw) // 2
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


# ============================================================
# 1. CIRCULARITY COMPARISON
# ============================================================
def vis_circularity():
    """Show shapes with different circularity values."""
    W, H = 900, 350
    img = np.full((H, W, 3), BG, dtype=np.uint8)
    put_text_center(img, "CIRCULARITY = 4*pi*area / perimeter^2", 30, 0.7, YELLOW, 2)
    put_text_center(img, "Higher = more round,  1.0 = perfect circle", 55, 0.45, GRAY)

    shapes = []

    # Perfect circle
    mask1 = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask1, (100, 100), 80, 255, -1)
    shapes.append(("Perfect Circle", mask1))

    # Slightly irregular (like bottle cap)
    mask2 = np.zeros((200, 200), dtype=np.uint8)
    cv2.ellipse(mask2, (100, 100), (85, 70), 0, 0, 360, 255, -1)
    shapes.append(("Ellipse", mask2))

    # Square-ish
    mask3 = np.zeros((200, 200), dtype=np.uint8)
    pts = np.array([[30, 40], [170, 30], [175, 170], [25, 165]], dtype=np.int32)
    cv2.fillPoly(mask3, [pts], 255)
    shapes.append(("Square-ish", mask3))

    # Elongated
    mask4 = np.zeros((200, 200), dtype=np.uint8)
    cv2.ellipse(mask4, (100, 100), (90, 30), 15, 0, 360, 255, -1)
    shapes.append(("Elongated", mask4))

    # Star shape
    mask5 = np.zeros((200, 200), dtype=np.uint8)
    angles = np.linspace(0, 2 * np.pi, 11)[:-1]
    pts = []
    for i, a in enumerate(angles):
        r = 80 if i % 2 == 0 else 35
        pts.append([int(100 + r * np.cos(a)), int(100 + r * np.sin(a))])
    cv2.fillPoly(mask5, [np.array(pts, dtype=np.int32)], 255)
    shapes.append(("Star", mask5))

    x_offset = 10
    spacing = W // len(shapes)
    for i, (name, mask) in enumerate(shapes):
        cx = x_offset + i * spacing + spacing // 2
        # Compute circularity
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        circ = (4 * np.pi * area) / (peri * peri) if peri > 0 else 0

        # Draw shape
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_small = cv2.resize(mask_rgb, (130, 130))
        sx = cx - 65
        sy = 80
        # Color the shape
        for c in range(3):
            region = img[sy:sy + 130, sx:sx + 130, c]
            mask_region = mask_small[:, :, 0] > 0
            if circ > 0.7:
                color_val = [50, 200, 50][c]  # green = DT method
            else:
                color_val = [200, 100, 50][c]  # blue = moments method
            region[mask_region] = color_val
            region[~mask_region] = region[~mask_region]

        # Draw contour on img
        contours2, _ = cv2.findContours(
            cv2.resize(mask, (130, 130)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt2 in contours2:
            cnt2[:, :, 0] += sx
            cnt2[:, :, 1] += sy
            cv2.drawContours(img, [cnt2], -1, WHITE, 1, cv2.LINE_AA)

        # Labels
        put_text(img, name, (sx, sy + 155), 0.4, WHITE)
        circ_color = GREEN if circ > 0.7 else ORANGE
        put_text(img, f"circ = {circ:.2f}", (sx, sy + 175), 0.45, circ_color, 1)
        if circ > 0.7:
            put_text(img, "> 0.7: DT", (sx + 5, sy + 195), 0.4, GREEN)
        else:
            put_text(img, "< 0.7: Moments", (sx - 5, sy + 195), 0.4, ORANGE)

    # Draw the threshold line
    put_text(img, "Threshold = 0.7", (W - 200, H - 30), 0.45, YELLOW)
    put_text(img, "GREEN = use Distance Transform    ORANGE = use Moments", (20, H - 10), 0.4, GRAY)

    cv2.imwrite(os.path.join(OUT, "01_circularity.png"), img)
    print("Saved 01_circularity.png")


# ============================================================
# 2. ASPECT RATIO COMPARISON
# ============================================================
def vis_aspect_ratio():
    W, H = 900, 350
    img = np.full((H, W, 3), BG, dtype=np.uint8)
    put_text_center(img, "ASPECT RATIO = long side / short side", 30, 0.7, YELLOW, 2)
    put_text_center(img, "Higher = more elongated,  1.0 = square", 55, 0.45, GRAY)

    ratios = [
        (1.0, "1.0 (square)"),
        (1.5, "1.5"),
        (2.0, "2.0 (threshold)"),
        (3.0, "3.0"),
        (5.0, "5.0 (very long)")
    ]

    spacing = W // len(ratios)
    for i, (ar, label) in enumerate(ratios):
        cx = i * spacing + spacing // 2
        # Draw a rotated rectangle with this aspect ratio
        short_side = 40
        long_side = int(short_side * ar)
        # Fit within 150px height
        scale = min(1.0, 140 / long_side)
        long_side = int(long_side * scale)
        short_side = int(short_side * scale)
        if short_side < 10:
            short_side = 10

        angle = 20  # slight tilt
        box = ((cx, 160), (short_side, long_side), angle)
        pts = cv2.boxPoints(box).astype(np.int32)

        color = RED if ar >= 2.0 else GREEN
        cv2.fillPoly(img, [pts], (color[0] // 3, color[1] // 3, color[2] // 3))
        cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)

        # Draw min bounding rect dimensions
        put_text(img, label, (cx - 40, 280), 0.45, WHITE)
        put_text(img, f"AR = {ar:.1f}", (cx - 30, 300), 0.5, color, 1)
        if ar >= 2.0:
            put_text(img, "ELONGATED", (cx - 40, 320), 0.4, RED)
        else:
            put_text(img, "compact", (cx - 25, 320), 0.4, GREEN)

    put_text(img, "Threshold = 2.0  |  Below: compact  |  Above: elongated (moments center may be near thin edge)",
             (20, H - 10), 0.38, GRAY)

    cv2.imwrite(os.path.join(OUT, "02_aspect_ratio.png"), img)
    print("Saved 02_aspect_ratio.png")


# ============================================================
# 3. SOLIDITY (CONCAVITY)
# ============================================================
def vis_solidity():
    W, H = 750, 350
    img = np.full((H, W, 3), BG, dtype=np.uint8)
    put_text_center(img, "SOLIDITY = area / convex hull area", 30, 0.7, YELLOW, 2)
    put_text_center(img, "Lower = more concave/hollow.  1.0 = fully solid", 55, 0.45, GRAY)

    shapes = []

    # Solid rectangle
    m1 = np.zeros((180, 180), dtype=np.uint8)
    cv2.rectangle(m1, (30, 30), (150, 150), 255, -1)
    shapes.append(("Solid Rect", m1))

    # L-shape
    m2 = np.zeros((180, 180), dtype=np.uint8)
    cv2.rectangle(m2, (20, 20), (80, 160), 255, -1)
    cv2.rectangle(m2, (80, 100), (160, 160), 255, -1)
    shapes.append(("L-Shape", m2))

    # Crescent
    m3 = np.zeros((180, 180), dtype=np.uint8)
    cv2.circle(m3, (90, 90), 70, 255, -1)
    cv2.circle(m3, (120, 80), 55, 0, -1)
    shapes.append(("Crescent", m3))

    # Star
    m4 = np.zeros((180, 180), dtype=np.uint8)
    angles = np.linspace(0, 2 * np.pi, 11)[:-1]
    pts = []
    for j, a in enumerate(angles):
        r = 75 if j % 2 == 0 else 30
        pts.append([int(90 + r * np.cos(a)), int(90 + r * np.sin(a))])
    cv2.fillPoly(m4, [np.array(pts, dtype=np.int32)], 255)
    shapes.append(("Star", m4))

    spacing = W // len(shapes)
    for i, (name, mask) in enumerate(shapes):
        cx = i * spacing + spacing // 2
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        area = cv2.contourArea(cnt)
        hull_area = cv2.contourArea(hull)
        sol = area / hull_area if hull_area > 0 else 1.0

        # Draw hull (dashed feel - just thin line)
        mask_small = cv2.resize(mask, (140, 140))
        sx, sy = cx - 70, 75

        # Draw convex hull region in dark
        hull_mask = np.zeros((180, 180), dtype=np.uint8)
        cv2.fillPoly(hull_mask, [hull], 255)
        hull_small = cv2.resize(hull_mask, (140, 140))
        hull_only = (hull_small > 0) & (mask_small == 0)
        img[sy:sy + 140, sx:sx + 140][hull_only] = np.array([40, 40, 80])  # dark red for hull-only

        # Draw shape
        shape_region = mask_small > 0
        color = GREEN if sol > 0.85 else ORANGE
        img[sy:sy + 140, sx:sx + 140][shape_region] = np.array([color[0] // 2, color[1] // 2, color[2] // 2])

        # Contour
        cnts2, _ = cv2.findContours(mask_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c2 in cnts2:
            c2[:, :, 0] += sx
            c2[:, :, 1] += sy
            cv2.drawContours(img, [c2], -1, WHITE, 1, cv2.LINE_AA)

        # Hull outline
        hull_cnts, _ = cv2.findContours(hull_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for hc in hull_cnts:
            hc[:, :, 0] += sx
            hc[:, :, 1] += sy
            cv2.drawContours(img, [hc], -1, (100, 100, 200), 1, cv2.LINE_AA)

        put_text(img, name, (sx, sy + 160), 0.42, WHITE)
        put_text(img, f"sol = {sol:.2f}", (sx, sy + 180), 0.45, color, 1)
        if sol < 0.85:
            put_text(img, "CONCAVE", (sx + 10, sy + 200), 0.4, ORANGE)
        else:
            put_text(img, "solid", (sx + 20, sy + 200), 0.4, GREEN)

    put_text(img, "BLUE outline = convex hull  |  DARK area = gap between hull and shape", (20, H - 10), 0.38, GRAY)

    cv2.imwrite(os.path.join(OUT, "03_solidity.png"), img)
    print("Saved 03_solidity.png")


# ============================================================
# 4. DT vs MOMENTS on elongated shape
# ============================================================
def vis_dt_vs_moments():
    """Show WHY DT fails on elongated shapes."""
    W, H = 850, 400
    img = np.full((H, W, 3), BG, dtype=np.uint8)
    put_text_center(img, "WHY Distance Transform FAILS on Non-Circular Shapes", 30, 0.65, YELLOW, 2)

    # Create a bottle-like shape (elongated with one fat end)
    mask = np.zeros((250, 350), dtype=np.uint8)
    # Body
    cv2.ellipse(mask, (175, 125), (150, 50), 0, 0, 360, 255, -1)
    # Fat cap end
    cv2.circle(mask, (50, 125), 60, 255, -1)

    # --- Left: DT visualization ---
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dt_norm = (dt / dt.max() * 255).astype(np.uint8)
    dt_color = cv2.applyColorMap(dt_norm, cv2.COLORMAP_JET)
    dt_color[mask == 0] = np.array(BG)

    _, _, _, max_loc = cv2.minMaxLoc(dt)
    dt_cx, dt_cy = max_loc

    # Place left
    sx1, sy1 = 20, 90
    img[sy1:sy1 + 250, sx1:sx1 + 350] = dt_color
    # Mark DT center
    cv2.circle(img, (sx1 + dt_cx, sy1 + dt_cy), 8, RED, -1)
    cv2.circle(img, (sx1 + dt_cx, sy1 + dt_cy), 8, WHITE, 2)
    put_text(img, "DT center", (sx1 + dt_cx + 12, sy1 + dt_cy - 5), 0.45, RED)
    put_text(img, f"({dt_cx}, {dt_cy})", (sx1 + dt_cx + 12, sy1 + dt_cy + 12), 0.4, RED)

    put_text(img, "Distance Transform", (sx1 + 80, sy1 - 10), 0.55, WHITE, 1)
    put_text(img, "(HOT = far from edge)", (sx1 + 85, sy1 + 270), 0.4, GRAY)
    put_text(img, "WRONG! Biased to fat end", (sx1 + 50, sy1 + 290), 0.45, RED, 1)

    # --- Right: Moments visualization ---
    M = cv2.moments(mask)
    mx = M["m10"] / M["m00"]
    my = M["m01"] / M["m00"]

    sx2, sy2 = 470, 90
    # Draw mask in blue
    mask_color = np.full((250, 350, 3), BG, dtype=np.uint8)
    mask_color[mask > 0] = np.array([150, 100, 50])
    # Draw contour
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask_color, cnts, -1, WHITE, 1)

    img[sy2:sy2 + 250, sx2:sx2 + 350] = mask_color

    # Mark moments center
    mx_i, my_i = int(mx), int(my)
    cv2.circle(img, (sx2 + mx_i, sy2 + my_i), 8, GREEN, -1)
    cv2.circle(img, (sx2 + mx_i, sy2 + my_i), 8, WHITE, 2)
    put_text(img, "Moments center", (sx2 + mx_i + 12, sy2 + my_i - 5), 0.45, GREEN)
    put_text(img, f"({mx_i}, {my_i})", (sx2 + mx_i + 12, sy2 + my_i + 12), 0.4, GREEN)

    # Also show DT center for comparison
    cv2.circle(img, (sx2 + dt_cx, sy2 + dt_cy), 6, RED, -1)
    cv2.circle(img, (sx2 + dt_cx, sy2 + dt_cy), 6, (150, 150, 150), 1)
    put_text(img, "DT (wrong)", (sx2 + dt_cx - 10, sy2 + dt_cy - 10), 0.35, (150, 150, 150))

    # Draw distance arrow
    cv2.arrowedLine(img, (sx2 + dt_cx, sy2 + dt_cy + 3), (sx2 + mx_i, sy2 + my_i + 3), YELLOW, 2, tipLength=0.15)
    dist = np.sqrt((mx - dt_cx)**2 + (my - dt_cy)**2)
    mid_x = sx2 + (dt_cx + mx_i) // 2
    mid_y = sy2 + (dt_cy + my_i) // 2
    put_text(img, f"{dist:.0f}px error!", (mid_x - 20, mid_y - 10), 0.45, YELLOW, 1)

    put_text(img, "Moments (Center of Mass)", (sx2 + 60, sy2 - 10), 0.55, WHITE, 1)
    put_text(img, "(average position of all pixels)", (sx2 + 55, sy2 + 270), 0.4, GRAY)
    put_text(img, "CORRECT! True geometric center", (sx2 + 35, sy2 + 290), 0.45, GREEN, 1)

    cv2.imwrite(os.path.join(OUT, "04_dt_vs_moments.png"), img)
    print("Saved 04_dt_vs_moments.png")


# ============================================================
# 5. SAFETY CHECK visualization
# ============================================================
def vis_safety_check():
    W, H = 850, 400
    img = np.full((H, W, 3), BG, dtype=np.uint8)
    put_text_center(img, "SAFETY CHECK: Is the center point safe for suction?", 30, 0.6, YELLOW, 2)

    # --- Case 1: Safe (far from edge) ---
    mask1 = np.zeros((200, 200), dtype=np.uint8)
    pts1 = np.array([[30, 30], [170, 40], [165, 170], [35, 165]], dtype=np.int32)
    cv2.fillPoly(mask1, [pts1], 255)
    dt1 = cv2.distanceTransform(mask1, cv2.DIST_L2, 5)
    M1 = cv2.moments(mask1)
    mx1, my1 = int(M1["m10"] / M1["m00"]), int(M1["m01"] / M1["m00"])
    edge_d1 = dt1[my1, mx1]
    max_dt1 = dt1.max()

    sx1, sy1 = 30, 80
    mask1_c = np.full((200, 200, 3), BG, dtype=np.uint8)
    mask1_c[mask1 > 0] = [60, 120, 60]
    cnts1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask1_c, cnts1, -1, WHITE, 1)
    # Draw clearance circle
    cv2.circle(mask1_c, (mx1, my1), int(edge_d1), (0, 180, 0), 1, cv2.LINE_AA)
    cv2.circle(mask1_c, (mx1, my1), 6, GREEN, -1)
    cv2.circle(mask1_c, (mx1, my1), 6, WHITE, 2)
    img[sy1:sy1 + 200, sx1:sx1 + 200] = mask1_c

    put_text(img, "CASE 1: SAFE", (sx1 + 40, sy1 - 10), 0.55, GREEN, 1)
    put_text(img, f"Edge dist = {edge_d1:.0f}px", (sx1, sy1 + 220), 0.4, WHITE)
    put_text(img, f"Min clearance = {max_dt1:.0f} x 20% = {max_dt1*0.2:.0f}px", (sx1, sy1 + 240), 0.4, WHITE)
    put_text(img, f"{edge_d1:.0f} >= {max_dt1*0.2:.0f} -> SAFE!", (sx1, sy1 + 260), 0.45, GREEN, 1)
    put_text(img, "Green circle = clearance", (sx1, sy1 + 280), 0.35, GRAY)

    # --- Case 2: Too close to edge ---
    mask2 = np.zeros((200, 300), dtype=np.uint8)
    cv2.ellipse(mask2, (150, 100), (140, 40), 0, 0, 360, 255, -1)
    cv2.circle(mask2, (20, 100), 50, 255, -1)
    dt2 = cv2.distanceTransform(mask2, cv2.DIST_L2, 5)
    M2 = cv2.moments(mask2)
    mx2, my2 = int(M2["m10"] / M2["m00"]), int(M2["m01"] / M2["m00"])
    edge_d2 = dt2[my2, mx2]
    max_dt2 = dt2.max()
    _, _, _, dt2_max_loc = cv2.minMaxLoc(dt2)

    sx2, sy2 = 290, 80
    mask2_c = np.full((200, 300, 3), BG, dtype=np.uint8)
    mask2_c[mask2 > 0] = [50, 50, 120]
    cnts2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask2_c, cnts2, -1, WHITE, 1)
    # Moments center - with tiny clearance circle
    cv2.circle(mask2_c, (mx2, my2), max(int(edge_d2), 3), (0, 0, 200), 1, cv2.LINE_AA)
    cv2.circle(mask2_c, (mx2, my2), 6, RED, -1)
    cv2.circle(mask2_c, (mx2, my2), 6, WHITE, 2)
    put_text(mask2_c, "moments", (mx2 + 8, my2 - 8), 0.35, RED)
    # DT max
    dtx2, dty2 = dt2_max_loc
    cv2.circle(mask2_c, (dtx2, dty2), 6, BLUE, -1)
    put_text(mask2_c, "DT max", (dtx2 + 8, dty2 - 5), 0.35, BLUE)
    # Blended result
    blend_x = int(0.3 * mx2 + 0.7 * dtx2)
    blend_y = int(0.3 * my2 + 0.7 * dty2)
    cv2.circle(mask2_c, (blend_x, blend_y), 7, GREEN, -1)
    cv2.circle(mask2_c, (blend_x, blend_y), 7, WHITE, 2)
    put_text(mask2_c, "BLENDED", (blend_x + 10, blend_y + 5), 0.35, GREEN)
    # Arrow from moments to blend
    cv2.arrowedLine(mask2_c, (mx2, my2), (blend_x, blend_y), YELLOW, 1, tipLength=0.2)
    img[sy2:sy2 + 200, sx2:sx2 + 300] = mask2_c

    put_text(img, "CASE 2: TOO CLOSE TO EDGE", (sx2 + 40, sy2 - 10), 0.55, RED, 1)
    put_text(img, f"Edge dist = {edge_d2:.0f}px", (sx2, sy2 + 220), 0.4, WHITE)
    put_text(img, f"Min clearance = {max_dt2:.0f} x 20% = {max_dt2*0.2:.0f}px", (sx2, sy2 + 240), 0.4, WHITE)
    put_text(img, f"{edge_d2:.0f} < {max_dt2*0.2:.0f} -> BLEND!", (sx2, sy2 + 260), 0.45, ORANGE, 1)
    put_text(img, "30% moments + 70% DT max", (sx2, sy2 + 280), 0.4, YELLOW)

    # --- Case 3: Outside mask (concave) ---
    mask3 = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(mask3, (10, 10), (80, 190), 255, -1)
    cv2.rectangle(mask3, (80, 130), (190, 190), 255, -1)
    M3 = cv2.moments(mask3)
    mx3, my3 = int(M3["m10"] / M3["m00"]), int(M3["m01"] / M3["m00"])

    sx3, sy3 = 640, 80
    mask3_c = np.full((200, 200, 3), BG, dtype=np.uint8)
    mask3_c[mask3 > 0] = [100, 50, 50]
    cnts3, _ = cv2.findContours(mask3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask3_c, cnts3, -1, WHITE, 1)
    # Moments center - outside!
    inside = mask3[my3, mx3] > 0
    cv2.circle(mask3_c, (mx3, my3), 6, RED, -1)
    cv2.drawMarker(mask3_c, (mx3, my3), RED, cv2.MARKER_TILTED_CROSS, 15, 2)
    put_text(mask3_c, "OUTSIDE!", (mx3 + 8, my3 - 5), 0.4, RED, 1)

    # DT max
    dt3 = cv2.distanceTransform(mask3, cv2.DIST_L2, 5)
    _, _, _, dt3_loc = cv2.minMaxLoc(dt3)
    dtx3, dty3 = dt3_loc
    cv2.circle(mask3_c, (dtx3, dty3), 5, BLUE, -1)

    # Nearest mask point
    coords3 = np.where(mask3 > 0)
    dists = (coords3[1] - mx3)**2 + (coords3[0] - my3)**2
    ni = np.argmin(dists)
    nx3, ny3 = int(coords3[1][ni]), int(coords3[0][ni])

    bx3 = (nx3 + dtx3) // 2
    by3 = (ny3 + dty3) // 2
    cv2.circle(mask3_c, (bx3, by3), 7, GREEN, -1)
    cv2.circle(mask3_c, (bx3, by3), 7, WHITE, 2)

    img[sy3:sy3 + 200, sx3:sx3 + 200] = mask3_c

    put_text(img, "CASE 3: OUTSIDE", (sx3 + 30, sy3 - 10), 0.55, RED, 1)
    put_text(img, f"Moments at ({mx3},{my3})", (sx3, sy3 + 220), 0.4, WHITE)
    put_text(img, f"mask[{my3},{mx3}] = {'inside' if inside else 'OUTSIDE'}", (sx3, sy3 + 240), 0.4, RED)
    put_text(img, "50% nearest + 50% DT max", (sx3, sy3 + 260), 0.4, YELLOW)

    cv2.imwrite(os.path.join(OUT, "05_safety_check.png"), img)
    print("Saved 05_safety_check.png")


# ============================================================
# 6. REAL T-Less example — full pipeline on one object
# ============================================================
def vis_tless_pipeline():
    """Show the full pipeline on a real T-Less object."""
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

    # Use object #1 (elongated)
    mask = masks[1]
    mask_uint8 = (mask > 0).astype(np.uint8)

    H_img, W_img = rgb.shape[:2]

    # Create a 5-panel image
    panel_w, panel_h = 250, 200
    W = panel_w * 3 + 80
    H = panel_h * 2 + 140
    img = np.full((H, W, 3), BG, dtype=np.uint8)
    put_text_center(img, "FULL PIPELINE: T-Less Frame 10, Object #1 (elongated, AR=2.27)", 25, 0.55, YELLOW, 2)

    # Crop around object
    coords = np.where(mask_uint8 > 0)
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    pad = 15
    y_min = max(0, y_min - pad)
    y_max = min(H_img, y_max + pad)
    x_min = max(0, x_min - pad)
    x_max = min(W_img, x_max + pad)

    crop_rgb = rgb[y_min:y_max, x_min:x_max].copy()
    crop_mask = mask_uint8[y_min:y_max, x_min:x_max]
    crop_depth = depth_mm[y_min:y_max, x_min:x_max]

    def resize_panel(src, pw=panel_w, ph=panel_h):
        return cv2.resize(src, (pw, ph), interpolation=cv2.INTER_LINEAR)

    # Panel 1: Original + mask overlay
    p1 = crop_rgb.copy()
    p1[crop_mask > 0] = (p1[crop_mask > 0] * 0.5 + np.array([100, 150, 255]) * 0.5).astype(np.uint8)
    cnts, _ = cv2.findContours(crop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(p1, cnts, -1, WHITE, 1)
    p1 = resize_panel(p1)
    sx, sy = 20, 50
    img[sy:sy + panel_h, sx:sx + panel_w] = p1
    put_text(img, "1. Segmented Mask", (sx, sy + panel_h + 15), 0.45, WHITE, 1)
    put_text(img, f"   area=33805px AR=2.27", (sx, sy + panel_h + 32), 0.38, GRAY)

    # Panel 2: Shape analysis (circularity visualization)
    cnt = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    p2 = np.full((crop_mask.shape[0], crop_mask.shape[1], 3), BG, dtype=np.uint8)
    p2[crop_mask > 0] = [80, 80, 80]
    cv2.drawContours(p2, cnts, -1, WHITE, 1)
    # Draw min area rect
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.polylines(p2, [box], True, YELLOW, 2)
    # Draw convex hull
    hull = cv2.convexHull(cnt)
    cv2.polylines(p2, [hull], True, CYAN, 1)
    p2 = resize_panel(p2)
    sx2 = 20 + panel_w + 20
    img[sy:sy + panel_h, sx2:sx2 + panel_w] = p2
    put_text(img, "2. Shape Analysis", (sx2, sy + panel_h + 15), 0.45, WHITE, 1)
    put_text(img, "   circ=0.455 -> NOT circular", (sx2, sy + panel_h + 32), 0.38, ORANGE)

    # Panel 3: Distance Transform heatmap
    dt = cv2.distanceTransform(crop_mask, cv2.DIST_L2, 5)
    dt_norm = (dt / max(dt.max(), 1) * 255).astype(np.uint8)
    p3 = cv2.applyColorMap(dt_norm, cv2.COLORMAP_JET)
    p3[crop_mask == 0] = np.array(BG)
    # Mark DT max
    _, _, _, dt_loc = cv2.minMaxLoc(dt)
    cv2.circle(p3, dt_loc, 5, WHITE, 2)
    put_text(p3, "DT max", (dt_loc[0] + 6, dt_loc[1] - 4), 0.3, WHITE)
    p3 = resize_panel(p3)
    sx3 = 20 + (panel_w + 20) * 2
    img[sy:sy + panel_h, sx3:sx3 + panel_w] = p3
    put_text(img, "3. Distance Transform", (sx3, sy + panel_h + 15), 0.45, WHITE, 1)
    put_text(img, "   DT biased to wide end!", (sx3, sy + panel_h + 32), 0.38, RED)

    # Panel 4: Moments center + safety check
    M = cv2.moments(crop_mask)
    mx_c = M["m10"] / M["m00"]
    my_c = M["m01"] / M["m00"]
    mx_i, my_i = int(mx_c), int(my_c)

    p4 = crop_rgb.copy()
    p4[crop_mask > 0] = (p4[crop_mask > 0] * 0.6 + np.array([80, 80, 80]) * 0.4).astype(np.uint8)
    cv2.drawContours(p4, cnts, -1, WHITE, 1)
    # Moments center
    edge_d = dt[my_i, mx_i]
    cv2.circle(p4, (mx_i, my_i), max(int(edge_d), 2), (0, 0, 200), 1)
    cv2.circle(p4, (mx_i, my_i), 5, RED, -1)
    put_text(p4, f"moments", (mx_i + 6, my_i - 6), 0.3, RED)
    put_text(p4, f"edge={edge_d:.0f}px", (mx_i + 6, my_i + 10), 0.3, RED)
    p4 = resize_panel(p4)
    sy2 = sy + panel_h + 50
    sx4 = 20
    img[sy2:sy2 + panel_h, sx4:sx4 + panel_w] = p4
    put_text(img, "4. Moments + Safety Check", (sx4, sy2 + panel_h + 15), 0.45, WHITE, 1)
    put_text(img, f"   edge=1px < 13px -> TOO CLOSE!", (sx4, sy2 + panel_h + 32), 0.38, RED)

    # Panel 5: Final blended result
    dtx, dty = dt_loc
    blend_x = int(0.3 * mx_c + 0.7 * dtx)
    blend_y = int(0.3 * my_c + 0.7 * dty)

    p5 = crop_rgb.copy()
    p5[crop_mask > 0] = (p5[crop_mask > 0] * 0.5 + np.array([100, 180, 100]) * 0.5).astype(np.uint8)
    cv2.drawContours(p5, cnts, -1, WHITE, 1)
    # Show all three points
    cv2.circle(p5, (mx_i, my_i), 4, RED, -1)
    put_text(p5, "moments", (mx_i - 35, my_i - 8), 0.25, RED)
    cv2.circle(p5, (dtx, dty), 4, BLUE, -1)
    put_text(p5, "DT max", (dtx + 5, dty - 5), 0.25, BLUE)
    cv2.circle(p5, (blend_x, blend_y), 7, GREEN, -1)
    cv2.circle(p5, (blend_x, blend_y), 7, WHITE, 2)
    put_text(p5, "RESULT", (blend_x + 8, blend_y + 4), 0.35, GREEN, 1)
    cv2.arrowedLine(p5, (mx_i, my_i), (blend_x, blend_y), YELLOW, 1, tipLength=0.15)
    p5 = resize_panel(p5)
    sx5 = 20 + panel_w + 20
    img[sy2:sy2 + panel_h, sx5:sx5 + panel_w] = p5
    put_text(img, "5. FINAL: Blend 30% moments + 70% DT", (sx5, sy2 + panel_h + 15), 0.45, GREEN, 1)
    put_text(img, f"   RED DOT at ({blend_x + x_min}, {blend_y + y_min})", (sx5, sy2 + panel_h + 32), 0.38, GREEN)

    # Panel 6: Final full image with both points
    p6 = rgb.copy()
    for mi, m in enumerate(masks):
        mb = m > 0
        if np.sum(mb) < 100:
            continue
        color = [(100, 150, 255), (100, 255, 150), (255, 150, 100), (200, 100, 255), (100, 255, 255)][mi % 5]
        p6[mb] = (p6[mb] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
    # Highlight object #1
    cv2.drawContours(p6, [cnt + np.array([x_min, y_min])], -1, YELLOW, 2)
    # Draw the final center
    cv2.circle(p6, (blend_x + x_min, blend_y + y_min), 8, (0, 0, 255), -1)
    cv2.circle(p6, (blend_x + x_min, blend_y + y_min), 8, WHITE, 2)

    p6_h = panel_h
    p6_w = int(p6.shape[1] * p6_h / p6.shape[0])
    p6 = cv2.resize(p6, (p6_w, p6_h))
    sx6 = 20 + (panel_w + 20) * 2
    if sx6 + p6_w > W:
        p6 = p6[:, :W - sx6]
        p6_w = W - sx6
    img[sy2:sy2 + panel_h, sx6:sx6 + p6_w] = p6
    put_text(img, "6. Full Frame Result", (sx6, sy2 + panel_h + 15), 0.45, WHITE, 1)
    put_text(img, "   Yellow = Object #1", (sx6, sy2 + panel_h + 32), 0.38, YELLOW)

    cv2.imwrite(os.path.join(OUT, "06_tless_pipeline.png"), img)
    print("Saved 06_tless_pipeline.png")


# ============================================================
# 7. SUCTION GRASP POINT (with depth)
# ============================================================
def vis_suction_grasp():
    """Show how suction grasp point differs from geometric center."""
    scene_dir = "/home/kosal/cbnu_project/AI/picking-arm-robot/dataset/t_less-data/000003"
    frame_id = 10
    fid = f"{frame_id:06d}"

    rgb = cv2.imread(os.path.join(scene_dir, "rgb", f"{fid}.png"))
    depth_raw = cv2.imread(os.path.join(scene_dir, "depth", f"{fid}.png"), cv2.IMREAD_UNCHANGED)
    with open(os.path.join(scene_dir, "scene_camera.json")) as f:
        cam = json.load(f)[str(frame_id)]
    depth_mm = depth_raw.astype(np.float32) * cam["depth_scale"]

    mask_files = sorted(glob.glob(os.path.join(scene_dir, "mask", f"{fid}_*.png")))
    mask = cv2.imread(mask_files[1], cv2.IMREAD_GRAYSCALE)
    mask_uint8 = (mask > 0).astype(np.uint8)

    H_img, W_img = rgb.shape[:2]

    coords = np.where(mask_uint8 > 0)
    y_min, y_max = coords[0].min() - 15, coords[0].max() + 15
    x_min, x_max = coords[1].min() - 15, coords[1].max() + 15
    y_min, x_min = max(0, y_min), max(0, x_min)
    y_max, x_max = min(H_img, y_max), min(W_img, x_max)

    crop_mask = mask_uint8[y_min:y_max, x_min:x_max]
    crop_depth = depth_mm[y_min:y_max, x_min:x_max]
    crop_rgb = rgb[y_min:y_max, x_min:x_max]

    panel_w, panel_h = 280, 200
    W = panel_w * 3 + 60
    H = panel_h + 140
    img = np.full((H, W, 3), BG, dtype=np.uint8)
    put_text_center(img, "SUCTION GRASP POINT: How depth helps find the BEST grip location", 25, 0.55, YELLOW, 2)

    def resize_p(src):
        return cv2.resize(src, (panel_w, panel_h), interpolation=cv2.INTER_LINEAR)

    # Panel 1: Depth on mask
    dep_on_mask = crop_depth.copy()
    dep_on_mask[crop_mask == 0] = 0
    valid = dep_on_mask[dep_on_mask > 0]
    if len(valid) > 0:
        d_min, d_max = valid.min(), valid.max()
        dep_norm = np.zeros_like(dep_on_mask, dtype=np.uint8)
        valid_mask = dep_on_mask > 0
        dep_norm[valid_mask] = ((dep_on_mask[valid_mask] - d_min) / max(d_max - d_min, 1) * 255).astype(np.uint8)
    else:
        dep_norm = np.zeros_like(crop_mask)
        d_min, d_max = 0, 0
    p1 = cv2.applyColorMap(dep_norm, cv2.COLORMAP_VIRIDIS)
    p1[crop_mask == 0] = np.array(BG)
    cnts, _ = cv2.findContours(crop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(p1, cnts, -1, WHITE, 1)
    p1 = resize_p(p1)
    sx1, sy1 = 20, 50
    img[sy1:sy1 + panel_h, sx1:sx1 + panel_w] = p1
    put_text(img, "1. Depth Map on Object", (sx1, sy1 + panel_h + 15), 0.45, WHITE, 1)
    put_text(img, f"   {d_min:.0f}mm (close) to {d_max:.0f}mm (far)", (sx1, sy1 + panel_h + 32), 0.38, GRAY)
    put_text(img, f"   BRIGHT = close to camera (top)", (sx1, sy1 + panel_h + 48), 0.35, CYAN)

    # Panel 2: Top surface + flatness
    depth_range = d_max - d_min
    top_thresh = d_min + depth_range * 0.20
    top_surface = (dep_on_mask > 0) & (dep_on_mask <= top_thresh)

    # Compute simple flatness from depth gradients
    dx = cv2.Sobel(crop_depth, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(crop_depth, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(dx**2 + dy**2)
    # Flatness = inverse of gradient magnitude
    flatness = np.zeros_like(gradient_mag)
    flatness[top_surface] = 1.0 - np.clip(gradient_mag[top_surface] / max(gradient_mag[top_surface].max(), 1), 0, 1)

    flat_vis = np.zeros_like(dep_norm)
    if np.any(top_surface):
        flat_vis[top_surface] = (flatness[top_surface] * 255).astype(np.uint8)
    p2 = cv2.applyColorMap(flat_vis, cv2.COLORMAP_HOT)
    p2[crop_mask == 0] = np.array(BG)
    p2[~top_surface & (crop_mask > 0)] = np.array([40, 40, 40])
    cv2.drawContours(p2, cnts, -1, WHITE, 1)
    p2 = resize_p(p2)
    sx2 = 20 + panel_w + 20
    img[sy1:sy1 + panel_h, sx2:sx2 + panel_w] = p2
    put_text(img, "2. Top Surface Flatness", (sx2, sy1 + panel_h + 15), 0.45, WHITE, 1)
    put_text(img, f"   Top 20% depth (< {top_thresh:.0f}mm)", (sx2, sy1 + panel_h + 32), 0.38, GRAY)
    put_text(img, f"   HOT = flat (good suction seal)", (sx2, sy1 + panel_h + 48), 0.35, ORANGE)

    # Panel 3: Final with both RED and GREEN
    p3 = crop_rgb.copy()
    p3[crop_mask > 0] = (p3[crop_mask > 0] * 0.5 + np.array([100, 180, 100]) * 0.5).astype(np.uint8)
    cv2.drawContours(p3, cnts, -1, WHITE, 1)

    # Geometric center (adaptive)
    M = cv2.moments(crop_mask)
    mx_c = M["m10"] / M["m00"]
    my_c = M["m01"] / M["m00"]
    dt = cv2.distanceTransform(crop_mask, cv2.DIST_L2, 5)
    _, _, _, dt_loc = cv2.minMaxLoc(dt)
    # Blended (same as walkthrough)
    edge_d = dt[int(my_c), int(mx_c)]
    max_dt = dt.max()
    if edge_d < max_dt * 0.20:
        cx_r = int(0.3 * mx_c + 0.7 * dt_loc[0])
        cy_r = int(0.3 * my_c + 0.7 * dt_loc[1])
    else:
        cx_r, cy_r = int(mx_c), int(my_c)

    # Suction point: scored by flatness + edge + center
    dt_top = cv2.distanceTransform(top_surface.astype(np.uint8), cv2.DIST_L2, 5)
    dt_top_norm = dt_top / max(dt_top.max(), 1)
    center_x, center_y = crop_mask.shape[1] // 2, crop_mask.shape[0] // 2
    yy, xx = np.mgrid[:crop_mask.shape[0], :crop_mask.shape[1]]
    center_dist = np.sqrt((xx - center_x)**2 + (yy - center_y)**2)
    center_prox = 1.0 - np.clip(center_dist / max(center_dist.max(), 1), 0, 1)

    score = np.zeros_like(flatness)
    score[top_surface] = (0.4 * flatness[top_surface] +
                          0.3 * dt_top_norm[top_surface] +
                          0.3 * center_prox[top_surface])
    if np.any(score > 0):
        best_idx = np.unravel_index(np.argmax(score), score.shape)
        gx, gy = best_idx[1], best_idx[0]
    else:
        gx, gy = cx_r, cy_r

    # Draw RED dot (center)
    cv2.circle(p3, (cx_r, cy_r), 7, (0, 0, 255), -1)
    cv2.circle(p3, (cx_r, cy_r), 7, WHITE, 2)
    put_text(p3, "CENTER", (cx_r - 30, cy_r - 10), 0.35, RED, 1)

    # Draw GREEN cross (grasp)
    cv2.drawMarker(p3, (gx, gy), GREEN, cv2.MARKER_CROSS, 20, 2)
    put_text(p3, "GRASP", (gx + 12, gy + 5), 0.35, GREEN, 1)

    # White line
    dist = np.sqrt((cx_r - gx)**2 + (cy_r - gy)**2)
    if dist > 3:
        cv2.line(p3, (cx_r, cy_r), (gx, gy), WHITE, 1)

    p3 = resize_p(p3)
    sx3 = 20 + (panel_w + 20) * 2
    img[sy1:sy1 + panel_h, sx3:sx3 + panel_w] = p3
    put_text(img, "3. RESULT: Two Points", (sx3, sy1 + panel_h + 15), 0.45, WHITE, 1)
    put_text(img, "   RED = geometric center", (sx3, sy1 + panel_h + 32), 0.38, RED)
    put_text(img, "   GREEN = best suction grip", (sx3, sy1 + panel_h + 48), 0.38, GREEN)

    cv2.imwrite(os.path.join(OUT, "07_suction_grasp.png"), img)
    print("Saved 07_suction_grasp.png")


# ============================================================
# 8. METHOD DECISION FLOWCHART (visual)
# ============================================================
def vis_flowchart():
    """Visual flowchart of the decision process."""
    W, H = 800, 550
    img = np.full((H, W, 3), BG, dtype=np.uint8)
    put_text_center(img, "DECISION FLOWCHART: How we choose the method", 25, 0.6, YELLOW, 2)

    # Box drawing helper
    def draw_box(x, y, w, h, text, color, text_color=WHITE):
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(img, (x + 1, y + 1), (x + w - 1, y + h - 1), (color[0]//4, color[1]//4, color[2]//4), -1)
        lines = text.split('\n')
        for i, line in enumerate(lines):
            put_text(img, line, (x + 8, y + 18 + i * 18), 0.4, text_color)

    def draw_diamond(cx, cy, text, color):
        size = 50
        pts = np.array([[cx, cy - size], [cx + size + 30, cy],
                        [cx, cy + size], [cx - size - 30, cy]], dtype=np.int32)
        cv2.fillPoly(img, [pts], (color[0]//4, color[1]//4, color[2]//4))
        cv2.polylines(img, [pts], True, color, 2)
        lines = text.split('\n')
        for i, line in enumerate(lines):
            (tw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            put_text(img, line, (cx - tw // 2, cy - 5 + i * 16), 0.38, WHITE)

    def arrow(x1, y1, x2, y2, label="", label_side="right"):
        cv2.arrowedLine(img, (x1, y1), (x2, y2), GRAY, 2, tipLength=0.1)
        if label:
            if label_side == "right":
                put_text(img, label, (max(x1, x2) + 5, (y1 + y2) // 2 + 4), 0.38, YELLOW)
            elif label_side == "left":
                put_text(img, label, (min(x1, x2) - 40, (y1 + y2) // 2 + 4), 0.38, YELLOW)
            elif label_side == "top":
                put_text(img, label, ((x1 + x2) // 2 - 10, min(y1, y2) - 5), 0.38, YELLOW)

    # Start
    draw_box(310, 50, 180, 30, "Binary Mask (input)", WHITE)

    arrow(400, 80, 400, 110)

    # Step 1: Analyze shape
    draw_box(300, 110, 200, 35, "analyze_mask_shape()\ncirc, AR, solidity", BLUE)

    arrow(400, 145, 400, 185)

    # Diamond: circular?
    draw_diamond(400, 230, "circular?\ncirc > 0.7", CYAN)

    # YES branch
    arrow(480, 230, 600, 230, "YES", "top")
    draw_box(600, 210, 170, 45, "Distance Transform\n(inscribed circle\n center)", GREEN)

    # NO branch
    arrow(400, 280, 400, 320, "NO", "right")

    # Moments
    draw_box(300, 320, 200, 30, "Moments (center of mass)", ORANGE)

    arrow(400, 350, 400, 380)

    # Diamond: inside & safe?
    draw_diamond(400, 420, "inside mask?\nedge > 20%?", CYAN)

    # YES - safe
    arrow(480, 420, 620, 420, "YES", "top")
    draw_box(620, 400, 150, 40, "Use moments\ndirectly", GREEN)

    # NO - too close
    arrow(320, 420, 160, 420, "NO", "top")
    draw_box(30, 400, 180, 40, "Blend moments\n+ DT max", ORANGE)

    # All lead to result
    arrow(685, 255, 685, 470)
    arrow(695, 440, 685, 470)
    arrow(120, 440, 120, 470)
    cv2.line(img, (120, 470), (685, 470), GRAY, 2)
    arrow(400, 470, 400, 495)

    draw_box(310, 495, 180, 35, "RED DOT position", RED)

    cv2.imwrite(os.path.join(OUT, "08_flowchart.png"), img)
    print("Saved 08_flowchart.png")


# ============================================================
# 9. No Depth vs Real Depth comparison
# ============================================================
def vis_depth_comparison():
    W, H = 700, 300
    img = np.full((H, W, 3), BG, dtype=np.uint8)
    put_text_center(img, "WITH vs WITHOUT Depth Camera", 25, 0.6, YELLOW, 2)

    # Left: no depth
    draw_w, draw_h = 200, 150
    sx1, sy1 = 60, 70

    # Fake object shape
    mask = np.zeros((draw_h, draw_w), dtype=np.uint8)
    cv2.ellipse(mask, (100, 75), (80, 50), 10, 0, 360, 255, -1)

    p1 = np.full((draw_h, draw_w, 3), BG, dtype=np.uint8)
    p1[mask > 0] = [80, 80, 120]
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(p1, cnts, -1, WHITE, 1)

    M = cv2.moments(mask)
    mx, my = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    cv2.circle(p1, (mx, my), 7, (0, 0, 255), -1)
    cv2.circle(p1, (mx, my), 7, WHITE, 2)
    cv2.drawMarker(p1, (mx, my), GREEN, cv2.MARKER_CROSS, 20, 2)
    put_text(p1, "SAME point", (mx + 10, my - 5), 0.35, YELLOW)

    img[sy1:sy1 + draw_h, sx1:sx1 + draw_w] = p1
    put_text(img, "WITHOUT Depth (phone photo)", (sx1, sy1 - 10), 0.48, WHITE, 1)
    put_text(img, "depth_range = 0mm -> fallback", (sx1, sy1 + draw_h + 18), 0.38, GRAY)
    put_text(img, "RED and GREEN = same position", (sx1, sy1 + draw_h + 36), 0.38, ORANGE)

    # Right: with depth
    sx2, sy2 = 400, 70
    p2 = np.full((draw_h, draw_w, 3), BG, dtype=np.uint8)
    p2[mask > 0] = [80, 120, 80]
    cv2.drawContours(p2, cnts, -1, WHITE, 1)

    # RED center
    cv2.circle(p2, (mx, my), 7, (0, 0, 255), -1)
    cv2.circle(p2, (mx, my), 7, WHITE, 2)
    put_text(p2, "CENTER", (mx - 30, my - 12), 0.3, RED)

    # GREEN shifted to "flattest" spot
    gx, gy = mx + 25, my - 15
    cv2.drawMarker(p2, (gx, gy), GREEN, cv2.MARKER_CROSS, 20, 2)
    put_text(p2, "GRASP", (gx + 12, gy - 3), 0.3, GREEN)
    cv2.line(p2, (mx, my), (gx, gy), WHITE, 1)

    img[sy2:sy2 + draw_h, sx2:sx2 + draw_w] = p2
    put_text(img, "WITH Depth (L515 camera)", (sx2, sy2 - 10), 0.48, WHITE, 1)
    put_text(img, "depth_range = 135mm -> full analysis", (sx2, sy2 + draw_h + 18), 0.38, GRAY)
    put_text(img, "RED = center, GREEN = flat+safe spot", (sx2, sy2 + draw_h + 36), 0.38, GREEN)

    cv2.imwrite(os.path.join(OUT, "09_depth_comparison.png"), img)
    print("Saved 09_depth_comparison.png")


# ============================================================
# Run all
# ============================================================
if __name__ == "__main__":
    print("Generating explanation visuals...\n")
    vis_circularity()
    vis_aspect_ratio()
    vis_solidity()
    vis_dt_vs_moments()
    vis_safety_check()
    vis_tless_pipeline()
    vis_suction_grasp()
    vis_flowchart()
    vis_depth_comparison()
    print(f"\nAll images saved to: {OUT}/")
