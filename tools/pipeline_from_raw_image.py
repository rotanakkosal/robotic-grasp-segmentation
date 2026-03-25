#!/usr/bin/env python3
"""
Generate the 6-panel pipeline visualization from a RAW image (no pre-computed masks).

Steps:
  1. Segment objects using simple color thresholding + connected components
  2. For each detected object, generate the 6-panel pipeline:
     Mask → Shape Analysis → DT heatmap → Method+Safety → Final → Full Frame

Usage:
    python tools/pipeline_from_raw_image.py
"""
import cv2
import numpy as np
import os

# ── Configuration ──
IMG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "sample_data",
    "Gemini_Generated_Image_f72zk1f72zk1f72z.png",
)
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output_raw_pipeline")

# ── Colors (BGR) ──
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 150, 50)
YELLOW = (0, 230, 255)
GRAY = (180, 180, 180)
ORANGE = (0, 140, 255)
CYAN = (255, 255, 0)
BG = (30, 30, 30)

OBJ_COLORS = [
    (100, 150, 255),
    (100, 255, 150),
    (255, 150, 100),
    (200, 100, 255),
    (100, 255, 255),
    (255, 200, 100),
    (150, 255, 200),
    (255, 100, 200),
]


def put_text(img, text, pos, scale=0.55, color=WHITE, thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def put_text_center(img, text, y, scale=0.6, color=WHITE, thickness=1):
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = (img.shape[1] - tw) // 2
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def segment_objects(rgb):
    """
    Simple object segmentation using Otsu thresholding + morphology + connected components.
    Returns a list of binary masks, one per detected object.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Adaptive threshold to handle varying lighting
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 51, 10,
    )

    # Also try Otsu on the edges for better separation
    edges = cv2.Canny(blurred, 30, 100)
    edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    # Combine threshold and edge info
    combined = cv2.bitwise_or(thresh, edges_dilated)

    # Morphological close to fill gaps within objects
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Remove small noise
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_small, iterations=2)

    # Connected components
    num_labels, labels = cv2.connectedComponents(cleaned)

    H, W = rgb.shape[:2]
    min_area = int(H * W * 0.005)  # at least 0.5% of image
    max_area = int(H * W * 0.5)    # at most 50% of image

    masks = []
    for label_id in range(1, num_labels):  # skip background (0)
        mask = (labels == label_id).astype(np.uint8)
        area = np.sum(mask)
        if min_area < area < max_area:
            # Fill holes in this mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filled = np.zeros_like(mask)
            cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
            masks.append(filled)

    print(f"  Detected {len(masks)} objects (min_area={min_area}, max_area={max_area})")
    return masks


def generate_object_pipeline(rgb, masks, obj_idx, out_path):
    """Generate a 6-panel pipeline image for one object."""
    mask = masks[obj_idx]
    mask_uint8 = (mask > 0).astype(np.uint8)
    H_img, W_img = rgb.shape[:2]
    area = int(np.sum(mask_uint8))

    # --- Compute shape metrics ---
    contours_full, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_full:
        print(f"  Object #{obj_idx}: no contours found, skipping")
        return None
    cnt_full = max(contours_full, key=cv2.contourArea)
    contour_area = cv2.contourArea(cnt_full)
    perimeter = cv2.arcLength(cnt_full, True)
    circularity = (4 * np.pi * contour_area) / (perimeter ** 2) if perimeter > 0 else 0
    rect = cv2.minAreaRect(cnt_full)
    w_rect, h_rect = rect[1]
    if h_rect > 0 and w_rect > 0:
        aspect_ratio = max(w_rect, h_rect) / min(w_rect, h_rect)
    else:
        aspect_ratio = 1.0
    hull_full = cv2.convexHull(cnt_full)
    hull_area = cv2.contourArea(hull_full)
    solidity = contour_area / hull_area if hull_area > 0 else 1.0

    is_circular = circularity > 0.7
    is_elongated = aspect_ratio > 2.0

    if is_circular:
        shape_label = f"circular, circ={circularity:.2f}"
    elif is_elongated:
        shape_label = f"elongated, AR={aspect_ratio:.1f}"
    else:
        shape_label = f"compact, circ={circularity:.2f}"

    # --- Crop around object ---
    coords = np.where(mask_uint8 > 0)
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    pad = 20
    y_min_c = max(0, y_min - pad)
    y_max_c = min(H_img, y_max + pad)
    x_min_c = max(0, x_min - pad)
    x_max_c = min(W_img, x_max + pad)

    crop_rgb = rgb[y_min_c:y_max_c, x_min_c:x_max_c].copy()
    crop_mask = mask_uint8[y_min_c:y_max_c, x_min_c:x_max_c]

    # --- DT on crop ---
    dt = cv2.distanceTransform(crop_mask, cv2.DIST_L2, 5)
    max_dt = dt.max()
    _, _, _, dt_loc = cv2.minMaxLoc(dt)
    dtx, dty = dt_loc

    # --- Moments on crop ---
    M = cv2.moments(crop_mask)
    if M["m00"] == 0:
        return None
    mx_c = M["m10"] / M["m00"]
    my_c = M["m01"] / M["m00"]
    mx_i, my_i = int(round(mx_c)), int(round(my_c))
    mx_i = np.clip(mx_i, 0, crop_mask.shape[1] - 1)
    my_i = np.clip(my_i, 0, crop_mask.shape[0] - 1)

    inside = crop_mask[my_i, mx_i] > 0
    edge_d = dt[my_i, mx_i] if inside else 0.0
    min_clearance = max_dt * 0.20

    # --- Decide method and final point ---
    if is_circular:
        method_name = "Distance Transform"
        final_cx, final_cy = dtx, dty
        safety_result = "N/A (circular -> DT)"
    else:
        method_name = "Moments"
        if inside and edge_d >= min_clearance:
            final_cx, final_cy = mx_i, my_i
            safety_result = f"SAFE ({edge_d:.0f}px >= {min_clearance:.0f}px)"
        elif inside:
            edge_ratio = edge_d / max_dt if max_dt > 0 else 0
            dt_weight = min(1.0 - (edge_ratio / 0.20), 0.7)
            final_cx = int((1.0 - dt_weight) * mx_c + dt_weight * dtx)
            final_cy = int((1.0 - dt_weight) * my_c + dt_weight * dty)
            safety_result = f"BLEND ({edge_d:.0f}px < {min_clearance:.0f}px)"
        else:
            distances = (coords[1] - (mx_c + x_min_c))**2 + (coords[0] - (my_c + y_min_c))**2
            nearest_idx = np.argmin(distances)
            nearest_x = float(coords[1][nearest_idx]) - x_min_c
            nearest_y = float(coords[0][nearest_idx]) - y_min_c
            final_cx = int(0.5 * nearest_x + 0.5 * dtx)
            final_cy = int(0.5 * nearest_y + 0.5 * dty)
            safety_result = "OUTSIDE -> blend nearest+DT"

    global_cx = final_cx + x_min_c
    global_cy = final_cy + y_min_c

    # --- Build the 6-panel image ---
    panel_w, panel_h = 280, 220
    canvas_w = panel_w * 3 + 80
    canvas_h = panel_h * 2 + 160
    img = np.full((canvas_h, canvas_w, 3), BG, dtype=np.uint8)

    title = f"Pipeline: Object #{obj_idx} ({shape_label})"
    put_text_center(img, title, 25, 0.55, YELLOW, 2)

    def resize_panel(src, pw=panel_w, ph=panel_h):
        return cv2.resize(src, (pw, ph), interpolation=cv2.INTER_LINEAR)

    cnts, _ = cv2.findContours(crop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt_crop = max(cnts, key=cv2.contourArea) if cnts else None

    # ── PANEL 1: Segmented Mask ──
    p1 = crop_rgb.copy()
    p1[crop_mask > 0] = (p1[crop_mask > 0] * 0.5 + np.array(OBJ_COLORS[obj_idx % len(OBJ_COLORS)]) * 0.5).astype(np.uint8)
    cv2.drawContours(p1, cnts, -1, WHITE, 1)
    p1 = resize_panel(p1)
    sx, sy = 20, 55
    img[sy:sy + panel_h, sx:sx + panel_w] = p1
    put_text(img, "1. Segmented Mask", (sx, sy + panel_h + 15), 0.45, WHITE, 1)
    put_text(img, f"   area={area}px", (sx, sy + panel_h + 32), 0.38, GRAY)

    # ── PANEL 2: Shape Analysis ──
    p2 = np.full((crop_mask.shape[0], crop_mask.shape[1], 3), BG, dtype=np.uint8)
    p2[crop_mask > 0] = [80, 80, 80]
    cv2.drawContours(p2, cnts, -1, WHITE, 1)
    if cnt_crop is not None:
        rect_crop = cv2.minAreaRect(cnt_crop)
        box = cv2.boxPoints(rect_crop).astype(np.int32)
        cv2.polylines(p2, [box], True, YELLOW, 2)
        hull_crop = cv2.convexHull(cnt_crop)
        cv2.polylines(p2, [hull_crop], True, CYAN, 1)
    p2 = resize_panel(p2)
    sx2 = 20 + panel_w + 20
    img[sy:sy + panel_h, sx2:sx2 + panel_w] = p2
    put_text(img, "2. Shape Analysis", (sx2, sy + panel_h + 15), 0.45, WHITE, 1)
    put_text(img, f"   circ={circularity:.3f} AR={aspect_ratio:.2f} sol={solidity:.3f}", (sx2, sy + panel_h + 32), 0.33, GRAY)
    if is_circular:
        put_text(img, f"   CIRCULAR -> use DT", (sx2, sy + panel_h + 48), 0.38, GREEN)
    elif is_elongated:
        put_text(img, f"   ELONGATED -> use Moments", (sx2, sy + panel_h + 48), 0.38, ORANGE)
    else:
        put_text(img, f"   COMPACT -> use Moments", (sx2, sy + panel_h + 48), 0.38, ORANGE)

    # ── PANEL 3: Distance Transform ──
    dt_norm = (dt / max(max_dt, 1) * 255).astype(np.uint8)
    p3 = cv2.applyColorMap(dt_norm, cv2.COLORMAP_JET)
    p3[crop_mask == 0] = np.array(BG)
    cv2.circle(p3, (dtx, dty), 5, WHITE, 2)
    put_text(p3, "DT max", (dtx + 6, dty - 4), 0.3, WHITE)
    put_text(p3, f"r={max_dt:.0f}px", (dtx + 6, dty + 12), 0.25, WHITE)
    p3 = resize_panel(p3)
    sx3 = 20 + (panel_w + 20) * 2
    img[sy:sy + panel_h, sx3:sx3 + panel_w] = p3
    put_text(img, "3. Distance Transform", (sx3, sy + panel_h + 15), 0.45, WHITE, 1)
    if is_circular:
        put_text(img, f"   DT center = geometric center!", (sx3, sy + panel_h + 32), 0.38, GREEN)
    else:
        put_text(img, f"   DT biased to widest part!", (sx3, sy + panel_h + 32), 0.38, RED)

    # ── PANEL 4: Method + Safety Check ──
    p4 = crop_rgb.copy()
    p4[crop_mask > 0] = (p4[crop_mask > 0] * 0.6 + np.array([80, 80, 80]) * 0.4).astype(np.uint8)
    cv2.drawContours(p4, cnts, -1, WHITE, 1)

    if is_circular:
        cv2.circle(p4, (dtx, dty), 7, GREEN, -1)
        cv2.circle(p4, (dtx, dty), 7, WHITE, 2)
        put_text(p4, f"DT center", (dtx + 8, dty - 5), 0.3, GREEN)
        cv2.circle(p4, (mx_i, my_i), 4, ORANGE, -1)
        put_text(p4, f"moments", (mx_i - 35, my_i - 8), 0.25, ORANGE)
    else:
        cv2.circle(p4, (mx_i, my_i), max(int(edge_d), 2), (0, 0, 200), 1)
        cv2.circle(p4, (mx_i, my_i), 5, RED, -1)
        put_text(p4, f"moments", (mx_i + 6, my_i - 6), 0.3, RED)
        put_text(p4, f"edge={edge_d:.0f}px", (mx_i + 6, my_i + 10), 0.3, RED)
        cv2.circle(p4, (dtx, dty), 4, BLUE, -1)
        put_text(p4, "DT", (dtx + 5, dty - 3), 0.25, BLUE)

    p4 = resize_panel(p4)
    sy2 = sy + panel_h + 60
    sx4 = 20
    img[sy2:sy2 + panel_h, sx4:sx4 + panel_w] = p4

    if is_circular:
        put_text(img, "4. Circular -> DT (no safety needed)", (sx4, sy2 + panel_h + 15), 0.42, GREEN, 1)
        put_text(img, f"   DT center IS the geometric center", (sx4, sy2 + panel_h + 32), 0.38, GREEN)
    else:
        put_text(img, "4. Moments + Safety Check", (sx4, sy2 + panel_h + 15), 0.45, WHITE, 1)
        if inside and edge_d >= min_clearance:
            put_text(img, f"   edge={edge_d:.0f}px >= {min_clearance:.0f}px -> SAFE!", (sx4, sy2 + panel_h + 32), 0.38, GREEN)
        elif inside:
            put_text(img, f"   edge={edge_d:.0f}px < {min_clearance:.0f}px -> BLEND!", (sx4, sy2 + panel_h + 32), 0.38, ORANGE)
        else:
            put_text(img, f"   OUTSIDE mask! -> blend nearest+DT", (sx4, sy2 + panel_h + 32), 0.38, RED)

    # ── PANEL 5: Final Result ──
    p5 = crop_rgb.copy()
    p5[crop_mask > 0] = (p5[crop_mask > 0] * 0.5 + np.array([100, 180, 100]) * 0.5).astype(np.uint8)
    cv2.drawContours(p5, cnts, -1, WHITE, 1)

    if is_circular:
        cv2.circle(p5, (final_cx, final_cy), 7, GREEN, -1)
        cv2.circle(p5, (final_cx, final_cy), 7, WHITE, 2)
        put_text(p5, "RESULT", (final_cx + 8, final_cy + 4), 0.35, GREEN, 1)
    else:
        cv2.circle(p5, (mx_i, my_i), 4, RED, -1)
        put_text(p5, "moments", (mx_i - 35, my_i - 8), 0.25, RED)
        cv2.circle(p5, (dtx, dty), 4, BLUE, -1)
        put_text(p5, "DT max", (dtx + 5, dty - 5), 0.25, BLUE)
        cv2.circle(p5, (final_cx, final_cy), 7, GREEN, -1)
        cv2.circle(p5, (final_cx, final_cy), 7, WHITE, 2)
        put_text(p5, "RESULT", (final_cx + 8, final_cy + 4), 0.35, GREEN, 1)
        if abs(mx_i - final_cx) + abs(my_i - final_cy) > 3:
            cv2.arrowedLine(p5, (mx_i, my_i), (final_cx, final_cy), YELLOW, 1, tipLength=0.15)

    p5 = resize_panel(p5)
    sx5 = 20 + panel_w + 20
    img[sy2:sy2 + panel_h, sx5:sx5 + panel_w] = p5

    if is_circular:
        put_text(img, f"5. FINAL: DT center directly", (sx5, sy2 + panel_h + 15), 0.42, GREEN, 1)
    elif inside and edge_d >= min_clearance:
        put_text(img, f"5. FINAL: Moments directly (safe)", (sx5, sy2 + panel_h + 15), 0.42, GREEN, 1)
    else:
        put_text(img, f"5. FINAL: Blended result", (sx5, sy2 + panel_h + 15), 0.42, GREEN, 1)
    put_text(img, f"   RED DOT at ({global_cx}, {global_cy})", (sx5, sy2 + panel_h + 32), 0.38, GREEN)

    # ── PANEL 6: Full Frame ──
    p6 = rgb.copy()
    for mi, m in enumerate(masks):
        mb = m > 0
        if np.sum(mb) < 100:
            continue
        color = OBJ_COLORS[mi % len(OBJ_COLORS)]
        p6[mb] = (p6[mb] * 0.5 + np.array(color) * 0.5).astype(np.uint8)

    # Highlight current object
    for c in contours_full:
        cv2.drawContours(p6, [c], -1, YELLOW, 2)

    cv2.circle(p6, (global_cx, global_cy), 8, (0, 0, 255), -1)
    cv2.circle(p6, (global_cx, global_cy), 8, WHITE, 2)
    cv2.putText(p6, f"#{obj_idx}", (global_cx + 10, global_cy - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2, cv2.LINE_AA)

    p6_h = panel_h
    p6_w = int(p6.shape[1] * p6_h / p6.shape[0])
    p6 = cv2.resize(p6, (p6_w, p6_h))
    sx6 = 20 + (panel_w + 20) * 2
    available_w = canvas_w - sx6
    if p6_w > available_w:
        p6 = p6[:, :available_w]
        p6_w = available_w
    img[sy2:sy2 + panel_h, sx6:sx6 + p6_w] = p6
    put_text(img, "6. Full Frame Result", (sx6, sy2 + panel_h + 15), 0.45, WHITE, 1)
    put_text(img, f"   Yellow = Object #{obj_idx}", (sx6, sy2 + panel_h + 32), 0.38, YELLOW)

    cv2.imwrite(out_path, img)
    print(f"  Saved: {out_path}")

    return {
        'obj_idx': obj_idx,
        'area': area,
        'circularity': circularity,
        'aspect_ratio': aspect_ratio,
        'solidity': solidity,
        'is_circular': is_circular,
        'method': method_name,
        'safety': safety_result,
        'center': (global_cx, global_cy),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading image: {IMG_PATH}")
    rgb = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(f"Cannot load: {IMG_PATH}")
    print(f"  Image size: {rgb.shape[1]}x{rgb.shape[0]}")

    # Step 1: Segment objects
    print("\nSegmenting objects...")
    masks = segment_objects(rgb)

    if not masks:
        print("No objects detected! Try adjusting segmentation parameters.")
        return

    # Save segmentation overview
    overview = rgb.copy()
    for i, m in enumerate(masks):
        mb = m > 0
        color = OBJ_COLORS[i % len(OBJ_COLORS)]
        overview[mb] = (overview[mb] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overview, cnts, -1, WHITE, 2)
        M = cv2.moments(m)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(overview, f"#{i}", (cx - 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
    overview_path = os.path.join(OUT_DIR, "00_segmentation_overview.png")
    cv2.imwrite(overview_path, overview)
    print(f"\n  Segmentation overview: {overview_path}")

    # Step 2: Generate pipeline for each object
    print(f"\nGenerating pipeline for {len(masks)} objects...\n")
    results = []
    for obj_idx in range(len(masks)):
        out_path = os.path.join(OUT_DIR, f"pipeline_obj{obj_idx}.png")
        result = generate_object_pipeline(rgb, masks, obj_idx, out_path)
        if result:
            results.append(result)

    # Print summary
    print(f"\n{'='*80}")
    print(f"  SUMMARY: All Detected Objects")
    print(f"{'='*80}")
    print(f"  {'Obj':>3} | {'Area':>7} | {'Circ':>5} | {'AR':>4} | {'Sol':>5} | {'Shape':>10} | {'Method':>10} | {'Safety':>30} | {'Center':>12}")
    print(f"  {'-'*3}-+-{'-'*7}-+-{'-'*5}-+-{'-'*4}-+-{'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*30}-+-{'-'*12}")
    for r in results:
        shape = "CIRC" if r['is_circular'] else "ELONG" if r['aspect_ratio'] > 2 else "COMP"
        print(f"  #{r['obj_idx']:>2} | {r['area']:>7} | {r['circularity']:>5.3f} | {r['aspect_ratio']:>4.2f} | {r['solidity']:>5.3f} | {shape:>10} | {r['method']:>10} | {r['safety']:>30} | ({r['center'][0]:>3}, {r['center'][1]:>3})")

    print(f"\nAll images saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
