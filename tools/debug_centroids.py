# Copyright (c) 2024. Debug tool for centroid computation analysis.
"""
Debug centroid computation from instance masks (PNG per object).

Usage:
    python tools/debug_centroids.py \\
        --image-path ./scene.png \\
        --amodal-masks-dir ./masks/scene/ \\
        --output-dir ./output_centroids
"""

import argparse
import glob
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_binary_masks(mask_dir: str, width: int, height: int):
    paths = sorted(
        glob.glob(os.path.join(mask_dir, "*.png"))
        + glob.glob(os.path.join(mask_dir, "*.jpg"))
    )
    if not paths:
        raise FileNotFoundError(f"No masks in {mask_dir}")
    masks = []
    for p in paths:
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        m = (m > 0).astype(np.float32)
        if m.shape[1] != width or m.shape[0] != height:
            m = cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST)
        masks.append(m)
    if not masks:
        raise ValueError(f"Could not read masks from {mask_dir}")
    return np.stack(masks, axis=0)


def analyze_mask_centroid(mask: np.ndarray, method_name: str, obj_id: int):
    mask_uint8 = (mask > 0).astype(np.uint8)
    results = {}
    dist_transform = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
    _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
    results["distance_transform"] = {
        "centroid": max_loc,
        "max_distance": max_val,
        "dist_map": dist_transform,
    }
    M = cv2.moments(mask_uint8)
    if M["m00"] > 0:
        cx_moments = M["m10"] / M["m00"]
        cy_moments = M["m01"] / M["m00"]
        results["moments"] = {"centroid": (cx_moments, cy_moments)}
    else:
        results["moments"] = {"centroid": None}
    coords = np.where(mask_uint8 > 0)
    if len(coords[0]) > 0:
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        results["mask_bbox"] = {
            "centroid": ((x_min + x_max) / 2, (y_min + y_max) / 2),
            "bbox": (x_min, y_min, x_max, y_max),
        }
    else:
        results["mask_bbox"] = {"centroid": None}
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        largest = max(contours, key=cv2.contourArea)
        results["contour_area"] = cv2.contourArea(largest)
        results["num_contours"] = len(contours)
    results["mask_sum"] = np.sum(mask_uint8)
    results["mask_unique_values"] = np.unique(mask.flatten())[:10]
    results["mask_dtype"] = str(mask.dtype)
    results["mask_shape"] = mask.shape
    return results


def visualize_debug(rgb_img, masks, visible_masks, analysis_results, output_path):
    num_objects = len(masks)
    fig, axes = plt.subplots(num_objects, 5, figsize=(20, 4 * num_objects))
    if num_objects == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_objects):
        amodal_mask = masks[i]
        visible_mask = visible_masks[i]
        results = analysis_results[i]

        axes[i, 0].imshow(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f"Object #{i}: All Centroids")
        colors = {"distance_transform": "red", "moments": "green", "mask_bbox": "blue"}
        for method, color in colors.items():
            if results[method]["centroid"] is not None:
                cx, cy = results[method]["centroid"]
                axes[i, 0].scatter(cx, cy, c=color, s=100, marker="x", linewidths=2, label=method)
        axes[i, 0].legend(fontsize=8)

        axes[i, 1].imshow(amodal_mask, cmap="gray")
        axes[i, 1].set_title(f"Amodal\nsum={results['mask_sum']}")

        axes[i, 2].imshow(visible_mask, cmap="gray")
        axes[i, 2].set_title(f"Visible\nsum={np.sum(visible_mask > 0)}")

        dist_map = results["distance_transform"]["dist_map"]
        axes[i, 3].imshow(dist_map, cmap="hot")
        cx, cy = results["distance_transform"]["centroid"]
        axes[i, 3].scatter(cx, cy, c="cyan", s=150, marker="*", linewidths=2)
        axes[i, 3].set_title("Distance Transform")

        diff_mask = (amodal_mask > 0).astype(np.uint8) - (visible_mask > 0).astype(
            np.uint8
        )
        axes[i, 4].imshow(diff_mask, cmap="RdBu", vmin=-1, vmax=1)
        axes[i, 4].set_title("Amodal − Visible")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Debug visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Debug centroid computation from masks")
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--amodal-masks-dir", required=True)
    parser.add_argument("--visible-masks-dir", default=None)
    parser.add_argument("--output-dir", default="./output_centroids")
    args = parser.parse_args()

    rgb_img = cv2.imread(args.image_path)
    if rgb_img is None:
        raise ValueError(f"Could not read image: {args.image_path}")
    h, w = rgb_img.shape[:2]

    pred_masks = load_binary_masks(args.amodal_masks_dir, w, h)
    if args.visible_masks_dir:
        pred_visible = load_binary_masks(args.visible_masks_dir, w, h)
        if pred_visible.shape[0] != pred_masks.shape[0]:
            raise ValueError("Visible mask count must match amodal mask count")
    else:
        pred_visible = pred_masks.copy()

    print(f"\n{'='*60}\nDEBUG: {args.image_path}\n{'='*60}")
    print(f"Objects: {len(pred_masks)}")

    analysis_results = []
    for i in range(len(pred_masks)):
        results = analyze_mask_centroid(pred_masks[i], "all", i)
        analysis_results.append(results)
        print(f"\n--- Object #{i} ---")
        dt = results["distance_transform"]["centroid"]
        mom = results["moments"]["centroid"]
        print(f"  DT: {dt}  Moments: {mom}")
        if dt and mom:
            dist = np.hypot(dt[0] - mom[0], dt[1] - mom[1])
            print(f"  |DT − Moments| = {dist:.1f} px")

    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.image_path))[0]
    out = os.path.join(args.output_dir, f"{base}_debug.png")
    visualize_debug(rgb_img, pred_masks, pred_visible, analysis_results, out)


if __name__ == "__main__":
    main()
