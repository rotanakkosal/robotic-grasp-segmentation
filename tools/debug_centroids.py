# Copyright (c) 2024. Debug tool for centroid computation analysis.
"""
Debug Centroid Computation

This script visualizes the intermediate steps of centroid computation
to diagnose why centroids may appear at incorrect positions.

Usage:
    python tools/debug_centroids.py --image-path ./sample_data/IMG_1474-Photoroom.png
"""

import argparse
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adet.config import get_cfg
from adet.utils.post_process import detector_postprocess, DefaultPredictor
from utils import normalize_depth, inpaint_depth


def analyze_mask_centroid(mask: np.ndarray, method_name: str, obj_id: int):
    """
    Analyze and visualize centroid computation for a single mask.
    Returns dict with all centroid methods results.
    """
    mask_uint8 = (mask > 0).astype(np.uint8)

    results = {}

    # 1. Distance Transform Analysis
    dist_transform = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
    _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
    results['distance_transform'] = {
        'centroid': max_loc,
        'max_distance': max_val,
        'dist_map': dist_transform
    }

    # 2. Moments (Center of Mass)
    M = cv2.moments(mask_uint8)
    if M["m00"] > 0:
        cx_moments = M["m10"] / M["m00"]
        cy_moments = M["m01"] / M["m00"]
        results['moments'] = {'centroid': (cx_moments, cy_moments)}
    else:
        results['moments'] = {'centroid': None}

    # 3. Mask Bounding Box Center
    coords = np.where(mask_uint8 > 0)
    if len(coords[0]) > 0:
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        cx_bbox = (x_min + x_max) / 2
        cy_bbox = (y_min + y_max) / 2
        results['mask_bbox'] = {
            'centroid': (cx_bbox, cy_bbox),
            'bbox': (x_min, y_min, x_max, y_max)
        }
    else:
        results['mask_bbox'] = {'centroid': None}

    # 4. Contour-based analysis
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        results['contour_area'] = cv2.contourArea(largest_contour)
        results['num_contours'] = len(contours)

        # Check if mask has holes (multiple contours)
        all_contours, _ = cv2.findContours(mask_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        results['has_holes'] = len(all_contours) > len(contours)

    # 5. Mask statistics
    results['mask_sum'] = np.sum(mask_uint8)
    results['mask_unique_values'] = np.unique(mask.flatten())[:10]  # First 10 unique values
    results['mask_dtype'] = str(mask.dtype)
    results['mask_shape'] = mask.shape

    return results


def visualize_debug(rgb_img, masks, visible_masks, boxes, analysis_results, output_path):
    """Create comprehensive debug visualization."""
    num_objects = len(masks)

    # Create figure with subplots for each object
    fig, axes = plt.subplots(num_objects, 5, figsize=(20, 4 * num_objects))
    if num_objects == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_objects):
        amodal_mask = masks[i]
        visible_mask = visible_masks[i]
        results = analysis_results[i]

        # Column 1: Original RGB with all centroids
        axes[i, 0].imshow(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f'Object #{i}: All Centroids')

        # Plot all centroids
        colors = {'distance_transform': 'red', 'moments': 'green', 'mask_bbox': 'blue'}
        for method, color in colors.items():
            if results[method]['centroid'] is not None:
                cx, cy = results[method]['centroid']
                axes[i, 0].scatter(cx, cy, c=color, s=100, marker='x', linewidths=2, label=method)
        axes[i, 0].legend(fontsize=8)

        # Column 2: Amodal Mask
        axes[i, 1].imshow(amodal_mask, cmap='gray')
        axes[i, 1].set_title(f'Amodal Mask\nsum={results["mask_sum"]}, dtype={results["mask_dtype"]}')

        # Column 3: Visible Mask
        axes[i, 2].imshow(visible_mask, cmap='gray')
        vis_sum = np.sum(visible_mask > 0)
        axes[i, 2].set_title(f'Visible Mask\nsum={vis_sum}')

        # Column 4: Distance Transform with max point
        dist_map = results['distance_transform']['dist_map']
        axes[i, 3].imshow(dist_map, cmap='hot')
        cx, cy = results['distance_transform']['centroid']
        axes[i, 3].scatter(cx, cy, c='cyan', s=150, marker='*', linewidths=2)
        axes[i, 3].set_title(f'Distance Transform\nmax={results["distance_transform"]["max_distance"]:.1f} at ({cx:.0f},{cy:.0f})')

        # Column 5: Difference between amodal and visible
        diff_mask = (amodal_mask > 0).astype(np.uint8) - (visible_mask > 0).astype(np.uint8)
        axes[i, 4].imshow(diff_mask, cmap='RdBu', vmin=-1, vmax=1)
        axes[i, 4].set_title(f'Amodal - Visible\n(red=amodal only, blue=visible only)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Debug visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Debug centroid computation")
    parser.add_argument("--config-file", default="configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml")
    parser.add_argument("--image-path", required=True, help="Path to input image")
    parser.add_argument("--output-dir", default="./output_centroids", help="Output directory")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    args = parser.parse_args()

    # Setup model
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = args.confidence_threshold
    predictor = DefaultPredictor(cfg)
    W, H = cfg.INPUT.IMG_SIZE

    # Load and process image
    rgb_img = cv2.imread(args.image_path)
    if rgb_img is None:
        raise ValueError(f"Could not read image: {args.image_path}")

    rgb_resized = cv2.resize(rgb_img, (W, H))

    # Create dummy depth
    depth_img_raw = np.ones((rgb_img.shape[0], rgb_img.shape[1]), dtype=np.float32) * 800.0
    depth_normalized = normalize_depth(depth_img_raw.copy())
    depth_normalized = cv2.resize(depth_normalized, (W, H), interpolation=cv2.INTER_NEAREST)
    depth_normalized = inpaint_depth(depth_normalized)

    # Prepare input
    if cfg.INPUT.DEPTH and not cfg.INPUT.DEPTH_ONLY:
        uoais_input = np.concatenate([rgb_resized, depth_normalized], -1)
    else:
        uoais_input = rgb_resized

    # Run inference
    outputs = predictor(uoais_input)
    instances = detector_postprocess(outputs['instances'], H, W).to('cpu')

    # Extract predictions
    pred_masks = instances.pred_masks.detach().cpu().numpy()
    pred_visible_masks = instances.pred_visible_masks.detach().cpu().numpy()
    pred_boxes = instances.pred_boxes.tensor.detach().cpu().numpy()

    print(f"\n{'='*60}")
    print(f"DEBUG ANALYSIS: {args.image_path}")
    print(f"{'='*60}")
    print(f"Detected {len(pred_masks)} objects")
    print(f"Mask shape: {pred_masks.shape}")
    print(f"Mask dtype: {pred_masks.dtype}")
    print(f"Mask value range: [{pred_masks.min()}, {pred_masks.max()}]")
    print()

    # Analyze each object
    analysis_results = []
    for i in range(len(pred_masks)):
        print(f"\n--- Object #{i} ---")
        results = analyze_mask_centroid(pred_masks[i], "all", i)
        analysis_results.append(results)

        print(f"  Mask unique values (first 10): {results['mask_unique_values']}")
        print(f"  Mask sum (area): {results['mask_sum']}")
        print(f"  Num contours: {results.get('num_contours', 'N/A')}")
        print(f"  Has holes: {results.get('has_holes', 'N/A')}")
        print()
        print(f"  Centroids:")
        print(f"    - Distance Transform: {results['distance_transform']['centroid']} (max_dist={results['distance_transform']['max_distance']:.2f})")
        print(f"    - Moments (CoM):      {results['moments']['centroid']}")
        print(f"    - Mask BBox Center:   {results['mask_bbox']['centroid']}")

        # Check if centroids differ significantly
        dt_centroid = results['distance_transform']['centroid']
        mom_centroid = results['moments']['centroid']
        if dt_centroid and mom_centroid:
            dist = np.sqrt((dt_centroid[0] - mom_centroid[0])**2 + (dt_centroid[1] - mom_centroid[1])**2)
            print(f"    - Distance between DT and Moments: {dist:.1f} pixels")
            if dist > 20:
                print(f"    ⚠️  WARNING: Large discrepancy - mask may be irregular or have artifacts")

    # Create visualization
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.image_path))[0]
    output_path = os.path.join(args.output_dir, f"{base_name}_debug.png")

    visualize_debug(rgb_resized, pred_masks, pred_visible_masks, pred_boxes, analysis_results, output_path)

    print(f"\n{'='*60}")
    print("DIAGNOSIS SUMMARY")
    print(f"{'='*60}")
    print("""
If distance_transform centroids are at edges:
1. Check if mask has irregular shape or multiple lobes
2. Check if amodal mask incorrectly extends beyond visible
3. Consider using 'moments' method for center-of-mass
4. Consider using 'mask_bbox' for simple geometric center

Recommended fix: Use moments-based centroid for general objects,
distance_transform only for objects where edge-distance matters (suction gripping).
""")


if __name__ == "__main__":
    main()
