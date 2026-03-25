#!/usr/bin/env python3
"""
Verify centroid approach on T-Less dataset with REAL depth data.
Compares:
  1. Suction method WITH real depth (full pipeline: surface normals + flatness)
  2. Adaptive method WITHOUT depth (our moments-based fallback)
  3. Ground truth mask centers for reference
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import json
import glob

from centroid_utils import (
    compute_suction_grasp_point,
    compute_centroid_adaptive,
    compute_centroid_distance_transform,
    compute_centroid_moments,
    compute_all_centroids,
    analyze_mask_shape,
    draw_centroids,
    ObjectCentroid,
)


def load_tless_frame(scene_dir, frame_id):
    """Load a single T-Less frame: RGB, depth, masks, camera."""
    fid = f"{frame_id:06d}"

    rgb = cv2.imread(os.path.join(scene_dir, "rgb", f"{fid}.png"))
    depth_raw = cv2.imread(os.path.join(scene_dir, "depth", f"{fid}.png"), cv2.IMREAD_UNCHANGED)

    with open(os.path.join(scene_dir, "scene_camera.json")) as f:
        cam_data = json.load(f)
    cam = cam_data[str(frame_id)]
    depth_scale = cam["depth_scale"]  # 0.1 = each unit is 0.1mm
    cam_K = cam["cam_K"]  # [fx, 0, cx, 0, fy, cy, 0, 0, 1]

    # Convert depth to mm
    depth_mm = depth_raw.astype(np.float32) * depth_scale

    # Load GT masks (amodal) and visible masks
    mask_files = sorted(glob.glob(os.path.join(scene_dir, "mask", f"{fid}_*.png")))
    visib_files = sorted(glob.glob(os.path.join(scene_dir, "mask_visib", f"{fid}_*.png")))

    masks = []
    visib_masks = []
    for mf in mask_files:
        m = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
        masks.append(m)
    for vf in visib_files:
        v = cv2.imread(vf, cv2.IMREAD_GRAYSCALE)
        visib_masks.append(v)

    # Load GT info for visibility fractions
    with open(os.path.join(scene_dir, "scene_gt_info.json")) as f:
        gt_info = json.load(f)
    frame_info = gt_info[str(frame_id)]

    camera_intrinsics = {
        'fx': cam_K[0], 'fy': cam_K[4],
        'cx': cam_K[2], 'cy': cam_K[5]
    }

    return rgb, depth_mm, masks, visib_masks, frame_info, camera_intrinsics


def verify_frame(scene_dir, frame_id, output_dir):
    """Verify centroids on a single frame."""
    rgb, depth_mm, masks, visib_masks, gt_info, cam_intrinsics = load_tless_frame(scene_dir, frame_id)
    H, W = rgb.shape[:2]

    print(f"\n{'='*70}")
    print(f"  Frame {frame_id:06d}: {len(masks)} objects, depth range: "
          f"{depth_mm[depth_mm > 0].min():.0f}-{depth_mm[depth_mm > 0].max():.0f}mm")
    print(f"{'='*70}")

    # Create visualization
    vis = rgb.copy()

    results = []
    for i, (mask, visib_mask) in enumerate(zip(masks, visib_masks)):
        if np.sum(mask > 0) < 100:
            continue

        visib_frac = gt_info[i]["visib_fract"] if i < len(gt_info) else 1.0
        shape = analyze_mask_shape(mask)
        shape_str = "?"
        if shape:
            if shape.is_circular: shape_str = "circular"
            elif shape.is_elongated: shape_str = f"elongated(ar={shape.aspect_ratio:.1f})"
            elif shape.solidity < 0.85: shape_str = f"irregular(sol={shape.solidity:.2f})"
            else: shape_str = f"compact(circ={shape.circularity:.2f})"

        # Method 1: Suction with REAL depth (full pipeline)
        suction_result = compute_suction_grasp_point(
            mask, depth_mm,
            suction_cup_radius=15,
            prefer_center=True
        )

        # Method 2: Adaptive WITHOUT depth (our fallback for phone photos)
        adaptive_result = compute_centroid_adaptive(mask)

        # Method 3: Simple moments (reference)
        moments_result = compute_centroid_moments(mask)

        # Method 4: Distance transform (reference)
        dt_result = compute_centroid_distance_transform(mask)

        # Compute distances between methods
        if suction_result and adaptive_result:
            suction_adaptive_dist = np.sqrt(
                (suction_result[0] - adaptive_result[0])**2 +
                (suction_result[1] - adaptive_result[1])**2
            )
        else:
            suction_adaptive_dist = -1

        print(f"\n  Object #{i} ({shape_str}, visibility={visib_frac:.0%}):")
        if suction_result:
            # Check depth at suction point
            sx, sy = int(suction_result[0]), int(suction_result[1])
            sx, sy = np.clip(sx, 0, W-1), np.clip(sy, 0, H-1)
            depth_at_point = depth_mm[sy, sx]
            print(f"    Suction (real depth) : ({suction_result[0]:6.1f}, {suction_result[1]:6.1f})  depth={depth_at_point:.0f}mm")
        if adaptive_result:
            print(f"    Adaptive (no depth)  : ({adaptive_result[0]:6.1f}, {adaptive_result[1]:6.1f})")
        if moments_result:
            print(f"    Moments (reference)  : ({moments_result[0]:6.1f}, {moments_result[1]:6.1f})")
        if suction_adaptive_dist >= 0:
            print(f"    Distance suction↔adaptive: {suction_adaptive_dist:.1f}px")

        results.append({
            'obj_id': i, 'shape': shape_str, 'visib': visib_frac,
            'suction': suction_result, 'adaptive': adaptive_result,
            'moments': moments_result, 'dt': dt_result,
            'dist': suction_adaptive_dist
        })

        # Draw mask overlay
        mask_bool = mask > 0
        color = np.random.randint(80, 220, 3).tolist()
        overlay = vis.copy()
        overlay[mask_bool] = (overlay[mask_bool] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
        vis = overlay
        contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (255, 255, 255), 1)

    # Use compute_all_centroids to get both geometric center and grasp point
    pred_masks_arr = np.array(masks)
    pred_visible_arr = np.array(visib_masks) if visib_masks else pred_masks_arr
    pred_boxes = np.zeros((len(masks), 4), dtype=int)
    pred_occs = np.zeros(len(masks), dtype=int)
    scores_arr = np.ones(len(masks))
    for idx, m in enumerate(masks):
        coords_m = np.where(m > 0)
        if len(coords_m[0]) > 0:
            pred_boxes[idx] = [coords_m[1].min(), coords_m[0].min(), coords_m[1].max(), coords_m[0].max()]

    centroids_obj = compute_all_centroids(
        pred_masks=pred_masks_arr,
        pred_visible_masks=pred_visible_arr,
        pred_boxes=pred_boxes,
        pred_occlusions=pred_occs,
        scores=scores_arr,
        depth_image=depth_mm,
        method="suction",
        suction_cup_radius=15
    )

    # Draw both: RED = geometric center, GREEN = suction grasp point
    vis = draw_centroids(vis, centroids_obj, draw_amodal=True, draw_grasp=True, draw_labels=True)

    # Add legend
    cv2.putText(vis, "RED dot = Geometric Center", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(vis, "GREEN cross = Suction Grasp Point", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(vis, "White line = center-to-grasp offset", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"tless_{frame_id:06d}_centroids.png")
    cv2.imwrite(output_path, vis)
    print(f"\n  Saved: {output_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Verify centroid methods on T-Less dataset")
    parser.add_argument("--scene-dir", type=str,
                        default="/home/kosal/cbnu_project/AI/picking-arm-robot/dataset/t_less-data/000003")
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 50, 100, 200, 300, 400, 500],
                        help="Frame IDs to process")
    parser.add_argument("--output-dir", type=str, default="./output_tless_verify")
    args = parser.parse_args()

    np.random.seed(42)

    print("=" * 70)
    print("  T-LESS CENTROID VERIFICATION")
    print("  Comparing: Suction (real depth) vs Adaptive (no depth)")
    print("=" * 70)

    all_results = []
    for fid in args.frames:
        try:
            results = verify_frame(args.scene_dir, fid, args.output_dir)
            all_results.extend(results)
        except Exception as e:
            print(f"  Frame {fid}: ERROR - {e}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    dists = [r['dist'] for r in all_results if r['dist'] >= 0]
    if dists:
        print(f"\n  Total objects analyzed: {len(all_results)}")
        print(f"  Suction ↔ Adaptive distance:")
        print(f"    Mean:   {np.mean(dists):.1f} px")
        print(f"    Median: {np.median(dists):.1f} px")
        print(f"    Max:    {np.max(dists):.1f} px")
        print(f"    < 5px:  {sum(1 for d in dists if d < 5)}/{len(dists)} ({sum(1 for d in dists if d < 5)/len(dists)*100:.0f}%)")
        print(f"    < 10px: {sum(1 for d in dists if d < 10)}/{len(dists)} ({sum(1 for d in dists if d < 10)/len(dists)*100:.0f}%)")
        print(f"    < 20px: {sum(1 for d in dists if d < 20)}/{len(dists)} ({sum(1 for d in dists if d < 20)/len(dists)*100:.0f}%)")

    print(f"\n  Output saved to: {args.output_dir}/")
