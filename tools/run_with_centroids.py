"""
Instance masks → centroid / suction grasp points for robot picking.

This script does not run a neural segmenter. Provide **binary masks** (one PNG per
object, sorted by filename) from your own segmentation method.

Usage:
    python tools/run_with_centroids.py \\
        --image-path ./scene.png \\
        --amodal-masks-dir ./masks/scene/ \\
        --output-dir ./output_centroids

    python tools/run_with_centroids.py \\
        --dataset-path ./sample_data \\
        --output-dir ./output_centroids

**Batch with mask folders:** for each `image_color/foo.png`, use `masks/foo/*.png`
or `amodal_masks/foo/*.png`.

**Bundled sample (no mask folders):** use classic CV with `--simple-masks`, e.g.
`sample_data/arm-robot-Dataset/arm_robot_images` (flat `IMG_*.png`).
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.centroid_utils import (
    compute_all_centroids,
    draw_centroids,
    get_best_grasp_point,
    centroids_to_dict,
)
from tools.pipeline_from_raw_image import segment_objects


def load_binary_masks(mask_dir: str, width: int, height: int) -> np.ndarray:
    """Load *.png / *.jpg from mask_dir, sorted; resize to (width, height)."""
    paths = sorted(
        glob.glob(os.path.join(mask_dir, "*.png"))
        + glob.glob(os.path.join(mask_dir, "*.jpg"))
    )
    if not paths:
        raise FileNotFoundError(f"No mask images in {mask_dir}")
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
        raise ValueError(f"Could not read any masks from {mask_dir}")
    return np.stack(masks, axis=0)


def masks_to_boxes(masks: np.ndarray) -> np.ndarray:
    """masks [N,H,W] → boxes [N,4] as x1,y1,x2,y2 float32."""
    boxes = []
    for i in range(masks.shape[0]):
        ys, xs = np.where(masks[i] > 0)
        if len(xs) == 0:
            boxes.append([0.0, 0.0, 0.0, 0.0])
        else:
            boxes.append(
                [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
            )
    return np.array(boxes, dtype=np.float32)


def visualize_masks_clean(rgb_img, masks, centroids, alpha=0.5):
    vis_img = rgb_img.copy()
    np.random.seed(42)
    colors = []
    for i in range(len(masks)):
        hue = (i * 0.618033988749895) % 1.0
        color = np.array(
            cv2.cvtColor(
                np.uint8([[[hue * 180, 200, 200]]]), cv2.COLOR_HSV2BGR
            )[0, 0],
            dtype=np.float32,
        )
        colors.append(color)

    for i, mask in enumerate(masks):
        mask_bool = mask > 0
        color = colors[i]
        vis_img[mask_bool] = (
            vis_img[mask_bool] * (1 - alpha) + color * alpha
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis_img, contours, -1, (255, 255, 255), 1)

    return draw_centroids(
        vis_img,
        centroids,
        draw_amodal=True,
        draw_grasp=True,
        draw_bbox_center=False,
        draw_bbox=False,
        draw_labels=True,
    )


def resolve_mask_dir_for_image(dataset_path: str, rgb_path: str) -> Optional[str]:
    base = os.path.splitext(os.path.basename(rgb_path))[0]
    for sub in ("masks", "amodal_masks"):
        d = os.path.join(dataset_path, sub, base)
        if not os.path.isdir(d):
            continue
        paths = glob.glob(os.path.join(d, "*.png")) + glob.glob(
            os.path.join(d, "*.jpg")
        )
        if paths:
            return d
    return None


def process_from_masks(
    rgb_img: np.ndarray,
    depth_img_raw: np.ndarray,
    pred_masks: np.ndarray,
    pred_visible_masks: np.ndarray,
    centroid_method: str = "suction",
    camera_intrinsics: Optional[Dict] = None,
    camera_extrinsics: Optional[Dict] = None,
    use_amodal: bool = True,
    draw_bbox: bool = False,
    suction_cup_radius: int = 15,
    workspace_bounds: Optional[Dict] = None,
):
    height, width = rgb_img.shape[:2]
    rgb_work = rgb_img
    depth_raw_resized = depth_img_raw.astype(np.float32)
    if depth_raw_resized.shape[:2] != (height, width):
        depth_raw_resized = cv2.resize(
            depth_raw_resized, (width, height), interpolation=cv2.INTER_NEAREST
        )

    pred_boxes = masks_to_boxes(pred_masks)
    pred_occlusions = np.zeros(len(pred_masks), dtype=np.float32)
    scores = np.ones(len(pred_masks), dtype=np.float32)

    centroids = compute_all_centroids(
        pred_masks=pred_masks,
        pred_visible_masks=pred_visible_masks,
        pred_boxes=pred_boxes,
        pred_occlusions=pred_occlusions,
        scores=scores,
        depth_image=depth_raw_resized,
        camera_intrinsics=camera_intrinsics,
        method=centroid_method,
        use_amodal_for_centroid=use_amodal,
        suction_cup_radius=suction_cup_radius,
    )

    if camera_extrinsics is not None:
        R = camera_extrinsics["R"]
        t = camera_extrinsics["t"]
        for obj in centroids:
            if obj.centroid_3d is not None:
                p = np.array(obj.centroid_3d) * 1000.0
                obj.centroid_3d = tuple((R @ p + t).tolist())
            if obj.grasp_point_3d is not None:
                p = np.array(obj.grasp_point_3d) * 1000.0
                obj.grasp_point_3d = tuple((R @ p + t).tolist())

    if workspace_bounds is not None and camera_extrinsics is not None:
        filtered = []
        rejected = []
        for obj in centroids:
            if obj.centroid_3d is None:
                rejected.append((obj.object_id, "no_3d"))
                continue
            X, Y, Z = obj.centroid_3d
            in_x = workspace_bounds["x_min"] <= X <= workspace_bounds["x_max"]
            in_y = workspace_bounds["y_min"] <= Y <= workspace_bounds["y_max"]
            in_z = workspace_bounds["z_min"] <= Z <= workspace_bounds["z_max"]
            if in_x and in_y and in_z:
                filtered.append(obj)
            else:
                reason = f"X={X:.0f}" if not in_x else (f"Y={Y:.0f}" if not in_y else f"Z={Z:.0f}")
                rejected.append((obj.object_id, reason))
        print(f"  Workspace filter: {len(filtered)} kept, {len(rejected)} rejected")
        for obj_id, reason in rejected:
            print(f"    rejected obj #{obj_id}: {reason} out of bounds")
        centroids = filtered

    vis_img_centroids = visualize_masks_clean(rgb_work, pred_masks, centroids, alpha=0.5)
    if draw_bbox:
        for obj in centroids:
            x1, y1, x2, y2 = [int(round(v)) for v in obj.bbox]
            cv2.rectangle(vis_img_centroids, (x1, y1), (x2, y2), (255, 255, 0), 1)

    return {
        "centroids": centroids,
        "masks": pred_masks,
        "visible_masks": pred_visible_masks,
        "boxes": pred_boxes,
        "occlusions": pred_occlusions,
        "scores": scores,
        "vis_image": vis_img_centroids,
        "rgb_resized": rgb_work,
        "depth_raw_resized": depth_raw_resized,
    }


def get_parser():
    p = argparse.ArgumentParser(
        description="Grasp / centroid computation from instance masks (PNG per object)."
    )
    p.add_argument(
        "--dataset-path",
        type=str,
        default="./sample_data",
        help="Folder with image_color/ or loose images; paired masks in masks/<stem>/",
    )
    p.add_argument(
        "--image-path",
        type=str,
        default=None,
        help="Single RGB image (needs --amodal-masks-dir and/or --simple-masks)",
    )
    p.add_argument(
        "--depth-path",
        type=str,
        default=None,
        help="16-bit or float depth image matching RGB size",
    )
    p.add_argument(
        "--amodal-masks-dir",
        type=str,
        default=None,
        help="Directory of binary mask images, one per instance (sorted by name)",
    )
    p.add_argument(
        "--visible-masks-dir",
        type=str,
        default=None,
        help="Optional; defaults to amodal masks if omitted",
    )
    p.add_argument(
        "--width",
        type=int,
        default=None,
        help="Resize RGB/depth/masks to this width (optional)",
    )
    p.add_argument(
        "--height",
        type=int,
        default=None,
        help="Resize RGB/depth/masks to this height (optional)",
    )
    p.add_argument("--output-dir", type=str, default="./output_centroids")
    p.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Process only the first N images (batch mode; order is sorted by filename)",
    )
    p.add_argument(
        "--simple-masks",
        action="store_true",
        help="Classic CV instance masks (adaptive threshold + CC), same as pipeline_from_raw_image",
    )
    p.add_argument("--use-dummy-depth", action="store_true")
    p.add_argument("--save-json", action="store_true")
    p.add_argument(
        "--centroid-method",
        type=str,
        default="suction",
        choices=[
            "suction",
            "adaptive",
            "skeleton_dt",
            "top_center",
            "distance_transform",
            "ellipse",
            "circle",
            "mask_bbox",
            "bbox",
            "moments",
            "median",
        ],
    )
    p.add_argument("--suction-cup-radius", type=int, default=15)
    p.add_argument(
        "--use-visible-mask",
        action="store_true",
        help="Use visible mask channel for geometry (if you supply separate visible masks)",
    )
    p.add_argument("--draw-bbox", action="store_true")
    p.add_argument("--fx", type=float, default=1349.4442138671875)
    p.add_argument("--fy", type=float, default=1350.024658203125)
    p.add_argument("--cx", type=float, default=988.072021484375)
    p.add_argument("--cy", type=float, default=559.8935546875)
    p.add_argument("--camera-json", type=str, default=None)
    p.add_argument("--extrinsics-json", type=str, default=None)
    p.add_argument("--workspace-filter", action="store_true", default=False)
    p.add_argument("--aruco-config", type=str, default=None)
    p.add_argument("--workspace-z-min", type=float, default=2300.0)
    p.add_argument("--workspace-z-max", type=float, default=2700.0)
    return p


def print_centroids_summary(centroids: List, image_name: str = ""):
    print(f"\n{'='*60}")
    print(f"CENTROID RESULTS: {image_name}")
    print(f"{'='*60}")
    print(f"Objects: {len(centroids)}\n")
    for obj in centroids:
        print(f"Object #{obj.object_id}:")
        print(f"  - Geometric Center:  ({obj.centroid_amodal[0]:.1f}, {obj.centroid_amodal[1]:.1f}) px")
        if obj.grasp_point is not None:
            print(f"  - Suction Grasp Pt:  ({obj.grasp_point[0]:.1f}, {obj.grasp_point[1]:.1f}) px")
        if obj.centroid_3d is not None:
            X, Y, Z = obj.centroid_3d
            if abs(Z) > 10:
                print(f"  - 3D Center (world): X={X:.1f}mm, Y={Y:.1f}mm, Z={Z:.1f}mm")
            else:
                print(f"  - 3D Center (camera): X={X:.3f}m, Y={Y:.3f}m, Z={Z:.3f}m")
        if obj.grasp_point_3d is not None and obj.grasp_point_3d != obj.centroid_3d:
            X, Y, Z = obj.grasp_point_3d
            if abs(Z) > 10:
                print(f"  - 3D Grasp Pt (world): X={X:.1f}mm, Y={Y:.1f}mm, Z={Z:.1f}mm")
            else:
                print(f"  - 3D Grasp Pt (camera): X={X:.3f}m, Y={Y:.3f}m, Z={Z:.3f}m")
        print(f"  - Occluded: {obj.is_occluded}")
        print(f"  - Confidence: {obj.confidence:.2f}")
        print()
    best = get_best_grasp_point(centroids)
    if best is not None:
        print(f"RECOMMENDED GRASP: Object #{best.object_id}")
        grasp = best.grasp_point if best.grasp_point else best.centroid_amodal
        print(f"  Grasp at pixel: ({grasp[0]:.1f}, {grasp[1]:.1f})")
    print(f"{'='*60}\n")


def main():
    args = get_parser().parse_args()

    if args.camera_json and os.path.exists(args.camera_json):
        with open(args.camera_json) as f:
            cam = json.load(f)
        raw_intrinsics = {
            "fx": cam["fx"],
            "fy": cam["fy"],
            "cx": cam["cx"],
            "cy": cam["cy"],
            "orig_w": cam.get("width", 1920),
            "orig_h": cam.get("height", 1080),
        }
    else:
        raw_intrinsics = {
            "fx": args.fx,
            "fy": args.fy,
            "cx": args.cx,
            "cy": args.cy,
            "orig_w": 1920,
            "orig_h": 1080,
        }

    camera_extrinsics = None
    if args.extrinsics_json and os.path.exists(args.extrinsics_json):
        with open(args.extrinsics_json) as f:
            ext = json.load(f)
        ext_data = next(iter(ext.values()))
        camera_extrinsics = {
            "R": np.array(ext_data["R"], dtype=np.float64),
            "t": np.array(ext_data["t"], dtype=np.float64),
        }

    workspace_bounds = None
    if args.workspace_filter:
        if camera_extrinsics is None:
            print("Warning: --workspace-filter needs --extrinsics-json; disabled.")
        else:
            ws_w, ws_h = 850.0, 480.0
            if args.aruco_config and os.path.exists(args.aruco_config):
                with open(args.aruco_config) as f:
                    aruco = json.load(f)
                ws_w = aruco.get("workspace_width_mm", ws_w)
                ws_h = aruco.get("workspace_height_mm", ws_h)
            workspace_bounds = {
                "x_min": -ws_w / 2,
                "x_max": ws_w / 2,
                "y_min": -ws_h / 2,
                "y_max": ws_h / 2,
                "z_min": args.workspace_z_min,
                "z_max": args.workspace_z_max,
            }

    os.makedirs(args.output_dir, exist_ok=True)

    if args.image_path:
        if not args.amodal_masks_dir and not args.simple_masks:
            raise SystemExit("--image-path requires --amodal-masks-dir and/or --simple-masks")
        rgb_paths = [args.image_path]
        depth_paths = [args.depth_path] if args.depth_path else [None]
        mask_dirs = [args.amodal_masks_dir]
        visible_dirs = [args.visible_masks_dir] if args.visible_masks_dir else [None]
    else:
        rgb_paths = sorted(
            glob.glob(f"{args.dataset_path}/image_color/*.png")
            + glob.glob(f"{args.dataset_path}/image_color/*.jpg")
            + glob.glob(f"{args.dataset_path}/*.png")
            + glob.glob(f"{args.dataset_path}/*.jpg")
        )
        depth_paths = sorted(glob.glob(f"{args.dataset_path}/disparity/*.png"))
        mask_dirs = []
        visible_dirs = []

        if not rgb_paths:
            raise SystemExit(f"No images under {args.dataset_path}")

        if args.max_images is not None:
            rgb_paths = rgb_paths[: max(0, args.max_images)]

        if args.simple_masks:
            mask_dirs = [None] * len(rgb_paths)
            visible_dirs = [None] * len(rgb_paths)
        else:
            for rp in rgb_paths:
                md = resolve_mask_dir_for_image(args.dataset_path, rp)
                if md is None:
                    raise SystemExit(
                        f"No mask folder for {rp}. Expected {args.dataset_path}/masks/<stem>/ "
                        f"or amodal_masks/<stem>/ (or pass --simple-masks)"
                    )
                mask_dirs.append(md)
                base = os.path.splitext(os.path.basename(rp))[0]
                vd = None
                for sub in ("visible_masks", "masks_visible"):
                    d = os.path.join(args.dataset_path, sub, base)
                    if os.path.isdir(d):
                        vd = d
                        break
                visible_dirs.append(vd)

        if len(depth_paths) == 0:
            print("Warning: no disparity/*.png; using dummy depth.")
            args.use_dummy_depth = True

    for idx, rgb_path in enumerate(rgb_paths):
        print(f"\n[{idx+1}/{len(rgb_paths)}] {rgb_path}")
        rgb_img = cv2.imread(rgb_path)
        if rgb_img is None:
            print(f"Skip (unreadable): {rgb_path}")
            continue

        if args.width and args.height:
            rgb_img = cv2.resize(rgb_img, (args.width, args.height))
        h, w = rgb_img.shape[:2]

        scale_x = w / raw_intrinsics["orig_w"]
        scale_y = h / raw_intrinsics["orig_h"]
        camera_intrinsics = {
            "fx": raw_intrinsics["fx"] * scale_x,
            "fy": raw_intrinsics["fy"] * scale_y,
            "cx": raw_intrinsics["cx"] * scale_x,
            "cy": raw_intrinsics["cy"] * scale_y,
        }

        if args.use_dummy_depth:
            depth_img_raw = np.ones((h, w), dtype=np.float32) * 800.0
        elif args.image_path:
            dp = args.depth_path
            if dp and os.path.isfile(dp):
                depth_img_raw = cv2.imread(dp, cv2.IMREAD_UNCHANGED)
                depth_img_raw = (
                    depth_img_raw.astype(np.float32)
                    if depth_img_raw is not None
                    else np.ones((h, w), dtype=np.float32) * 800.0
                )
            else:
                depth_img_raw = np.ones((h, w), dtype=np.float32) * 800.0
        elif idx < len(depth_paths):
            depth_img_raw = cv2.imread(depth_paths[idx], cv2.IMREAD_UNCHANGED)
            depth_img_raw = (
                depth_img_raw.astype(np.float32)
                if depth_img_raw is not None
                else np.ones((h, w), dtype=np.float32) * 800.0
            )
        else:
            depth_img_raw = np.ones((h, w), dtype=np.float32) * 800.0
        if depth_img_raw.shape[:2] != (h, w):
            depth_img_raw = cv2.resize(
                depth_img_raw, (w, h), interpolation=cv2.INTER_NEAREST
            )

        if args.simple_masks:
            print("  --simple-masks: running classic CV segmentation…")
            mask_list = segment_objects(rgb_img)
            if not mask_list:
                print("  Skip: no instances found.")
                continue
            amodal = np.stack([m.astype(np.float32) for m in mask_list], axis=0)
            visible = amodal.copy()
        else:
            amodal = load_binary_masks(mask_dirs[idx], w, h)
            if visible_dirs[idx]:
                visible = load_binary_masks(visible_dirs[idx], w, h)
                if visible.shape[0] != amodal.shape[0]:
                    raise ValueError("Visible mask count must match amodal mask count")
            else:
                visible = amodal.copy()

        use_amodal = not args.use_visible_mask

        results = process_from_masks(
            rgb_img=rgb_img,
            depth_img_raw=depth_img_raw,
            pred_masks=amodal,
            pred_visible_masks=visible,
            centroid_method=args.centroid_method,
            camera_intrinsics=camera_intrinsics,
            camera_extrinsics=camera_extrinsics,
            use_amodal=use_amodal,
            draw_bbox=args.draw_bbox,
            suction_cup_radius=args.suction_cup_radius,
            workspace_bounds=workspace_bounds,
        )

        base_name = os.path.splitext(os.path.basename(rgb_path))[0]
        print_centroids_summary(results["centroids"], base_name)

        vis_combined = np.hstack([results["rgb_resized"], results["vis_image"]])
        out_vis = os.path.join(args.output_dir, f"{base_name}_centroids.png")
        cv2.imwrite(out_vis, vis_combined)
        print(f"Saved: {out_vis}")

        if args.save_json:
            data = {
                "image_path": rgb_path,
                "mask_dir": mask_dirs[idx] if not args.simple_masks else "simple_cv",
                "num_objects": len(results["centroids"]),
                "centroids": centroids_to_dict(results["centroids"]),
                "camera_intrinsics": camera_intrinsics,
                "coordinate_frame": "world_mm" if camera_extrinsics is not None else "camera_m",
            }
            with open(
                os.path.join(args.output_dir, f"{base_name}_centroids.json"), "w"
            ) as f:
                json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
