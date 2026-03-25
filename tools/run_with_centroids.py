# Copyright (c) 2024. UOAIS inference with centroid computation for robot picking.
"""
UOAIS Inference with Centroid Computation

This script demonstrates how to:
1. Run UOAIS instance segmentation on RGB-D images
2. Compute object centroids (center points) for robot picking
3. Visualize results with centroid markers
4. Output 3D coordinates for robot control

Usage:
    python tools/run_with_centroids.py \
        --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml \
        --dataset-path ./sample_data \
        --output-dir ./output_centroids

    # With foreground segmentation filtering:
    python tools/run_with_centroids.py \
        --use-cgnet \
        --dataset-path ./sample_data

    # Single image:
    python tools/run_with_centroids.py \
        --image-path ./my_image.png \
        --depth-path ./my_depth.png
"""

import argparse
import glob
import os
import sys
import json
import cv2
import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adet.config import get_cfg
from adet.utils.visualizer import visualize_pred_amoda_occ, Visualizer
from adet.utils.post_process import detector_postprocess, DefaultPredictor

from utils import normalize_depth, inpaint_depth, standardize_image, array_to_tensor
from foreground_segmentation.model import Context_Guided_Network

# Import centroid utilities
from tools.centroid_utils import (
    compute_all_centroids,
    draw_centroids,
    get_best_grasp_point,
    centroids_to_dict,
    ObjectCentroid
)


def visualize_masks_clean(rgb_img, masks, centroids, alpha=0.5):
    """
    Clean visualization that accurately shows mask boundaries and centroids.

    This avoids the visual illusion caused by thick borders in the original
    visualize_pred_amoda_occ function.

    Args:
        rgb_img: RGB image [H, W, 3]
        masks: Amodal masks [N, H, W]
        centroids: List of ObjectCentroid objects
        alpha: Mask transparency (0-1)

    Returns:
        Visualization image with masks and centroids
    """
    vis_img = rgb_img.copy()

    # Generate distinct colors for each instance
    np.random.seed(42)  # For reproducibility
    colors = []
    for i in range(len(masks)):
        # Use HSV color space for better distribution
        hue = (i * 0.618033988749895) % 1.0  # Golden ratio for good distribution
        color = np.array(cv2.cvtColor(
            np.uint8([[[hue * 180, 200, 200]]]),
            cv2.COLOR_HSV2BGR
        )[0, 0], dtype=np.float32)
        colors.append(color)

    # Draw each mask with its color
    for i, mask in enumerate(masks):
        mask_bool = mask > 0
        color = colors[i]

        # Blend mask color with original image
        vis_img[mask_bool] = (
            vis_img[mask_bool] * (1 - alpha) +
            color * alpha
        ).astype(np.uint8)

        # Draw thin contour (1-2 pixels) to show exact boundary
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis_img, contours, -1, (255, 255, 255), 1)

    # Draw centroids: RED = geometric center, GREEN = suction grasp point
    vis_img = draw_centroids(
        vis_img,
        centroids,
        draw_amodal=True,       # Draw geometric center (red dot)
        draw_grasp=True,        # Draw suction grasp point (green cross)
        draw_bbox_center=False,
        draw_bbox=False,
        draw_labels=True
    )

    return vis_img


def get_parser():
    parser = argparse.ArgumentParser(
        description="UOAIS Instance Segmentation with Centroid Computation for Robot Picking"
    )
    parser.add_argument(
        "--config-file",
        default="configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--use-cgnet",
        action="store_true",
        help="Use foreground segmentation model to filter out background instances"
    )
    parser.add_argument(
        "--cgnet-weight-path",
        type=str,
        default="./foreground_segmentation/rgbd_fg.pth",
        help="path to foreground segmentation weight"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="./sample_data",
        help="path to the dataset (with image_color/ and disparity/ subdirs)"
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default=None,
        help="path to a single RGB image file (overrides --dataset-path)"
    )
    parser.add_argument(
        "--depth-path",
        type=str,
        default=None,
        help="path to corresponding depth image (used with --image-path)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output_centroids",
        help="directory to save visualization results and centroid data"
    )
    parser.add_argument(
        "--use-dummy-depth",
        action="store_true",
        help="Use dummy depth image if real depth unavailable"
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save centroid data as JSON files"
    )
    parser.add_argument(
        "--centroid-method",
        type=str,
        default="suction",
        choices=["suction", "adaptive", "skeleton_dt", "top_center", "distance_transform", "ellipse", "circle", "mask_bbox", "bbox", "moments", "median"],
        help="Method for computing centroids: suction=surface-normal based optimal point (BEST for vacuum grippers with depth, based on Dex-Net 3.0), adaptive=shape-adaptive auto-selection (BEST without depth, based on Cartman/ICRoM 2023), skeleton_dt=medial axis skeleton + distance transform (for complex shapes), top_center=center of top region (for picking lids), distance_transform=point furthest from edges, ellipse=fitted ellipse center, circle=enclosing circle center, mask_bbox=mask extent center, bbox=predicted box center, moments=center of mass, median=robust to outliers"
    )
    parser.add_argument(
        "--suction-cup-radius",
        type=int,
        default=15,
        help="Radius of suction cup in pixels (used for 'suction' method to check contact area)"
    )
    parser.add_argument(
        "--use-visible-mask",
        action="store_true",
        help="Use visible mask instead of amodal mask for centroid computation (default: use amodal for true object center)"
    )
    parser.add_argument(
        "--draw-bbox",
        action="store_true",
        help="Draw bounding boxes around objects (default: off, only centroids shown)"
    )
    parser.add_argument(
        "--clean-vis",
        action="store_true",
        help="Use clean visualization (thin borders, accurate mask fill) instead of default thick-border style"
    )
    # Camera intrinsics for 3D computation (defaults: RealSense L515 at 1920x1080)
    parser.add_argument("--fx", type=float, default=1349.4442138671875, help="Camera focal length x")
    parser.add_argument("--fy", type=float, default=1350.024658203125,  help="Camera focal length y")
    parser.add_argument("--cx", type=float, default=988.072021484375,   help="Camera principal point x")
    parser.add_argument("--cy", type=float, default=559.8935546875,     help="Camera principal point y")
    parser.add_argument("--camera-json", type=str, default=None,
                        help="Path to camera.json (overrides --fx/fy/cx/cy if provided)")
    parser.add_argument("--extrinsics-json", type=str, default=None,
                        help="Path to camera_extrinsics.json for camera→world transform")

    return parser


def process_single_image(
    rgb_img: np.ndarray,
    depth_img_raw: np.ndarray,
    predictor,
    cfg,
    fg_model=None,
    centroid_method: str = "suction",
    camera_intrinsics: dict = None,
    camera_extrinsics: dict = None,
    use_amodal: bool = True,
    draw_bbox: bool = False,
    clean_vis: bool = False,
    suction_cup_radius: int = 15
):
    """
    Process a single image and return segmentation results with centroids.

    Args:
        rgb_img: RGB image [H, W, 3] (BGR format)
        depth_img_raw: Raw depth image [H, W] in millimeters
        predictor: UOAIS predictor
        cfg: Detectron2 config
        fg_model: Optional CG-Net foreground segmentation model
        centroid_method: Method for centroid computation
        camera_intrinsics: Camera parameters for 3D projection
        use_amodal: If True, use amodal mask for centroid (true object center)
        draw_bbox: If True, draw bounding boxes on visualization
        clean_vis: If True, use clean visualization with thin borders
        suction_cup_radius: Radius of suction cup in pixels (for 'suction' method)

    Returns:
        dict with keys: 'centroids', 'masks', 'visible_masks', 'boxes', 'occlusions', 'vis_image'
    """
    W, H = cfg.INPUT.IMG_SIZE

    # Resize RGB
    rgb_resized = cv2.resize(rgb_img, (W, H))

    # Normalize and prepare depth
    depth_normalized = normalize_depth(depth_img_raw.copy())
    depth_normalized = cv2.resize(depth_normalized, (W, H), interpolation=cv2.INTER_NEAREST)
    depth_normalized = inpaint_depth(depth_normalized)

    # Also resize raw depth for 3D computation
    depth_raw_resized = cv2.resize(depth_img_raw, (W, H), interpolation=cv2.INTER_NEAREST)

    # Prepare UOAIS input
    if cfg.INPUT.DEPTH and cfg.INPUT.DEPTH_ONLY:
        uoais_input = depth_normalized
    elif cfg.INPUT.DEPTH and not cfg.INPUT.DEPTH_ONLY:
        uoais_input = np.concatenate([rgb_resized, depth_normalized], -1)
    else:
        uoais_input = rgb_resized

    # Run UOAIS inference
    outputs = predictor(uoais_input)
    instances = detector_postprocess(outputs['instances'], H, W).to('cpu')

    # Extract predictions
    pred_masks = instances.pred_masks.detach().cpu().numpy()
    pred_visible_masks = instances.pred_visible_masks.detach().cpu().numpy()
    pred_boxes = instances.pred_boxes.tensor.detach().cpu().numpy()
    pred_occlusions = instances.pred_occlusions.detach().cpu().numpy()
    scores = instances.scores.detach().cpu().numpy() if hasattr(instances, 'scores') else np.ones(len(pred_masks))

    # CG-Net foreground filtering
    if fg_model is not None:
        fg_rgb_input = standardize_image(cv2.resize(rgb_resized, (320, 240)))
        fg_rgb_input = array_to_tensor(fg_rgb_input).unsqueeze(0)
        fg_depth_input = cv2.resize(depth_normalized, (320, 240))
        fg_depth_input = array_to_tensor(fg_depth_input[:, :, 0:1]).unsqueeze(0) / 255
        fg_input = torch.cat([fg_rgb_input, fg_depth_input], 1)
        fg_output = fg_model(fg_input.cuda())
        fg_output = fg_output.cpu().data[0].numpy().transpose(1, 2, 0)
        fg_output = np.asarray(np.argmax(fg_output, axis=2), dtype=np.uint8)
        fg_output = cv2.resize(fg_output, (W, H), interpolation=cv2.INTER_NEAREST)

        # Filter out background instances
        remove_idxs = []
        for i, pred_visible in enumerate(pred_visible_masks):
            iou = np.sum(np.bitwise_and(pred_visible, fg_output)) / (np.sum(pred_visible) + 1e-6)
            if iou < 0.5:
                remove_idxs.append(i)

        if remove_idxs:
            pred_masks = np.delete(pred_masks, remove_idxs, 0)
            pred_visible_masks = np.delete(pred_visible_masks, remove_idxs, 0)
            pred_boxes = np.delete(pred_boxes, remove_idxs, 0)
            pred_occlusions = np.delete(pred_occlusions, remove_idxs, 0)
            scores = np.delete(scores, remove_idxs, 0)

    # Compute centroids for all detected objects
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
        suction_cup_radius=suction_cup_radius
    )

    # Apply extrinsics: convert 3D camera frame → world frame (mm)
    if camera_extrinsics is not None:
        R = camera_extrinsics['R']
        t = camera_extrinsics['t']
        for obj in centroids:
            if obj.centroid_3d is not None:
                p = np.array(obj.centroid_3d) * 1000.0  # meters → mm
                p_world = R @ p + t
                obj.centroid_3d = tuple(p_world.tolist())
            if obj.grasp_point_3d is not None:
                p = np.array(obj.grasp_point_3d) * 1000.0
                p_world = R @ p + t
                obj.grasp_point_3d = tuple(p_world.tolist())

    # Create visualization
    if clean_vis:
        # Use clean visualization with thin borders (accurate mask boundaries)
        vis_img_centroids = visualize_masks_clean(rgb_resized, pred_masks, centroids, alpha=0.5)
    else:
        # Use original UOAIS visualization (thick borders)
        # Reorder for visualization (occluded objects first)
        if len(pred_occlusions) > 0:
            idx_shuf = np.concatenate((np.where(pred_occlusions == 1)[0], np.where(pred_occlusions == 0)[0]))
            pred_masks_vis = pred_masks[idx_shuf]
            pred_occs_vis = pred_occlusions[idx_shuf]
            pred_boxes_vis = pred_boxes[idx_shuf]
        else:
            pred_masks_vis = pred_masks
            pred_occs_vis = pred_occlusions
            pred_boxes_vis = pred_boxes

        # UOAIS visualization
        vis_img = visualize_pred_amoda_occ(rgb_resized, pred_masks_vis, pred_boxes_vis, pred_occs_vis)

        # Draw centroids on visualization
        vis_img_centroids = draw_centroids(
            vis_img,
            centroids,
            draw_amodal=True,
            draw_visible=True,
            draw_bbox_center=False,
            draw_bbox=draw_bbox,
            draw_labels=True
        )

    return {
        'centroids': centroids,
        'masks': pred_masks,
        'visible_masks': pred_visible_masks,
        'boxes': pred_boxes,
        'occlusions': pred_occlusions,
        'scores': scores,
        'vis_image': vis_img_centroids,
        'rgb_resized': rgb_resized,
        'depth_raw_resized': depth_raw_resized
    }


def print_centroids_summary(centroids: list, image_name: str = ""):
    """Print a summary of detected objects and their centroids."""
    print(f"\n{'='*60}")
    print(f"CENTROID RESULTS: {image_name}")
    print(f"{'='*60}")
    print(f"Detected {len(centroids)} objects\n")

    for obj in centroids:
        print(f"Object #{obj.object_id}:")
        print(f"  - Geometric Center:  ({obj.centroid_amodal[0]:.1f}, {obj.centroid_amodal[1]:.1f}) px")
        if obj.grasp_point is not None:
            print(f"  - Suction Grasp Pt:  ({obj.grasp_point[0]:.1f}, {obj.grasp_point[1]:.1f}) px")
        if obj.centroid_3d is not None:
            X, Y, Z = obj.centroid_3d
            # World frame values are in mm (large numbers); camera frame in meters (small)
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
        print(f"  - Visible/Total Area: {obj.visible_area}/{obj.mask_area} "
              f"({100*obj.visible_area/(obj.mask_area+1e-6):.1f}%)")
        print()

    # Recommend best grasp point
    best = get_best_grasp_point(centroids)
    if best is not None:
        print(f"RECOMMENDED GRASP: Object #{best.object_id}")
        grasp = best.grasp_point if best.grasp_point else best.centroid_amodal
        print(f"  Grasp at pixel: ({grasp[0]:.1f}, {grasp[1]:.1f})")
        if best.grasp_point_3d is not None:
            X, Y, Z = best.grasp_point_3d
            print(f"  3D coordinates: ({X:.3f}, {Y:.3f}, {Z:.3f}) meters")
    print(f"{'='*60}\n")


def main():
    args = get_parser().parse_args()

    # Setup UOAIS model
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = args.confidence_threshold
    predictor = DefaultPredictor(cfg)
    W, H = cfg.INPUT.IMG_SIZE

    # Setup CG-Net (optional foreground segmentation)
    fg_model = None
    if args.use_cgnet:
        print("Loading CG-Net foreground segmentation model...")
        checkpoint = torch.load(args.cgnet_weight_path)
        fg_model = Context_Guided_Network(classes=2, in_channel=4)
        fg_model.load_state_dict(checkpoint['model'])
        fg_model.cuda()
        fg_model.eval()

    # Camera intrinsics for 3D projection
    # Load from camera.json if provided, otherwise use CLI args
    if args.camera_json and os.path.exists(args.camera_json):
        with open(args.camera_json) as f:
            cam = json.load(f)
        raw_intrinsics = {
            'fx': cam['fx'], 'fy': cam['fy'],
            'cx': cam['cx'], 'cy': cam['cy'],
            'orig_w': cam.get('width', 1920),
            'orig_h': cam.get('height', 1080),
        }
        print(f"Loaded intrinsics from {args.camera_json}: fx={cam['fx']:.1f}, fy={cam['fy']:.1f}")
    else:
        raw_intrinsics = {
            'fx': args.fx, 'fy': args.fy,
            'cx': args.cx, 'cy': args.cy,
            'orig_w': 1920, 'orig_h': 1080,
        }

    # Scale intrinsics to match the model's resized resolution (e.g. 640x480)
    scale_x = W / raw_intrinsics['orig_w']
    scale_y = H / raw_intrinsics['orig_h']
    camera_intrinsics = {
        'fx': raw_intrinsics['fx'] * scale_x,
        'fy': raw_intrinsics['fy'] * scale_y,
        'cx': raw_intrinsics['cx'] * scale_x,
        'cy': raw_intrinsics['cy'] * scale_y,
    }
    print(f"Scaled intrinsics ({raw_intrinsics['orig_w']}x{raw_intrinsics['orig_h']} → {W}x{H}): "
          f"fx={camera_intrinsics['fx']:.1f}, cx={camera_intrinsics['cx']:.1f}")

    # Load extrinsics for camera→world transform (optional)
    camera_extrinsics = None
    if args.extrinsics_json and os.path.exists(args.extrinsics_json):
        with open(args.extrinsics_json) as f:
            ext = json.load(f)
        # Take the first entry (camera serial key)
        ext_data = next(iter(ext.values()))
        camera_extrinsics = {
            'R': np.array(ext_data['R'], dtype=np.float64),
            't': np.array(ext_data['t'], dtype=np.float64),
        }
        print(f"Loaded extrinsics from {args.extrinsics_json}: "
              f"t=[{ext_data['t'][0]:.1f}, {ext_data['t'][1]:.1f}, {ext_data['t'][2]:.1f}]mm")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect image paths
    if args.image_path is not None:
        if not os.path.exists(args.image_path):
            raise ValueError(f"Image not found: {args.image_path}")
        rgb_paths = [args.image_path]
        depth_paths = [args.depth_path] if args.depth_path else [None]
        print(f"Processing single image: {args.image_path}")
    else:
        # Load from dataset directory
        rgb_paths = sorted(
            glob.glob(f"{args.dataset_path}/image_color/*.png") +
            glob.glob(f"{args.dataset_path}/image_color/*.jpg") +
            glob.glob(f"{args.dataset_path}/*.png") +
            glob.glob(f"{args.dataset_path}/*.jpg")
        )
        depth_paths = sorted(glob.glob(f"{args.dataset_path}/disparity/*.png"))

        if len(rgb_paths) == 0:
            raise ValueError(f"No images found in {args.dataset_path}")

        if len(depth_paths) == 0:
            print("Warning: No depth images found. Using dummy depth.")
            args.use_dummy_depth = True

    # Process all images
    all_results = []

    for idx, rgb_path in enumerate(rgb_paths):
        print(f"\nProcessing [{idx+1}/{len(rgb_paths)}]: {rgb_path}")

        # Load RGB image
        rgb_img = cv2.imread(rgb_path)
        if rgb_img is None:
            print(f"Warning: Could not read {rgb_path}")
            continue

        # Load depth image
        if args.use_dummy_depth or (idx >= len(depth_paths)) or depth_paths[idx] is None:
            depth_img_raw = np.ones((rgb_img.shape[0], rgb_img.shape[1]), dtype=np.float32) * 800.0
        else:
            # Use cv2.IMREAD_UNCHANGED to preserve 16-bit depth values
            depth_img_raw = cv2.imread(depth_paths[idx], cv2.IMREAD_UNCHANGED).astype(np.float32)

        # Process image
        results = process_single_image(
            rgb_img=rgb_img,
            depth_img_raw=depth_img_raw,
            predictor=predictor,
            cfg=cfg,
            fg_model=fg_model,
            centroid_method=args.centroid_method,
            camera_intrinsics=camera_intrinsics,
            camera_extrinsics=camera_extrinsics,
            use_amodal=not args.use_visible_mask,
            draw_bbox=args.draw_bbox,
            clean_vis=args.clean_vis,
            suction_cup_radius=args.suction_cup_radius
        )

        # Print summary
        base_name = os.path.splitext(os.path.basename(rgb_path))[0]
        print_centroids_summary(results['centroids'], base_name)

        # Save visualization
        output_vis_path = os.path.join(args.output_dir, f"{base_name}_centroids.png")

        # Create combined visualization: RGB | Segmentation+Centroids
        vis_combined = np.hstack([results['rgb_resized'], results['vis_image']])
        cv2.imwrite(output_vis_path, vis_combined)
        print(f"Saved visualization: {output_vis_path}")

        # Save JSON data
        if args.save_json:
            json_data = {
                'image_path': rgb_path,
                'num_objects': len(results['centroids']),
                'centroids': centroids_to_dict(results['centroids']),
                'camera_intrinsics': camera_intrinsics,
                'coordinate_frame': 'world_mm' if camera_extrinsics is not None else 'camera_m',
            }
            json_path = os.path.join(args.output_dir, f"{base_name}_centroids.json")
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            print(f"Saved JSON data: {json_path}")

        all_results.append({
            'image': base_name,
            'centroids': results['centroids']
        })

    print(f"\nProcessed {len(all_results)} images. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
