# Quick Start: Centroid Computation for Robot Picking

This guide helps you quickly get started with computing grasp points for vacuum suction cup grippers.

## Installation

No additional installation required. The centroid utilities use only numpy and OpenCV which are already dependencies of UOAIS.

## Basic Usage

### Command Line

```bash
# Navigate to UOAIS directory
cd /path/to/uoais

# Run with default settings (suction method)
python tools/run_with_centroids.py \
    --image-path ./sample_data/your_image.png \
    --output-dir ./output_centroids

# Without depth data (uses distance_transform fallback)
python tools/run_with_centroids.py \
    --image-path ./sample_data/your_image.png \
    --use-dummy-depth \
    --output-dir ./output_centroids
```

### Python API

```python
import cv2
import numpy as np
from tools.centroid_utils import compute_all_centroids, draw_centroids

# After UOAIS segmentation, you have:
# - pred_masks: [N, H, W] amodal masks
# - pred_visible_masks: [N, H, W] visible masks
# - pred_boxes: [N, 4] bounding boxes
# - pred_occlusions: [N] occlusion flags
# - scores: [N] confidence scores
# - depth_raw: [H, W] raw depth image (optional)

# Compute centroids
centroids = compute_all_centroids(
    pred_masks=pred_masks,
    pred_visible_masks=pred_visible_masks,
    pred_boxes=pred_boxes,
    pred_occlusions=pred_occlusions,
    scores=scores,
    depth_image=depth_raw,  # Can be None
    method="suction"        # or "top_center", "distance_transform"
)

# Get grasp points
for obj in centroids:
    # 2D pixel coordinates for grasp
    grasp_x, grasp_y = obj.centroid_visible

    # 3D coordinates in camera frame (if depth available)
    if obj.centroid_3d:
        X, Y, Z = obj.centroid_3d  # in meters

    print(f"Object {obj.object_id}: grasp at ({grasp_x:.1f}, {grasp_y:.1f})")

# Visualize
vis_image = draw_centroids(rgb_image, centroids)
cv2.imwrite("output.png", vis_image)
```

## Method Selection Guide

| Your Use Case | Recommended Method | Command |
|---------------|-------------------|---------|
| Vacuum suction with depth camera | `suction` | `--centroid-method suction` |
| Vacuum suction without depth | `distance_transform` | `--centroid-method distance_transform --use-dummy-depth` |
| Picking bottles by lid | `top_center` | `--centroid-method top_center` |
| Round objects | `circle` | `--centroid-method circle` |
| Elongated objects | `ellipse` | `--centroid-method ellipse` |

## Output Format

Each `ObjectCentroid` contains:

```python
ObjectCentroid(
    object_id=0,                      # Object index
    centroid_visible=(320.5, 245.2),  # USE THIS for grasping (pixels)
    centroid_amodal=(318.1, 243.8),   # Complete object center
    centroid_bbox=(315.0, 240.0),     # Bounding box center
    centroid_3d=(0.052, -0.018, 0.65),# 3D position (X, Y, Z) meters
    bbox=(200, 150, 430, 340),        # x1, y1, x2, y2
    is_occluded=False,                # True if partially hidden
    confidence=0.92,                  # Detection confidence
    mask_area=4521,                   # Total pixels
    visible_area=4200                 # Visible pixels
)
```

## Tips

1. **Use amodal mask** (default) - gives true object center even for occluded objects
2. **Adjust suction_cup_radius** - match your actual suction cup size in pixels
3. **Check is_occluded** - occluded objects may be harder to pick
4. **Use confidence threshold** - filter out low-confidence detections

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Centroid at edge of object | Increase `--suction-cup-radius` |
| No depth data | Use `--use-dummy-depth` (falls back to distance_transform) |
| Wrong center for bottles | Try `--centroid-method top_center` |
| Centroid outside mask | Check mask quality, try `--centroid-method mask_bbox` |

## Next Steps

- See [CENTROID_COMPUTATION.md](CENTROID_COMPUTATION.md) for detailed method descriptions
- See [LITERATURE_REVIEW.md](LITERATURE_REVIEW.md) for research background
- Integrate with your robot control system using the 3D coordinates
