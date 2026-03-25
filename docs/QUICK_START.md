# Quick start: grasp points from instance masks

This guide covers computing grasp points for vacuum suction cups **after** you already have per-object masks from your own segmenter.

## Installation

```bash
pip install numpy opencv-python
# optional
pip install matplotlib scikit-image
```

## Mask format

- One **binary mask image per object** in a folder (PNG or JPG). Nonzero pixels = object.
- Filenames are sorted lexically to define instance order (`00.png`, `01.png`, …).

Optional: separate visible masks in `--visible-masks-dir`, or under `visible_masks/<stem>/` in batch mode.

## Command line

### Sample dataset (no mask folders)

`sample_data/arm-robot-Dataset/arm_robot_images/` contains flat `IMG_*.png` top-down captures. Use built-in classic CV segmentation:

```bash
cd /path/to/this/repo

python tools/run_with_centroids.py \
    --dataset-path ./sample_data/arm-robot-Dataset/arm_robot_images \
    --simple-masks \
    --use-dummy-depth \
    --centroid-method adaptive \
    --output-dir ./output_centroids
```

Add `--max-images N` to only process the first *N* files (sorted by name). Prefer `adaptive` or `distance_transform` when using `--use-dummy-depth`.

### With mask folders

```bash
python tools/run_with_centroids.py \
    --image-path ./sample_data/your_image.png \
    --amodal-masks-dir ./masks/your_image/ \
    --output-dir ./output_centroids

python tools/run_with_centroids.py \
    --image-path ./sample_data/your_image.png \
    --amodal-masks-dir ./masks/your_image/ \
    --use-dummy-depth \
    --output-dir ./output_centroids
```

### Batch layout (your masks)

For each `image_color/foo.png`, provide masks in `masks/foo/*.png` or `amodal_masks/foo/*.png`. Or use `--simple-masks` on a folder of loose `*.png` / `*.jpg` images.

## Python API

```python
import cv2
import numpy as np
from tools.centroid_utils import compute_all_centroids, draw_centroids

# You supply (e.g. from your segmenter):
# pred_masks: [N, H, W]
# pred_visible_masks: [N, H, W]  (often same as amodal if you only have one mask type)
# pred_boxes: [N, 4]  (x1,y1,x2,y2)
# pred_occlusions: [N]  (0/1 if known, else zeros)
# scores: [N]  (confidence, or ones)
# depth_raw: [H, W] in mm (optional)

centroids = compute_all_centroids(
    pred_masks=pred_masks,
    pred_visible_masks=pred_visible_masks,
    pred_boxes=pred_boxes,
    pred_occlusions=pred_occlusions,
    scores=scores,
    depth_image=depth_raw,
    method="suction",
)

for obj in centroids:
    grasp_x, grasp_y = obj.centroid_visible
    if obj.centroid_3d:
        X, Y, Z = obj.centroid_3d

vis_image = draw_centroids(rgb_image, centroids)
cv2.imwrite("output.png", vis_image)
```

## Method selection

| Use case | Method | CLI |
|----------|--------|-----|
| Suction + depth camera | `suction` | `--centroid-method suction` |
| Suction, no depth | `distance_transform` or `adaptive` | `--centroid-method adaptive --use-dummy-depth` |
| Lids / caps from above | `top_center` | `--centroid-method top_center` |

## Next steps

- [CENTROID_COMPUTATION.md](CENTROID_COMPUTATION.md) — all methods and parameters  
- [PROJECT_PIPELINE.md](PROJECT_PIPELINE.md) — full robot pipeline  
- [LITERATURE_REVIEW.md](LITERATURE_REVIEW.md) — references  
