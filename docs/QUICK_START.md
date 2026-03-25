# Quick start: grasp points (centroids)

Two paths:

1. **Trained amodal segmentation (in this repo)** — run `tools/run_with_centroids.py` with the same environment as the main [README](../README.md) (PyTorch, Detectron2, `python setup.py build develop`, checkpoints under `output/` and `foreground_segmentation/rgbd_fg.pth` if using CG-Net).

2. **Your own masks** — call `compute_all_centroids` in `tools/centroid_utils.py` with `pred_masks` / depth from any source.

## 1) End-to-end with `run_with_centroids.py`

Install per main README, then example single image (dummy depth for 2D-only smoke test):

```bash
cd /path/to/this/repo

python tools/run_with_centroids.py \
    --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml \
    --image-path ./sample_data/arm-robot-Dataset/arm_robot_images/IMG_1761.png \
    --use-dummy-depth \
    --output-dir ./output_centroids
```

With real depth:

```bash
python tools/run_with_centroids.py \
    --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml \
    --image-path ./your.png \
    --depth-path ./your_depth.png \
    --camera-json ./sample_data/arm-robot-Dataset/custom_bop_bin_picking_dataset/camera.json \
    --save-json \
    --output-dir ./output_centroids
```

Batch OSD-style layout: `--dataset-path ./sample_data` with `image_color/` and `disparity/`.

Foreground filtering: add `--use-cgnet`.

## 2) Python API (masks you already have)

```python
import cv2
from tools.centroid_utils import compute_all_centroids, draw_centroids

centroids = compute_all_centroids(
    pred_masks=pred_masks,
    pred_visible_masks=pred_visible_masks,
    pred_boxes=pred_boxes,
    pred_occlusions=pred_occlusions,
    scores=scores,
    depth_image=depth_raw,
    method="suction",
)

vis_image = draw_centroids(rgb_image, centroids)
cv2.imwrite("output.png", vis_image)
```

## Method selection

| Use case | Method |
|----------|--------|
| Suction + depth | `suction` |
| No / dummy depth | `adaptive` or `distance_transform` |
| Lids from above | `top_center` |

See [CENTROID_COMPUTATION.md](CENTROID_COMPUTATION.md) for all options.

## Classic CV masks only

For threshold-based instances without the neural network, see `tools/pipeline_from_raw_image.py` (visualization pipeline); you can still pass resulting masks into `compute_all_centroids` in a short script if needed.
