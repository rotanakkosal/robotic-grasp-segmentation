# Bin-picking grasp utilities

Tools for computing **2D / 3D grasp points** (e.g. vacuum suction) from **instance masks** produced by any segmentation method you choose (classical CV, SAM, Mask R-CNN, etc.). The core logic lives in `tools/centroid_utils.py`; the CLI is `tools/run_with_centroids.py`.

## First version (v0.1) — not end-to-end

This release is intentionally **narrow**: masks in → grasp pixels (and optional camera-frame 3D with real depth + intrinsics) out, plus visualizations and JSON. There is **no** live camera loop, **no** robot driver, and **no** full bin-picking stack here. Treat [docs/PROJECT_PIPELINE.md](docs/PROJECT_PIPELINE.md) as a **design target**, not something this repo implements yet.

## Documentation

| Document | Contents |
|----------|----------|
| [docs/QUICK_START.md](docs/QUICK_START.md) | Install, mask layout, running the CLI |
| [docs/CENTROID_COMPUTATION.md](docs/CENTROID_COMPUTATION.md) | Grasp methods and API |
| [docs/PROJECT_PIPELINE.md](docs/PROJECT_PIPELINE.md) | Full pick-and-place pipeline design (segmentation → grasp) |
| [docs/LITERATURE_REVIEW.md](docs/LITERATURE_REVIEW.md) | Research background |

## Requirements

- Python ≥ 3.8  
- `numpy`, `opencv-python`  
- Optional: `matplotlib` (for `tools/debug_centroids.py` and some visualization scripts), `scikit-image` (for `walkthrough_frame10.py`)

```bash
pip install numpy opencv-python matplotlib scikit-image
# optional editable install metadata:
pip install -e .
```

## Quick start

### A. Sample folder (classic CV masks, no files to prepare)

Top-down phone photos live in `sample_data/arm-robot-Dataset/arm_robot_images/` (`IMG_*.png`).
Use **adaptive threshold + connected components** (same idea as `tools/pipeline_from_raw_image.py`):

```bash
python tools/run_with_centroids.py \
  --dataset-path ./sample_data/arm-robot-Dataset/arm_robot_images \
  --simple-masks \
  --use-dummy-depth \
  --centroid-method adaptive \
  --output-dir ./output_centroids
```

Optional: `--max-images 5` to process only the first five after sorting. With real depth, drop `--use-dummy-depth`, add `--depth-path` (single image) or `disparity/*.png` next to your RGB layout.

### B. Your own masks (one PNG per instance)

Sort filenames (e.g. `00.png`, `01.png`). Then:

```bash
python tools/run_with_centroids.py \
  --image-path ./your_scene.png \
  --amodal-masks-dir ./masks/your_scene/ \
  --depth-path ./your_depth.png \
  --save-json \
  --output-dir ./output_centroids
```

**Batch with masks:** RGB under `image_color/` or flat in the dataset root; masks under `masks/<image_stem>/` or `amodal_masks/<stem>/`.

## Repository layout

- `tools/run_with_centroids.py` — main CLI (masks → centroids / visualization / JSON)  
- `tools/centroid_utils.py` — grasp algorithms (`suction`, `adaptive`, distance transform, …)  
- `tools/debug_centroids.py` — debug plots per mask  
- `tools/pipeline_from_raw_image.py` — demo pipeline using **threshold segmentation** (no learned model)  
- `sample_data/` — example images / calibration JSON you can copy  

## License

See [LICENSE.md](./LICENSE.md). Third-party notices there still apply where relevant; the learned segmentation stack has been removed from this checkout.
