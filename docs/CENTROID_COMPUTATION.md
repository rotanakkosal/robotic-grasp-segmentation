# Centroid Computation for Robot Picking

This document describes the centroid computation methods implemented for finding optimal grasp points in the UOAIS instance segmentation pipeline, specifically designed for vacuum suction cup grippers.

## Table of Contents

1. [Overview](#overview)
2. [Literature Review](#literature-review)
3. [Available Methods](#available-methods)
4. [Research-Based Suction Method](#research-based-suction-method)
5. [Usage](#usage)
6. [API Reference](#api-reference)

---

## Overview

After performing instance segmentation with UOAIS, we need to find the optimal point on each object for the robot to pick. For vacuum suction cup grippers, this point must be:

1. **On a flat surface** - to form a proper seal
2. **Far from edges** - to ensure full contact with the suction cup
3. **Near the object center** - for stable lifting

This module provides multiple methods for computing grasp points, ranging from simple geometric approaches to research-backed algorithms.

---

## Literature Review

### Key Papers

| Paper | Year | Key Contribution |
|-------|------|------------------|
| [Dex-Net 3.0](https://arxiv.org/abs/1709.06670) | ICRA 2018 | Compliant suction contact model, seal formation analysis |
| [Learning Suction Graspability](https://pmc.ncbi.nlm.nih.gov/articles/PMC8987443/) | Frontiers 2022 | Grasp quality formula using surface normal variance |
| [Amazon Robotics Challenge](https://journals.sagepub.com/doi/10.1177/0278364919868017) | 2017-2019 | Centroid + surface normal combination approach |

### Key Findings

#### 1. What Makes a Good Suction Grasp Point?

From **Dex-Net 3.0** (Mahler et al., ICRA 2018):
- **Seal Formation**: The suction cup must form a tight seal with the surface
- **Wrench Resistance**: The grasp must resist gravity and disturbances
- **Surface Curvature**: Flat surfaces near the object center enable better seal formation

From **Learning Suction Graspability** (Frontiers in Neurorobotics, 2022):
- **Grasp Quality Formula**:
  ```
  Js = 0.9 × var(ns) + 0.1 × e^(-5 × res(ps))
  ```
  Where:
  - `var(ns)` = variance of surface normals (lower = flatter = better)
  - `res(ps)` = residual error of plane fitting (smoothness measure)

- **Distance to Center**: Grasp points close to the area center are more stable

#### 2. Amazon Robotics Challenge Approaches

- **Team MIT (2015, 2nd place)**: Suctioning on the centroid of objects with flat surfaces
- **Team Delft (2016, winner)**: Approaching the estimated object centroid along the inward surface normal

#### 3. Surface Normal Analysis

From the research, the key insight is:
- **Low variance of surface normals = flat surface = good for suction**
- Surface normals can be computed from depth images using gradient methods
- The optimal point combines flatness, edge clearance, and center proximity

---

## Available Methods

| Method | Description | Best For | Requires Depth |
|--------|-------------|----------|----------------|
| `suction` | Surface-normal based optimal point | **Vacuum suction grippers** | Yes (falls back to distance_transform) |
| `top_center` | Center of top 20% region | Picking lids/caps from above | No |
| `distance_transform` | Point furthest from all edges | General picking, stable point | No |
| `ellipse` | Center of fitted ellipse | Elongated objects (bottles, tools) | No |
| `circle` | Center of minimum enclosing circle | Round/compact objects | No |
| `mask_bbox` | Center of mask's bounding box | General purpose | No |
| `bbox` | Center of predicted bounding box | Fast approximation | No |
| `moments` | Center of mass | Uniform density objects | No |
| `median` | Median of pixel coordinates | Noisy/irregular masks | No |

---

## Research-Based Suction Method

The `suction` method implements a research-backed algorithm for finding optimal vacuum suction grasp points.

### Algorithm

```
1. Identify TOP SURFACE using depth (for top-down picking)
   - Find the region closest to camera (smallest depth values)
   - Use depth threshold: min_depth + 20% of depth_range
   - This isolates the lid/cap area on bottles and containers
   - If depth is unavailable/uniform, fall back to geometric top (y-coordinate)

2. Compute surface normals from depth image
   - Using Sobel gradients: normal = (-dz/dx, -dz/dy, 1)
   - Normalize to unit vectors

3. Compute local normal variance (flatness measure)
   - Lower variance = flatter surface = better for suction
   - Formula: var(ns) = Σ||ns,i - n̄s||² / (N-1)

4. Compute distance transform (edge clearance)
   - Points far from edges have more contact area for suction cup

5. Compute center proximity (within TOP SURFACE region)
   - Points near region center are more stable for lifting

6. Combine scores:
   Score = 0.4 × Flatness + 0.3 × EdgeClearance + 0.3 × CenterProximity

7. Select point with highest score within TOP SURFACE
```

### Key Improvement: Depth-Based Top Surface Detection

For objects like bottles, the algorithm now:
1. Uses depth data to identify the **top surface** (closest to camera)
2. Only considers points on this top surface for suction
3. Falls back to geometric top (based on y-coordinate) if depth is unavailable

### Visual Explanation

```
Surface Normal Variance (Flatness):

Flat surface (good):          Curved surface (bad):
    ↑ ↑ ↑ ↑ ↑                     ↑ ↗ → ↘ ↓
    ↑ ↑ ↑ ↑ ↑                   ↖ ↑ ↗ → ↘
    ↑ ↑ ↑ ↑ ↑                   ← ↖ ↑ ↗ →

Low variance = all              High variance = normals
normals point same way          point different directions


Distance Transform (Edge Clearance):

    ████████████
    ██1 2 3 2 1██
    ██2 3 4 3 2██    ← The "4" is furthest from edges
    ██1 2 3 2 1██       = best edge clearance
    ████████████
```

### Implementation

```python
def compute_suction_grasp_point(
    mask: np.ndarray,
    depth_image: np.ndarray,
    suction_cup_radius: int = 15,
    prefer_center: bool = True,
    center_weight: float = 0.3
) -> Optional[Tuple[float, float]]:
    """
    Compute optimal suction grasp point using surface normal analysis.

    Based on research from:
    - Dex-Net 3.0 (Mahler et al., ICRA 2018)
    - Learning Suction Graspability (Frontiers, 2022)
    - Amazon Robotics Challenge approaches
    """
```

---

## Usage

### Command Line

```bash
# Use the research-based suction method (default)
python tools/run_with_centroids.py \
    --image-path ./sample_data/image.png \
    --centroid-method suction \
    --suction-cup-radius 15 \
    --output-dir ./output_centroids

# Use top_center for picking bottle lids
python tools/run_with_centroids.py \
    --image-path ./sample_data/image.png \
    --centroid-method top_center \
    --output-dir ./output_centroids

# Use distance_transform (no depth required)
python tools/run_with_centroids.py \
    --image-path ./sample_data/image.png \
    --centroid-method distance_transform \
    --use-dummy-depth \
    --output-dir ./output_centroids
```

### Python API

```python
from tools.centroid_utils import (
    compute_all_centroids,
    compute_suction_grasp_point,
    compute_centroid_top_center,
    compute_centroid_distance_transform,
    draw_centroids
)

# After running UOAIS segmentation...
centroids = compute_all_centroids(
    pred_masks=pred_masks,           # [N, H, W] amodal masks
    pred_visible_masks=visible_masks, # [N, H, W] visible masks
    pred_boxes=pred_boxes,           # [N, 4] bounding boxes
    pred_occlusions=pred_occlusions, # [N] occlusion flags
    scores=scores,                   # [N] confidence scores
    depth_image=depth_raw,           # [H, W] raw depth in mm
    camera_intrinsics=intrinsics,    # dict with fx, fy, cx, cy
    method="suction",                # or "top_center", "distance_transform", etc.
    use_amodal_for_centroid=True,    # use complete object mask
    suction_cup_radius=15            # pixels
)

# Access centroid data
for obj in centroids:
    print(f"Object {obj.object_id}:")
    print(f"  Grasp point (2D): {obj.centroid_visible}")
    print(f"  Grasp point (3D): {obj.centroid_3d}")  # (X, Y, Z) in meters
    print(f"  Is occluded: {obj.is_occluded}")
```

---

## API Reference

### ObjectCentroid

```python
@dataclass
class ObjectCentroid:
    object_id: int                              # Object index
    centroid_amodal: Tuple[float, float]        # Center of complete object (px)
    centroid_visible: Tuple[float, float]       # Grasp point (px) - USE THIS
    centroid_bbox: Tuple[float, float]          # Bounding box center (px)
    centroid_3d: Optional[Tuple[float, float, float]]  # (X, Y, Z) in meters
    bbox: Optional[Tuple[int, int, int, int]]   # x1, y1, x2, y2
    is_occluded: bool                           # True if object is occluded
    confidence: float                           # Detection confidence
    mask_area: int                              # Total mask pixels
    visible_area: int                           # Visible mask pixels
```

### Key Functions

#### compute_suction_grasp_point()
```python
def compute_suction_grasp_point(
    mask: np.ndarray,
    depth_image: np.ndarray,
    suction_cup_radius: int = 15,
    top_k_candidates: int = 10,
    prefer_center: bool = True,
    center_weight: float = 0.3
) -> Optional[Tuple[float, float]]
```

Computes optimal suction point using surface normal analysis. Returns `(cx, cy)` pixel coordinates.

#### compute_centroid_top_center()
```python
def compute_centroid_top_center(
    mask: np.ndarray,
    top_percent: float = 20.0
) -> Optional[Tuple[float, float]]
```

Computes center of the top region of the object. Best for picking lids/caps from above.

#### compute_surface_normals()
```python
def compute_surface_normals(
    depth_image: np.ndarray
) -> np.ndarray
```

Computes surface normals from depth image. Returns `[H, W, 3]` array of unit normal vectors.

#### compute_normal_variance()
```python
def compute_normal_variance(
    normals: np.ndarray,
    mask: np.ndarray,
    window_size: int = 15
) -> np.ndarray
```

Computes local variance of surface normals. Lower variance = flatter surface.

---

## References

1. Mahler, J., Matl, M., et al. (2018). **Dex-Net 3.0: Computing Robust Vacuum Suction Grasp Targets in Point Clouds using a New Analytic Model and Deep Learning**. ICRA 2018. [arXiv:1709.06670](https://arxiv.org/abs/1709.06670)

2. Cao, H., et al. (2022). **Learning Suction Graspability Considering Grasp Quality and Robot Reachability for Bin-Picking**. Frontiers in Neurorobotics. [PMC8987443](https://pmc.ncbi.nlm.nih.gov/articles/PMC8987443/)

3. Zeng, A., Song, S., et al. (2022). **Robotic Pick-and-Place of Novel Objects in Clutter with Multi-Affordance Grasping and Cross-Domain Image Matching**. IJRR. [DOI:10.1177/0278364919868017](https://journals.sagepub.com/doi/10.1177/0278364919868017)

4. Badino, H., Huber, D., et al. **Fast and Accurate Computation of Surface Normals from Range Images**. ICRA 2011.

5. ten Pas, A., et al. (2017). **Grasp Pose Detection in Point Clouds**. IJRR. [DOI:10.1177/0278364917735594](https://journals.sagepub.com/doi/10.1177/0278364917735594)
