# Autonomous Pick-and-Place Pipeline: Deep Technical Documentation

**Project**: Suction-Based Autonomous Robot Arm for Pharmaceutical Container Picking
**Camera**: Intel RealSense L515 (production) / Phone photo (testing)
**Last Updated**: 2026-03-24

---

## Table of Contents

1. [Project Goal](#1-project-goal)
2. [Target Objects](#2-target-objects)
3. [Full Pipeline Overview](#3-full-pipeline-overview)
4. [Stage 1: Instance Segmentation (UOAIS)](#4-stage-1-instance-segmentation-uoais)
5. [Stage 2: Shape Classification](#5-stage-2-shape-classification)
6. [Stage 3: 2D Grasp Point — Adaptive Method](#6-stage-3-2d-grasp-point--adaptive-method)
7. [Stage 4: Safety Check & Blending](#7-stage-4-safety-check--blending)
8. [Stage 5: 3D Grasp Point (Real Depth)](#8-stage-5-3d-grasp-point-real-depth)
9. [Why Not Just Use One Method?](#9-why-not-just-use-one-method)
10. [Failure Cases](#10-failure-cases)
11. [Current Status](#11-current-status)
12. [Next Steps](#12-next-steps)

---

## 1. Project Goal

The goal is a **fully autonomous pick-and-place system**. The robot arm looks at a bin of
pharmaceutical bottles from a top-down camera, identifies each object, decides where to grab
it with a suction cup, picks it up, and places it somewhere else — without any human input.

```
Camera view (top-down)
         ↓
Detect and segment all objects (even occluded ones)
         ↓
Find the best suction grasp point on each object
         ↓
Convert 2D pixel → 3D world coordinates
         ↓
Robot arm picks each object and places it
         ↓
Repeat until bin is empty
```

---

## 2. Target Objects

The objects are **Korean pharmaceutical and medical containers**:

| Type          | Shape             | Challenge                          |
|---------------|-------------------|------------------------------------|
| Pill bottles  | Tall cylinder     | Elongated → DT bias toward cap end |
| Supplement jars | Short wide cylinder | Nearly circular from top        |
| Spray bottles | Tall with nozzle  | Irregular top, nozzle changes shape|
| Dark glass bottles | Small cylinder | Low contrast, hard segmentation  |
| Caps / lids   | Small round disk  | Very circular, small area          |

All objects are viewed **from above** (top-down camera). The suction cup grabs from the
**top surface** of each object.

Key challenges:
- Objects look very similar (mostly white, cylindrical)
- They overlap and occlude each other in the bin
- Different shapes need different grasp strategies
- Some objects may be lying on their side (limited top surface)

---

## 3. Full Pipeline Overview

```
INPUT IMAGE (top-down RGB photo)
        │
        ▼
┌───────────────────────────────────────────────────────┐
│ STAGE 1: UOAIS Instance Segmentation                  │
│  → Per-object binary masks (amodal: includes occluded)│
│  → Bounding boxes, confidence scores                  │
└──────────────────────────┬────────────────────────────┘
                           │  N masks [H, W]
                           ▼
┌───────────────────────────────────────────────────────┐
│ STAGE 2: Shape Classification (per mask)              │
│  → Circularity  = 4π×area / perimeter²                │
│  → Aspect Ratio = long side / short side              │
│  → Solidity     = area / convex hull area             │
│  → Result: CIRCULAR / ELONGATED / COMPACT / CONCAVE   │
└──────────────────────────┬────────────────────────────┘
                           │  shape type per object
                           ▼
┌───────────────────────────────────────────────────────┐
│ STAGE 3: 2D Grasp Point (Adaptive Method)             │
│  Circular  → Distance Transform (DT center)           │
│  Others    → Moments (center of mass)                 │
└──────────────────────────┬────────────────────────────┘
                           │  candidate point (cx, cy)
                           ▼
┌───────────────────────────────────────────────────────┐
│ STAGE 4: Safety Check & Blending                      │
│  Is point inside mask?                                │
│  Is edge clearance ≥ 20% of max_dt?                   │
│  If unsafe → dynamic blend toward DT safe point       │
└──────────────────────────┬────────────────────────────┘
                           │  safe 2D grasp point
                           ▼
┌───────────────────────────────────────────────────────┐
│ STAGE 5: 3D Grasp Point (requires real depth)         │
│  Top surface extraction (depth threshold 20%)         │
│  Surface normal computation (Sobel gradients)         │
│  Flatness scoring (normal variance)                   │
│  Combined score: 40% flatness + 30% edge + 30% center │
└──────────────────────────┬────────────────────────────┘
                           │  final 3D (X, Y, Z) meters
                           ▼
                    ROBOT ARM EXECUTION
                    Pick → Place → Repeat
```

---

## 4. Stage 1: Instance Segmentation (UOAIS)

**UOAIS** = Unseen Object Amodal Instance Segmentation.

The model takes an RGB image and outputs per-object masks. The key word is **amodal**:
even if Object A is partially hidden behind Object B, UOAIS still predicts the full
shape of Object A (including the occluded part).

This is critical for pharmaceutical bin-picking because bottles in a bin almost always
overlap each other.

### Output per object:
- `pred_masks[i]` — amodal binary mask [H, W] (full shape, includes occluded area)
- `pred_visible_masks[i]` — visible binary mask [H, W] (only the visible region)
- `pred_boxes[i]` — bounding box [x1, y1, x2, y2]
- `scores[i]` — detection confidence [0, 1]
- `pred_occlusions[i]` — is this object occluded by another?

### Why amodal matters for grasping:

If we only use the visible mask, the centroid of an occluded bottle shifts toward
the visible portion, away from the true object center. The robot would then grab
off-center and risk dropping the object.

```
Occluded bottle (Object B hidden behind Object A):

Visible mask centroid:   ●  (shifted toward exposed side)
Amodal mask centroid:    ⊙  (true center of full object)

Robot should grab at: ⊙
```

---

## 5. Stage 2: Shape Classification

Before computing the grasp point, we analyze the shape of each object mask.
Three metrics are computed:

---

### 5.1 Circularity

**Formula:**
```
circularity = (4 × π × area) / perimeter²
```

**Range:** 0.0 (very irregular) → 1.0 (perfect circle)

**Intuition:** A circle is the most "efficient" shape — it encloses the maximum
area for a given perimeter. Any deviation from circular lowers this ratio.

| Shape               | Circularity |
|---------------------|-------------|
| Perfect circle      | 1.00        |
| Round bottle cap    | ~0.85       |
| Square box          | ~0.785      |
| Tall bottle         | ~0.30       |
| Star shape          | < 0.20      |

**Implementation** (`centroid_utils.py:80-87`):
```python
perimeter = cv2.arcLength(largest_contour, True)
area = cv2.contourArea(largest_contour)
circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
circularity = min(circularity, 1.0)
```

**Decision threshold:** `circularity > 0.7` → classified as circular → use **DT**

---

### 5.2 Aspect Ratio

**Formula:**
```
aspect_ratio = max(width, height) / min(width, height)
```
Always ≥ 1.0

**Intuition:** Measures how stretched the shape is. Uses a **rotated** bounding
rectangle (`cv2.minAreaRect`) so it works correctly for tilted objects.

| Shape               | Aspect Ratio |
|---------------------|--------------|
| Circle or square    | 1.0          |
| Slightly oval       | 1.3          |
| Standard bottle     | 2.0–3.5      |
| Pen / screwdriver   | 8.0+         |

**Implementation** (`centroid_utils.py:89-98`):
```python
_, (w, h), _ = cv2.minAreaRect(largest_contour)  # rotated bounding box
aspect_ratio = max(w, h) / min(w, h)
```

**Decision threshold:** `aspect_ratio > 2.0` → classified as elongated

---

### 5.3 Solidity

**Formula:**
```
solidity = area / convex_hull_area
```

**Intuition:** The convex hull is like stretching a rubber band around the shape.
Solidity measures how much of that "rubber band" area is actually filled by the
object. Low solidity means the shape has dents, holes, or concavities.

| Shape               | Solidity |
|---------------------|----------|
| Solid circle/square | ~1.0     |
| Bottle (solid)      | ~0.95    |
| C-shape             | ~0.75    |
| Star shape          | ~0.50    |
| Donut               | ~0.60    |

**Implementation** (`centroid_utils.py:100-106`):
```python
hull = cv2.convexHull(largest_contour)
hull_area = cv2.contourArea(hull)
solidity = area / hull_area
```

**Why it matters:** If solidity < 0.85, the shape is concave. The Moments
center (center of mass) of a concave shape may land **outside the mask** —
in empty space. We detect this and apply the safety fallback.

---

### 5.4 Classification Decision

```
circ > 0.7?  ──YES──→  CIRCULAR  →  use Distance Transform
     │NO
AR > 2.0?    ──YES──→  ELONGATED →  use Moments + Safety Check
     │NO
sol < 0.85?  ──YES──→  CONCAVE   →  use Moments + Safety Check (extra care)
     │NO
             ────────→  COMPACT   →  use Moments + Safety Check
```

---

## 6. Stage 3: 2D Grasp Point — Adaptive Method

Based on the shape classification, one of two methods is used.

---

### 6.1 Distance Transform (for circular objects)

**What it computes:** For every pixel inside the mask, compute its distance to the
nearest edge. The pixel with the **highest value** is the center of the largest
circle that can be inscribed inside the mask.

```
Distance Transform heatmap:

    ░░░░░░░░░░░
    ░ 1 2 3 2 1 ░
    ░ 2 3 4 3 2 ░    ← "4" is the max DT point (white dot in heatmap)
    ░ 1 2 3 2 1 ░
    ░░░░░░░░░░░
```

**Why it works for circles:** For a circular mask, the point furthest from all
edges is the geometric center. DT and Moments give the same result.

**Bonus:** The max DT value = radius of largest inscribed circle. This directly
tells us whether the suction cup can fit:

```
max_dt = 30px, suction_cup_radius = 15px
30 > 15  →  suction cup fits ✓

max_dt = 8px, suction_cup_radius = 15px
8 < 15   →  suction cup hangs over edge ✗
```

**Implementation** (`centroid_utils.py:212-238`):
```python
dist_transform = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
_, _, _, max_loc = cv2.minMaxLoc(dist_transform)
cx, cy = max_loc
```

---

### 6.2 Moments / Center of Mass (for all other shapes)

**What it computes:** The weighted average position of all pixels in the mask.
Every pixel votes equally, so the result is the true geometric center.

**Formula:**
```
cx = Σ(x × pixel_value) / Σ(pixel_value)  =  M10 / M00
cy = Σ(y × pixel_value) / Σ(pixel_value)  =  M01 / M00
```

**Why DT fails for non-circular shapes:**

```
Tall bottle mask (elongated):

    ┌──┐
    │  │   ← narrow neck: DT values ~10px from edge
    │  │
    ├──┤
    │    │
    │    │  ← wide body: DT values ~35px from edge  ← DT picks HERE (widest!)
    │    │
    └────┘

    DT result:   bottom 1/3 of bottle  (WRONG — biased to widest part)
    Moments:     middle of bottle       (CORRECT — true geometric center)
    Error:       ~70-115px off-center
```

**Implementation** (`centroid_utils.py:117-139`):
```python
M = cv2.moments(mask_uint8)
cx = M["m10"] / M["m00"]
cy = M["m01"] / M["m00"]
```

---

## 7. Stage 4: Safety Check & Blending

After computing the candidate point (from DT or Moments), we verify it is safe for
suction. This stage only applies to the Moments path (circular objects skip this).

### Problem 1: Moments point is OUTSIDE the mask

This happens with concave shapes (C-shape, L-bracket, donut). The average of all
pixels can fall in the empty "hole" of the shape.

```
L-bracket mask:

  ██████████
  ██████████
  ██
  ██           ⊙  ← Moments center (average of all pixels)
  ██               BUT this point is in empty space!
  ██████████
  ██████████
```

**Check:** `mask[moments_y, moments_x] > 0`

**Fallback:** Find the nearest mask pixel to the Moments center, then blend
50% nearest-point + 50% DT max for edge safety.

---

### Problem 2: Moments point is too close to the edge

Even if inside the mask, the suction cup needs clearance. If the center is only
3px from the edge, the suction cup (radius 15px) will hang over the boundary
and lose seal.

```
┌──────────────────────────┐
│                          │
│                    ⊙─3px─│ edge
│                          │
└──────────────────────────┘
Suction cup at ⊙: cup extends 15px right → hangs over edge → air leak → fail
```

**Check:**
```python
edge_distance = dist_transform[moments_y, moments_x]  # DT value at Moments point
min_clearance = max_dt * 0.20                          # 20% of max inscribed radius
```

**Why 20% relative (not a fixed pixel value)?**

The suction cup radius check in pixel space is not reliable because we don't know
the real-world scale yet (depends on camera distance). Instead, 20% of max_dt
means "stay within the inner 80% safe zone of the object." The real suction cup
radius check happens later in 3D.

---

### 7.1 The Three Outcomes

**Case 1: SAFE — use Moments directly**
```
edge_distance >= min_clearance  →  Moments point is balanced and safe
```

**Case 2: BLEND — too close to edge, push toward DT**
```
edge_distance < min_clearance  (but inside mask)
```

Dynamic weight based on how close to edge:
```python
edge_ratio = edge_distance / max_dt        # 0 = at edge, 1 = at max
dt_weight = min(1.0 - (edge_ratio / 0.20), 0.7)   # max cap: 70% DT

final_x = (1 - dt_weight) × moments_x + dt_weight × dt_x
final_y = (1 - dt_weight) × moments_y + dt_weight × dt_y
```

The blend is **dynamic** — the closer to the edge, the harder the push toward DT:

```
edge_distance    dt_weight    interpretation
─────────────    ─────────    ──────────────────────────────────────────
6px  (almost ok)    14%       small nudge toward DT
4px  (moderate)     43%       significant push toward DT
2px  (very close)   70%       maximum push — 30% Moments + 70% DT
1px  (at edge)      70%       same maximum (capped at 70%)
```

The 70% cap ensures the final point never fully abandons the geometric center.
Even in the worst case we keep 30% Moments influence.

**Case 3: FALLBACK — Moments outside mask, use DT**
```
moments_inside_mask == False
→ nearest mask point to Moments + 50% blend with DT max
→ if still outside mask: use DT directly
```

---

### 7.2 Final Safety Check

After blending, one last check:
```python
if mask[final_cy, final_cx] == 0:
    return (dt_cx, dt_cy)   # blended point still outside — use DT as last resort
```

---

## 8. Stage 5: 3D Grasp Point (Real Depth)

> **Status:** Code fully implemented. Waiting for real RealSense L515 depth data.
> Currently the pipeline uses dummy depth (800mm flat), causing this stage to fall
> back to Stage 3 (2D adaptive method).

When real depth data is available, the pipeline upgrades from "balanced 2D center"
to "flattest point on the top surface."

### 8.1 Top Surface Extraction

For a standing bottle, only the **top cap area** is graspable. We use depth to isolate it:

```python
min_depth = depth within mask (closest to camera = top surface)
max_depth = depth within mask (furthest = bottom or table)
depth_range = max_depth - min_depth

top_surface_threshold = min_depth + depth_range * 0.20
top_surface_mask = pixels where depth <= threshold
```

This selects the top 20% of the object's depth range — the cap/lid area.

```
Side view of standing bottle:
        ┌──────┐  ← min_depth (closest to camera)
        │ cap  │  ← top_surface_mask selects this
        ├──────┤  ← min_depth + 20% of depth_range
        │      │
        │ body │
        │      │
        └──────┘  ← max_depth
```

If the top surface region is too small for the suction cup, the threshold is
expanded to 30%, 40%, 50% of depth_range until enough area is found.

---

### 8.2 Surface Normal Computation

Surface normals describe which direction each surface patch is facing. For suction,
we want surfaces facing directly upward (normal pointing toward camera).

**Method: Sobel gradients** (`centroid_utils.py:485-519`):
```python
dzdx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=5)   # depth gradient in X
dzdy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=5)   # depth gradient in Y
normal = (-dzdx, -dzdy, 1)   →  normalize to unit vector
```

```
Flat surface:          Curved surface:
  ↑ ↑ ↑ ↑ ↑              ↑ ↗ → ↘ ↓
  ↑ ↑ ↑ ↑ ↑            ↖ ↑ ↗ → ↘
  ↑ ↑ ↑ ↑ ↑            ← ↖ ↑ ↗ →

All normals same →      Normals vary →
LOW VARIANCE (good)     HIGH VARIANCE (bad)
```

---

### 8.3 Flatness Scoring

For each pixel, compute the variance of surface normals in a local window:

```python
# Formula from "Learning Suction Graspability" (Frontiers 2022):
# var(ns) = Σ||ns,i - n̄s||² / (N-1)

variance_map = compute_normal_variance(normals, top_surface_mask, window_size=30)
flatness_score = 1.0 - normalize(variance_map)   # high flatness = low variance
```

---

### 8.4 Combined Scoring

Three factors are combined for the final score:

```
Score = 0.4 × flatness + 0.3 × edge_clearance + 0.3 × center_proximity
```

| Factor            | Weight | Meaning                                           |
|-------------------|--------|---------------------------------------------------|
| Flatness          |  40%   | Surface normal variance (lower = flatter = better)|
| Edge clearance    |  30%   | DT value at candidate point (farther from edge)   |
| Center proximity  |  30%   | Distance from top-surface centroid (closer = stable)|

The point with the highest combined score = final 3D grasp point.

---

### 8.5 2D → 3D Coordinate Transform

After finding the best pixel (u, v) in the depth image:

```python
X = (u - cx_intrinsic) * Z / fx
Y = (v - cy_intrinsic) * Z / fy
Z = depth[v, u]   # in meters
```

Where `fx, fy, cx, cy` are the RealSense L515 camera intrinsic parameters.

The resulting (X, Y, Z) is in the **camera coordinate frame**. A further
camera-to-robot transform is needed to get robot arm coordinates.

---

## 9. Why Not Just Use One Method?

| Method          | Circular | Elongated | Concave | Bonus          |
|-----------------|----------|-----------|---------|----------------|
| Distance Transform | ✅ perfect | ❌ biased to widest | ✅ ok | inscribed radius |
| Moments         | ✅ ~same as DT | ✅ true center | ⚠️ may be outside mask | none |
| DT only         | ✅ | ❌ 70-115px off | ✅ | radius |
| Moments only    | ✅ | ✅ | ❌ outside mask risk | none |

No single method handles all cases. The adaptive approach uses:
- DT for circles (gives both center and suction feasibility check for free)
- Moments for everything else (true geometric center)
- Safety check to catch the Moments failure case (outside mask or near edge)

---

## 10. Failure Cases

### Failure Case 1: DT Bias (solved)

**Problem:** DT alone biases toward the widest part of the object, not the
geometric center. For a tall bottle, this means the DT center lands at the
wide base/body area, 70–115px away from the true center.

**Solution:** Moments-based adaptive method routes non-circular objects to
Moments instead of DT.

**Visualization:** `output/distance_transform_visualization.png` shows this
bias clearly — the DT center and Moments center diverge significantly for
elongated objects.

### Failure Case 2: Moments Outside Mask (handled)

**Problem:** For concave shapes, Moments can fall outside the mask in empty space.

**Solution:** Safety check detects this and applies nearest-point + DT blend.

### Failure Case 3: Segmentation Quality

**Problem:** If UOAIS produces a poor mask (fragmented, oversized, undersized),
the computed grasp point may be inaccurate regardless of centroid method.

**Current status:** Not yet handled — we trust the segmentation quality.

### Failure Case 4: Uniform Depth (current limitation)

**Problem:** Without real depth, all Z values are 800mm (dummy). The 3D grasp
point pipeline falls back to 2D. Top surface detection and flatness scoring
are bypassed entirely.

**Solution:** Connect RealSense L515 for real depth data.

---

## 11. Current Status

| Stage | Status | Notes |
|-------|--------|-------|
| UOAIS Segmentation | ✅ Working | Tested on IMG_1761 dataset (11 objects) |
| Shape Classification | ✅ Working | circ, AR, sol correctly computed |
| 2D Grasp Point (DT) | ✅ Working | Correct for circular objects |
| 2D Grasp Point (Moments) | ✅ Working | Correct for elongated/compact objects |
| Safety Check + Blending | ✅ Working | Dynamic weighting implemented |
| 3D Surface Normals | ⚠️ Code ready | Falls back to 2D due to dummy depth |
| 3D Flatness Scoring | ⚠️ Code ready | Falls back to 2D due to dummy depth |
| Real Depth Integration | ❌ Not started | RealSense L515 not yet connected |
| Robot Arm Communication | ❌ Not started | No control code yet |
| Live Stream / Loop | ❌ Not started | Currently per single image only |

**Output files from current testing (`output_img1761_split/`):**
- `obj_00/` to `obj_10/` — 11 per-object folders
- Each contains: mask overlay, shape analysis, DT heatmap, moments+safety,
  final center, full scene highlight, DT vs Moments comparison, clean crop,
  binary mask, info.txt

**JSON output (`output_img1761_raw/IMG_1761_centroids.json`):**
- All centroid data stored per object
- `centroid_3d.z = 0.8` (dummy — will be real depth when camera connected)
- `grasp_point_3d = null` (blocked on real depth)

---

## 12. Next Steps

1. **Resolve Distance Transform bias problem** — Validate the adaptive method
   (Moments + Safety Check) produces correct grasp points across all object shapes
   in the full dataset

2. **Integrate 3D depth data** — Connect RealSense L515 to get real depth,
   enabling top surface extraction and flatness scoring for the final 3D grasp point

3. **Combine 2D + 3D** — Full pipeline with real depth: 2D shape-adaptive center
   as starting candidate, refined by 3D flatness scoring → final (X, Y, Z) output

4. **Connect to robot arm** — Send (X, Y, Z) coordinates to robot arm controller,
   implement pick-and-place execution sequence

---

## Related Documents

- [CENTROID_COMPUTATION.md](CENTROID_COMPUTATION.md) — API reference for all centroid functions
- [LITERATURE_REVIEW.md](LITERATURE_REVIEW.md) — Research papers backing the approach
- [QUICK_START.md](QUICK_START.md) — How to run the pipeline
