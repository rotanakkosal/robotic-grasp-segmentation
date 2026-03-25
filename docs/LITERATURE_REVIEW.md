# Literature Review: Grasp Point Detection for Suction-Based Robot Picking

This document provides a comprehensive literature review on methods for finding the optimal central/grasp point on objects for robotic suction cup picking, with a focus on approaches that work with instance segmentation masks (e.g., from UOAIS).

**Last Updated**: 2026-02-23

## Table of Contents

1. [Research Question](#research-question)
2. [Foundational Works: The Dex-Net Family](#1-foundational-works-the-dex-net-family)
3. [Large-Scale Benchmarks and Datasets](#2-large-scale-benchmarks-and-datasets)
4. [Deep Learning Approaches for Suction Grasp Detection](#3-deep-learning-approaches-for-suction-grasp-detection)
5. [RGB-Only and Depth-Free Approaches](#4-rgb-only-and-depth-free-approaches)
6. [Geometric and Heuristic Approaches](#5-geometric-and-heuristic-approaches)
7. [Segmentation-to-Grasp Pipelines](#6-segmentation-to-grasp-pipelines)
8. [Shape-Adaptive Grasp Point Selection](#7-shape-adaptive-grasp-point-selection)
9. [Grasp Quality Metrics for Suction Cups](#8-grasp-quality-metrics-for-suction-cups)
10. [Comparison of Approaches](#9-comparison-of-approaches)
11. [Recommended Multi-Level Strategy](#10-recommended-multi-level-strategy)
12. [References](#references)

---

## Research Question

**How to find the optimal grasp point for a vacuum suction cup gripper given instance segmentation masks and optionally RGB-D data?**

Sub-questions:
- What makes a good suction grasp point?
- How can the grasp point dynamically adapt to different object shapes?
- What methods work without depth data (RGB-only)?
- How to evaluate surface suitability for suction from 2D masks alone?
- What role do geometric vs. learned approaches play?

---

## 1. Foundational Works: The Dex-Net Family

### 1.1 Dex-Net 3.0 (Mahler et al., ICRA 2018)

**Title**: Computing Robust Vacuum Suction Grasp Targets in Point Clouds Using a New Analytic Model and Deep Learning

**URL**: https://arxiv.org/abs/1709.06670

**Key Approach**: Proposes a compliant suction contact model that evaluates two critical factors:
1. **Seal quality** between the suction cup and the target surface
2. **Wrench resistance** — the ability of the grasp to resist an external gravity wrench

A dataset of 2.8 million point clouds with suction grasps and robustness labels was generated from 1,500 3D object models. A Grasp Quality Convolutional Neural Network (GQ-CNN) was trained to classify suction grasp robustness from depth images.

**Results**:

| Object Type | Success Rate |
|-------------|-------------|
| Basic (prismatic) | 98% |
| Typical (complex) | 82% |
| Adversarial | 58% → 81% (with adversarial training) |

**Key Insight**:
> "Traditional heuristics such as grasping near the object centroid or at the center of planar surfaces work well for prismatic objects but may fail on objects with non-planar surfaces near the object centroid."

**Relevance**: The suction contact model and wrench resistance metric are directly applicable as evaluation criteria for grasp points derived from UOAIS masks. The GQ-CNN architecture is a benchmark for suction grasp scoring.

---

### 1.2 Dex-Net 4.0 / Ambidextrous Grasping (2019)

**Title**: Learning Ambidextrous Robot Grasping Policies

**Authors**: Jeffrey Mahler, Matthew Matl, et al. (UC Berkeley)

**Venue**: Science Robotics, 2019

**Key Approach**: Trains separate GQ-CNNs for parallel-jaw and suction cup grippers on 5 million synthetic depth images from 1,664 object models in simulated heaps. A POMDP framework selects between gripper types using robust wrench resistance as a common reward function.

**Results**: >95% reliability clearing bins of 25 novel objects at >300 mean picks per hour.

**Relevance**: Validates the suction-specific GQ-CNN pipeline at scale. The concept of choosing between gripper modalities based on surface properties is useful for systems with hybrid grippers.

---

### 1.3 FC-GQ-CNN (Satish et al., 2019)

**Title**: On-Policy Dataset Synthesis for Learning Robot Grasping Policies Using Fully Convolutional Deep Networks

**Venue**: IEEE Robotics and Automation Letters, 2019

**Key Approach**: Extends GQ-CNN to a fully convolutional architecture (FC-GQ-CNN) that produces **pixel-wise grasp quality maps** in a single forward pass, evaluating millions of 4-DOF grasp candidates simultaneously.

**Results**: Plans grasps in 0.625s while considering 5000x more candidates; achieves up to 296 mean picks per hour.

**Relevance**: The pixel-wise output format directly maps to using segmentation masks as attention regions. One can mask the FC-GQ-CNN output with UOAIS segmentation masks to select per-object optimal suction points.

---

## 2. Large-Scale Benchmarks and Datasets

### 2.1 GraspNet-1Billion (Fang et al., CVPR 2020)

**Title**: GraspNet-1Billion: A Large-Scale Benchmark for General Object Grasping

**URL**: https://graspnet.net/

**Key Stats**: 97,280 RGB-D images from 190 cluttered scenes containing 88 daily objects. Over 1.1 billion grasp poses annotated via analytic force-closure computation.

**Relevance**: Provides the infrastructure (scenes, objects, segmentation masks) upon which SuctionNet-1Billion was built.

---

### 2.2 SuctionNet-1Billion (Cao et al., RA-L 2021)

**Title**: SuctionNet-1Billion: A Large-Scale Benchmark for Suction Grasping

**URL**: https://arxiv.org/abs/2103.12311

**Key Approach**: Built on GraspNet-1Billion (97,280 RGB-D images, 190 scenes). Provides more than **1 billion suction grasp annotations** with two label types:
1. **Seal labels** — whether suction can form a seal
2. **Wrench labels** — ability to resist gravity wrench

Proposes a new physical model for analytically evaluating seal formation and wrench resistance.

**Relevance**: **Highly relevant.** First billion-scale suction grasp benchmark. The seal+wrench evaluation framework can be applied to score grasp points identified from UOAIS masks.

---

## 3. Deep Learning Approaches for Suction Grasp Detection

### 3.1 Suction-Based Affordance Maps (Utomo & Cahyadi, 2021)

**Title**: Suction-based Grasp Point Estimation in Cluttered Environment for Robotic Manipulator Using Deep Learning-based Affordance Map

**Venue**: Machine Intelligence Research, 2021

**Key Approach**: Uses a modified deep neural network backbone for semantic segmentation to produce an **affordance map** — a pixel-wise probability map representing the probability of successful suctioning at each pixel location.

**Results**: 88.83% precision vs. 83.4% for prior state-of-the-art.

**Relevance**: **Highly relevant.** The affordance map concept directly complements UOAIS masks — one can multiply the affordance map by the segmentation mask to find the optimal suction point per object.

---

### 3.2 SG-U-Net++ (Jiang et al., Frontiers 2022)

**Title**: Learning Suction Graspability Considering Grasp Quality and Robot Reachability for Bin-Picking

**URL**: https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2022.806898/full

**Key Approach**: Proposes the **Suction Graspability U-Net++ (SG-U-Net++)** that predicts pixel-wise maps for both grasp quality and robot reachability. Two key evaluation metrics:

- **Jq (Grasp Quality)** = combination of:
  - **Jc** — normalized distance to center of graspable area (closer to center = more stable)
  - **Js** — surface flatness/smoothness measured by surface normal variance

**Grasp Quality Formula**:
```
Js = 0.9 × var(ns) + 0.1 × e^(-5 × res(ps))
```
Where `var(ns)` = variance of surface normals, `res(ps)` = plane fitting residual.

**Results**: 560 picks per hour in industrial bin-picking.

**Relevance**: **Very highly relevant.** The Jc metric (center proximity) and Js metric (surface flatness) provide a principled framework for scoring grasp points within UOAIS segmentation masks.

---

### 3.3 Sim-Suction (Li & Cappelleri, IEEE T-RO 2023)

**Title**: Sim-Suction: Learning a Suction Grasp Policy for Cluttered Environments Using a Synthetic Benchmark

**URL**: https://arxiv.org/abs/2305.16378

**Key Approach**: Creates the **Sim-Suction-Dataset** (500 cluttered environments, 3.2 million annotated suction poses) combining analytic models with dynamic physical simulations. Trains **Sim-Suction-PointNet** to generate 6D suction grasp poses by learning point-wise affordances.

**Results**: 96.76% (Level 1), 94.23% (Level 2), 92.39% (mixed) — outperforming benchmarks by ~21% in mixed scenes.

**Relevance**: Demonstrates the power of combining segmentation with point-wise suction affordance learning.

---

### 3.4 DYNAMO-GRASP (Yang et al., CoRL 2023)

**Title**: DYNAMO-GRASP: DYNAMics-aware Optimization for GRASP Point Detection in Suction Grippers

**URL**: https://sites.google.com/view/dynamo-grasp

**Key Approach**: Goes beyond static grasp quality to model **object dynamics during the picking process** — whether the object will rotate, slide, or detach during lifting/transport. Uses dynamic simulation to score candidate grasp points.

**Results**: 98% in simulated tasks; 80% in real-world adversarial scenarios; up to **48% improvement** over state-of-the-art.

**Key Insight**: Static seal quality alone is insufficient — objects that detach or rotate during transport due to poor grasp placement relative to center of mass represent a key failure mode.

**Relevance**: **Highly relevant.** Motivates choosing grasp points that are dynamically stable (near center of mass), not just geometrically centered.

---

### 3.5 AnyGrasp (Fang et al., IEEE T-RO 2023)

**Title**: AnyGrasp: Robust and Efficient Grasp Perception in Spatial and Temporal Domains

**URL**: https://arxiv.org/abs/2212.08333

**Key Approach**: Dense supervision strategy generating accurate 7-DoF grasp poses. Incorporates **center-of-mass awareness** into the learning process to improve grasping stability.

**Results**: 93.3% success rate clearing bins of >300 unseen objects, on par with human subjects.

**Relevance**: Demonstrates that center-of-mass awareness significantly improves grasp stability.

---

### 3.6 Diffusion-Suction (Huang et al., ICRA 2025)

**Title**: Diffusion Suction Grasping with Large-Scale Parcel Dataset

**URL**: https://arxiv.org/abs/2502.07238

**Key Approach**: Introduces the **Parcel-Suction-Dataset** (25,000 cluttered scenes, 410 million suction annotations) and **Diffusion-Suction**, which reformulates suction grasp prediction as a conditional denoising diffusion process.

**Relevance**: Represents the cutting edge of suction grasp prediction. State-of-the-art on SuctionNet-1Billion benchmark.

---

## 4. RGB-Only and Depth-Free Approaches

### 4.1 OptiGrasp (2024)

**Title**: OptiGrasp: Optimized Grasp Pose Detection Using RGB Images for Warehouse Picking Robots

**URL**: https://arxiv.org/abs/2409.19494

**Key Approach**: Leverages foundation models (DINOv2 backbone from Depth Anything with frozen weights, Dense Prediction Transformer decoder) to predict suction grasp poses from **RGB images only**, trained solely on synthetic data.

**How It Determines Grasp Point**: Predicts three dense pixel-wise maps: an affordance grasp score map, a pitch angle map, and a yaw angle map. The pixel with the highest affordance grasp score is selected as the suction grasp point.

**Results**: 82.3% success rate in real-world (176/215 picks); >90% on easy objects (boxes, bottles).

**Relevance**: **Directly relevant** for scenarios where depth sensing is unavailable. Shows that RGB-only suction grasp prediction is viable using foundation model features.

---

### 4.2 MonoGraspNet (Zhai et al., 2023)

**Title**: MonoGraspNet: 6-DoF Grasping with a Single RGB Image

**Key Approach**: First deep learning pipeline for 6-DoF robotic grasping using only a single RGB image, eliminating dependency on depth sensors. Particularly effective for transparent/reflective objects where depth sensors fail.

**Relevance**: Useful fallback when depth data is unreliable or unavailable.

---

### 4.3 RGBGrasp (Liu et al., 2024)

**Title**: RGBGrasp: Image-based Object Grasping by Capturing Multiple Views during Robot Arm Movement with Neural Radiance Fields

**URL**: https://arxiv.org/abs/2311.16592

**Key Approach**: Uses an eye-on-hand RGB camera that captures multiple views as the robot arm approaches. Employs NeRF to reconstruct 3D geometry from RGB views.

**Results**: Maintains over 80% success rate. Particularly effective for transparent and specular objects.

---

### 4.4 GR-ConvNet v2 (2022)

**Title**: GR-ConvNet v2: A Real-Time Multi-Grasp Detection Network for Robotic Grasping

**URL**: https://www.mdpi.com/1424-8220/22/16/6208

**Key Approach**: Generative Residual CNN that generates grasp quality, angle, and width per pixel. Can operate on RGB-only or RGB-D input at **20ms inference time**.

**Results**: 98.8% accuracy on Cornell, 95.4% grasp success on real-world novel objects.

---

## 5. Geometric and Heuristic Approaches

### 5.1 Cartman / Boundary-Distance Heuristic (Morrison et al., ICRA 2018)

**Title**: Cartman: The Low-Cost Cartesian Manipulator that Won the Amazon Robotics Challenge

**URL**: https://arxiv.org/abs/1709.06283

**Key Approach**: For each potential suction grasp point on a segmented point cloud:
- **75% weight**: Normalized distance from the object boundary (farther from edges = better)
- **25% weight**: Surface curvature at the point (flatter = better)

**Results**: **Won the 2017 Amazon Robotics Challenge stowing task.**

**Relevance**: **Extremely relevant and immediately implementable.** Given a UOAIS segmentation mask, the distance transform finds the point farthest from the boundary (equivalent to the maximum inscribed circle center). This is a proven, competition-winning approach.

---

### 5.2 Distance Transform / Maximum Inscribed Circle (Classic)

**Concept**: Apply the distance transform to a binary segmentation mask. The pixel with the maximum distance value is the center of the **maximum inscribed circle** — the point farthest from all edges.

**Algorithm**:
```python
dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
_, _, _, max_loc = cv2.minMaxLoc(dist)
cx, cy = max_loc  # Optimal grasp point
```

**Strengths**:
- **Shape-adaptive**: naturally handles convex, concave, and irregular shapes
- **O(n) computation**: extremely fast
- **No training required**: purely geometric
- **Robust**: always finds a point inside the mask

**Limitations**: Does not consider surface flatness, seal quality, or mass distribution (purely 2D).

**Relevance**: **The most directly applicable geometric method for UOAIS masks.** Works without depth data. The competition-winning Cartman approach uses this as its primary component (75% of the score).

---

### 5.3 Medial Axis Transform / Skeletonization

**Title**: Planning Grasps for Robotic Hands Using a Novel Object Representation Based on the Medial Axis Transform

**Venue**: IEEE IROS 2011 (foundational); extended through 2024

**Key Approach**: Represents objects as a set of inscribed spheres along the medial axis (skeleton). The skeleton captures object symmetry, thickness, and structural topology.

**For Suction Grasping**: The skeleton point with the **highest distance-transform value** combines structural centering with boundary avoidance — ideal for finding the most interior, stable suction point.

**Strengths**: Captures object structure (necks, forks, branches); identifies natural grasp regions.

---

### 5.4 2D Skeleton-Based Keypoint Generation (ICRoM 2023)

**Title**: 2D Skeleton-Based Keypoint Generation Method for Grasping Objects with Roughly Uniform Height Variation

**URL**: https://ieeexplore.ieee.org/document/10412575

**Key Approach**: Applies the **Straight Skeleton (StSkel)** method to object contours from segmentation masks:
1. Capture 2D image and create mask
2. Detect contour and construct polygon
3. Apply Straight Skeleton to compute skeleton
4. Generate grasp keypoints from skeleton structure

**Results**: **98% grasp success rate**, outperforming state-of-the-art by 2%.

**Relevance**: **Very highly relevant.** Directly operates on segmentation masks (like UOAIS output) to produce grasp keypoints via skeletal analysis. The straight skeleton method is particularly well-suited for finding natural grasp points on irregularly shaped objects.

---

### 5.5 Team Delft / Centroid-Along-Normal (ARC 2016 Winner)

**Key Approach**: Approaches the estimated object centroid along the inward surface normal. The centroid of the segmented point cloud is used as the grasp target.

**Strengths**: Extremely simple, fast, effective for convex objects.

**Limitations**: Centroid may fall outside the object for concave shapes; does not evaluate surface quality.

---

### 5.6 Fast Suction-Grasp-Difficulty Estimation (2023)

**Title**: Fast Suction-Grasp-Difficulty Estimation for High Throughput Plastic-Waste Sorting

**Venue**: Journal of Mechanical Science and Technology, 2023

**Results**: 30.9 ms for multi-object estimation; 1.65% average error; **94.4% grasping success**.

**Relevance**: Demonstrates that fast geometric heuristics achieve high success rates — supporting lightweight methods over heavy deep learning for real-time picking.

---

## 6. Segmentation-to-Grasp Pipelines

### 6.1 Multi-Affordance Grasping (Zeng et al., IJRR 2018)

**Title**: Robotic Pick-and-Place of Novel Objects in Clutter with Multi-Affordance Grasping

**URL**: https://arxiv.org/abs/1710.01330

**Key Approach**: Uses two-tower ResNet-101 (RGB + Depth) FCNs to produce dense **pixel-wise affordance maps** for multiple grasping primitives. For suction, the affordance map gives a 0-1 probability per pixel. The grasp point is the argmax within the target object region.

**Key Paradigm**: **"Predict dense affordance map, then select best point per object using segmentation mask."** This is precisely the pipeline architecture that works with UOAIS masks.

---

### 6.2 Mask-GD: Segmentation Based Robotic Grasp Detection (2021)

**URL**: https://arxiv.org/abs/2101.08183

**Key Approach**: Two-stage architecture: (1) instance segmentation isolates the target object, (2) grasp detector operates only on the segmented mask region, eliminating background interference.

**Results**: 96.4% image-wise, 96.5% object-wise accuracy.

---

### 6.3 GraspSAM (2024)

**Title**: GraspSAM: When Segment Anything Model Meets Grasp Detection

**URL**: https://arxiv.org/abs/2409.12521

**Key Approach**: Extends SAM (Segment Anything Model) for prompt-driven, category-agnostic grasp detection. Uses adapters, learnable tokens, and a lightweight decoder to predict grasp poses from segmentation.

---

### 6.4 SuctionPrompt (2024)

**Title**: SuctionPrompt: Visual-Assisted Robotic Picking with a Suction Cup Using Vision-Language Models

**Key Approach**: Uses vision-language models (VLMs) with prompting techniques for suction-cup-based picking without task-specific training.

**Results**: 75.4% suction point accuracy; 65.0% picking success rate.

---

## 7. Shape-Adaptive Grasp Point Selection

### 7.1 ZeroGrasp (Iwase et al., CVPR 2025)

**Title**: ZeroGrasp: Zero-Shot Shape Reconstruction Enabled Robotic Grasping

**URL**: https://arxiv.org/abs/2504.10857

**Key Approach**: Simultaneously performs 3D reconstruction and grasp pose prediction. Uses multi-object encoder and 3D occlusion fields for inter-object modeling. Trained on 1M synthetic images with 11.3B grasp annotations for 12K objects.

**Shape-Adaptiveness**: Full shape reconstruction before proposing grasps — understands complete geometry even from partial views.

---

### 7.2 CenterGrasp (Chisari et al., RA-L 2024)

**Title**: CenterGrasp: Object-Aware Implicit Representation Learning for Simultaneous Shape Reconstruction and 6-DoF Grasp Estimation

**URL**: https://arxiv.org/abs/2312.08240

**Key Approach**: Learns a continuous latent space encoding shapes and valid grasps. Different object shapes map to different grasp distributions.

**Results**: 38.5mm improvement in shape reconstruction and 33 percentage point improvement in grasp success.

---

### 7.3 Adaptive Grasp Pose Optimization (MDPI 2025)

**Title**: Adaptive Grasp Pose Optimization for Robotic Arms Using Low-Cost Depth Sensors

**URL**: https://www.mdpi.com/1424-8220/25/3/909

**Key Approach**: Three-stage pipeline:
1. Segment the target object
2. Fit an **ellipsoid** to determine principal axes
3. Apply nonlinear optimization for a 6-DoF grasp pose

The object is grasped by its thinnest part as determined by the ellipsoid axis ratios. Different shapes produce different principal axis configurations.

---

### 7.4 PCF-Grasp (2025)

**Title**: PCF-Grasp: Converting Point Completion to Geometry Feature to Enhance 6-DoF Grasp

**URL**: https://arxiv.org/abs/2504.16320

**Key Approach**: Converts point completion results (completed 3D shapes from partial views) into geometry features for grasp networks. Inspired by how humans infer full object geometry from a single view.

**Results**: 17.8% improvement over SOTA in real-world experiments.

---

## 8. Grasp Quality Metrics for Suction Cups

### 8.1 Seal Formation Criteria

Key factors for evaluating whether a suction cup can form a seal:

| Factor | Description | Measurement |
|--------|-------------|-------------|
| **Surface flatness** | Vacuum cup works better on flat, smooth surfaces | Plane fitting residual, normal variance |
| **Surface curvature** | High curvature compromises seal formation | Sobel-based curvature estimation |
| **Normal alignment** | Approach vector should align with surface normal | Dot product of approach and normal |
| **Material properties** | Porosity and deformability affect seal | Not measurable from vision alone |

### 8.2 Wrench Resistance Criteria

| Factor | Description | Impact |
|--------|-------------|--------|
| **Gravity wrench** | Must support object weight | Off-center grasps create torques |
| **Dynamic resistance** | Forces during acceleration/transport | DYNAMO-GRASP addresses this |
| **Center-of-mass proximity** | Closer to COM = more stable | Rotational displacement is primary failure mode |

### 8.3 Combined Scoring (From Literature)

**Cartman (Competition Winner)**:
```
Score = 0.75 × BoundaryDistance + 0.25 × SurfaceFlatness
```

**SG-U-Net++ (Industrial Bin-Picking)**:
```
Jq = Jc (center proximity) + Js (surface quality)
Js = 0.9 × var(normals) + 0.1 × e^(-5 × planeFitResidual)
```

**Key Principle from Literature**: Boundary distance (edge clearance) is the single most important factor for suction success.

---

## 9. Comparison of Approaches

### Summary Table

| Method | Year | Input | Suction-Specific | Mask-Compatible | Depth Required | Real-time |
|--------|------|-------|------------------|-----------------|----------------|-----------|
| Dex-Net 3.0 / GQ-CNN | 2018 | Depth | Yes | No (point cloud) | Yes | No |
| FC-GQ-CNN | 2019 | Depth | Yes | Yes (maskable) | Yes | Yes |
| Cartman Heuristic | 2017 | Mask + Depth | Yes | **Yes** | Partial | **Yes** |
| SuctionNet-1Billion | 2021 | RGB-D | Yes | Yes | Yes | N/A |
| Affordance Map | 2021 | RGB-D | Yes | Yes (maskable) | Yes | Near |
| SG-U-Net++ | 2022 | Depth | Yes | Yes (pixel-wise) | Yes | Yes |
| Sim-Suction | 2023 | Point cloud | Yes | Yes | Yes | Near |
| DYNAMO-GRASP | 2023 | Point cloud + sim | Yes | Yes | Yes | No |
| Skeleton Keypoints | 2023 | **2D mask only** | Yes (top-down) | **Yes** | **No** | **Yes** |
| OptiGrasp | 2024 | **RGB only** | Yes | Implicit | **No** | Yes |
| Diffusion-Suction | 2025 | Point cloud | Yes | Yes | Yes | No |
| **Distance Transform** | Classic | **2D mask only** | Yes (top-down) | **Yes** | **No** | **Yes** |

### When to Use Each Approach

| Scenario | Best Method | Why |
|----------|-------------|-----|
| **No depth, regular shapes** | Distance Transform | Shape-adaptive, finds true center, fast |
| **No depth, irregular shapes** | Skeleton + Distance Transform | Captures topology, handles complex shapes |
| **With depth, flat objects** | Suction (normal variance + boundary distance) | Surface flatness analysis |
| **With depth, cluttered scene** | FC-GQ-CNN or Affordance Map | Learned quality scoring |
| **Dynamic picking** | DYNAMO-GRASP | Accounts for transport stability |
| **Industrial speed** | Distance Transform or Fast Heuristic | Sub-millisecond computation |

---

## 10. Recommended Multi-Level Strategy

Based on the literature, the following **dynamic, shape-adaptive approach** is recommended for computing optimal suction grasp points from UOAIS segmentation masks:

### Tier 1: Geometric Baseline (No Learning, Mask-Only) — ALWAYS AVAILABLE

These methods work with segmentation masks alone, no depth required:

1. **Distance Transform Maximum** (Primary)
   - Apply `cv2.distanceTransform` to the binary mask and take `argmax`
   - Finds the point maximally distant from all boundaries
   - Equivalent to the center of the **maximum inscribed circle**
   - Naturally handles convex, concave, and irregular shapes
   - This is the core of the Cartman approach (75% of score) that **won the Amazon Robotics Challenge**

2. **Skeleton + Distance Transform** (For Complex Shapes)
   - Compute the medial axis/skeleton of the mask
   - Select the skeleton point with the highest distance-transform value
   - Combines structural centering with boundary avoidance
   - Based on 2D Skeleton-Based Keypoint method (98% success rate)

3. **Weighted Centroid with Fallback**
   - Compute geometric centroid (moments or median)
   - If centroid falls outside the mask (concave shapes), fall back to distance transform
   - Simple, fast, effective for regular shapes

### Tier 2: Geometric + Depth Scoring — WHEN DEPTH AVAILABLE

4. **Multi-Factor Scoring** (Based on Cartman + SG-U-Net++)
   - Use distance transform for top-k candidate points
   - Score each candidate:
     - **Jc** (center proximity) — already from distance transform
     - **Js** (surface flatness) — normal variance from depth
     - **Edge clearance** — distance from mask boundary
   - Weighted combination: `Score = 0.4 × Flatness + 0.35 × EdgeClearance + 0.25 × CenterProximity`

5. **Top-Surface Detection** (For 3D Objects)
   - Use depth to identify the topmost surface (closest to camera)
   - Apply distance transform within the top surface mask only
   - Based on insights from Dex-Net 3.0 about seal formation on the approach surface

### Tier 3: Learned Affordance — WHEN TRAINING DATA AVAILABLE

6. **Pixel-Wise Affordance Network**
   - Train FC-GQ-CNN or SG-U-Net++ style network
   - Mask output with UOAIS segmentation masks
   - Select per-object maximum as grasp point

### Key Design Principles (Consensus Across Literature)

| Principle | Source | Weight |
|-----------|--------|--------|
| **Boundary distance is #1** | Cartman (2017), Dex-Net 3.0 (2018) | 75% in competition winner |
| **Surface flatness matters** | SG-U-Net++ (2022), Dex-Net 3.0 | Secondary factor (25%) |
| **Center-of-mass proximity** | DYNAMO-GRASP (2023), AnyGrasp (2023) | Improves dynamic stability |
| **Shape-adaptiveness** | Skeleton (2023), Distance Transform | Critical for diverse objects |
| **Dynamic stability** | DYNAMO-GRASP (2023) | Frontier challenge |
| **Affordance maps + masks** | Zeng et al. (2018), FC-GQ-CNN (2019) | Dominant paradigm |

---

## References

### Foundational
1. Mahler, J., Matl, M., et al. (2018). *Dex-Net 3.0: Computing Robust Vacuum Suction Grasp Targets in Point Clouds Using a New Analytic Model and Deep Learning.* ICRA 2018. https://arxiv.org/abs/1709.06670
2. Mahler, J., et al. (2019). *Learning Ambidextrous Robot Grasping Policies.* Science Robotics. https://doi.org/10.1126/scirobotics.aau4984
3. Satish, V., Mahler, J., Goldberg, K. (2019). *On-Policy Dataset Synthesis for Learning Robot Grasping Policies Using Fully Convolutional Deep Networks.* RA-L. https://berkeleyautomation.github.io/fcgqcnn/

### Benchmarks
4. Fang, H., Wang, C., Gou, M., Lu, C. (2020). *GraspNet-1Billion: A Large-Scale Benchmark for General Object Grasping.* CVPR 2020. https://graspnet.net/
5. Cao, H., Fang, H., Liu, W., Lu, C. (2021). *SuctionNet-1Billion: A Large-Scale Benchmark for Suction Grasping.* RA-L. https://arxiv.org/abs/2103.12311

### Deep Learning
6. Utomo, Cahyadi (2021). *Suction-based Grasp Point Estimation in Cluttered Environment Using Deep Learning-based Affordance Map.* Machine Intelligence Research. https://doi.org/10.1007/s11633-020-1260-1
7. Jiang, P., et al. (2022). *Learning Suction Graspability Considering Grasp Quality and Robot Reachability for Bin-Picking.* Frontiers in Neurorobotics. https://doi.org/10.3389/fnbot.2022.806898
8. Li, J., Cappelleri, D. (2023). *Sim-Suction: Learning a Suction Grasp Policy for Cluttered Environments Using a Synthetic Benchmark.* IEEE T-RO. https://arxiv.org/abs/2305.16378
9. Yang, B., et al. (2023). *DYNAMO-GRASP: DYNAMics-aware Optimization for GRASP Point Detection in Suction Grippers.* CoRL 2023. https://sites.google.com/view/dynamo-grasp
10. Fang, H., et al. (2023). *AnyGrasp: Robust and Efficient Grasp Perception in Spatial and Temporal Domains.* IEEE T-RO. https://arxiv.org/abs/2212.08333
11. Huang, He, et al. (2025). *Diffusion Suction Grasping with Large-Scale Parcel Dataset.* ICRA 2025. https://arxiv.org/abs/2502.07238

### RGB-Only
12. OptiGrasp (2024). *Optimized Grasp Pose Detection Using RGB Images for Warehouse Picking Robots.* https://arxiv.org/abs/2409.19494
13. Zhai, et al. (2023). *MonoGraspNet: 6-DoF Grasping with a Single RGB Image.*
14. Liu, C., et al. (2024). *RGBGrasp: Image-based Object Grasping with Neural Radiance Fields.* RA-L. https://arxiv.org/abs/2311.16592

### Geometric / Heuristic
15. Morrison, D., et al. (2018). *Cartman: The Low-Cost Cartesian Manipulator that Won the Amazon Robotics Challenge.* ICRA 2018. https://arxiv.org/abs/1709.06283
16. 2D Skeleton-Based Keypoint (2023). *Grasping Objects with Roughly Uniform Height Variation.* ICRoM 2023. https://ieeexplore.ieee.org/document/10412575
17. Fast Suction Estimation (2023). *Fast Suction-Grasp-Difficulty Estimation for High Throughput Plastic-Waste Sorting.* J. Mech. Sci. Technol.

### Segmentation-to-Grasp Pipelines
18. Zeng, A., et al. (2018). *Robotic Pick-and-Place of Novel Objects in Clutter with Multi-Affordance Grasping.* IJRR. https://arxiv.org/abs/1710.01330
19. Dong, W., et al. (2021). *Mask-GD: Segmentation Based Robotic Grasp Detection.* https://arxiv.org/abs/2101.08183
20. GraspSAM (2024). *When Segment Anything Model Meets Grasp Detection.* https://arxiv.org/abs/2409.12521

### Shape-Adaptive
21. Iwase, S., et al. (2025). *ZeroGrasp: Zero-Shot Shape Reconstruction Enabled Robotic Grasping.* CVPR 2025. https://arxiv.org/abs/2504.10857
22. Chisari, E., et al. (2024). *CenterGrasp: Object-Aware Implicit Representation Learning.* RA-L. https://arxiv.org/abs/2312.08240
23. Adaptive Grasp Pose Optimization (2025). *Low-Cost Depth Sensors in Complex Environments.* MDPI Sensors. https://www.mdpi.com/1424-8220/25/3/909

### Other
24. Back, S., et al. (2022). *Unseen Object Amodal Instance Segmentation via Hierarchical Occlusion Modeling.* ICRA 2022. https://arxiv.org/abs/2109.11103
25. Badino, H., et al. (2011). *Fast and Accurate Computation of Surface Normals from Range Images.* ICRA 2011.
