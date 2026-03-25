# Copyright (c) 2024. Centroid computation utilities for robot picking applications.
"""
Centroid Computation Utilities for UOAIS Instance Segmentation

This module provides methods to compute object centroids (center points) from
instance segmentation masks. These centroids can be used for robot picking
applications to determine grasp points.

Methods:
    1. Moment-based centroid (center of mass)
    2. Median-based centroid (robust to outliers)
    3. Bounding box center (fastest, less accurate)
    4. 3D centroid using depth data (for robot coordinate transform)
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass
from skimage.morphology import medial_axis


@dataclass
class ObjectCentroid:
    """Data class to store centroid information for a single object."""
    object_id: int
    # 2D centroids (in pixel coordinates)
    centroid_amodal: Tuple[float, float]      # Geometric center of complete object (including occluded)
    centroid_visible: Tuple[float, float]     # Geometric center of visible region only
    centroid_bbox: Tuple[float, float]        # Center of bounding box
    # Grasp point: optimal suction point (may differ from geometric center)
    grasp_point: Optional[Tuple[float, float]] = None  # Best suction grasp point (flattest, highest surface)
    # 3D positions (in camera coordinates, meters)
    centroid_3d: Optional[Tuple[float, float, float]] = None  # 3D geometric center
    grasp_point_3d: Optional[Tuple[float, float, float]] = None  # 3D grasp point
    # Additional info
    bbox: Optional[Tuple[int, int, int, int]] = None  # x1, y1, x2, y2
    is_occluded: bool = False
    confidence: float = 0.0
    mask_area: int = 0
    visible_area: int = 0


@dataclass
class ShapeMetrics:
    """Shape analysis metrics for adaptive centroid method selection."""
    aspect_ratio: float      # width / height (1.0 = square)
    circularity: float       # 4*pi*area / perimeter^2 (1.0 = perfect circle)
    solidity: float          # area / convex_hull_area (1.0 = fully convex)
    is_circular: bool        # circularity > 0.7
    is_elongated: bool       # aspect_ratio > 2.0 or < 0.5


def analyze_mask_shape(mask: np.ndarray) -> Optional[ShapeMetrics]:
    """
    Analyze a binary mask to determine its geometric shape properties.

    Used by the adaptive centroid method to select the best algorithm
    for each specific object shape.

    Args:
        mask: Binary mask [H, W] (bool or uint8)

    Returns:
        ShapeMetrics with shape analysis, or None if mask is empty/too small
    """
    mask_uint8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Use the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)

    if area < 50:  # Too small to analyze meaningfully
        return None

    perimeter = cv2.arcLength(largest_contour, True)

    # Circularity: 4*pi*area / perimeter^2 (1.0 = perfect circle)
    if perimeter > 0:
        circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
    else:
        circularity = 0.0
    circularity = min(circularity, 1.0)  # Clamp to [0, 1]

    # Aspect ratio from rotated bounding box
    if len(largest_contour) >= 5:
        _, (w, h), _ = cv2.minAreaRect(largest_contour)
        if h > 0 and w > 0:
            aspect_ratio = max(w, h) / min(w, h)
        else:
            aspect_ratio = 1.0
    else:
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = max(w, h) / max(min(w, h), 1)

    # Solidity: area / convex hull area (1.0 = fully convex)
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0:
        solidity = area / hull_area
    else:
        solidity = 1.0

    return ShapeMetrics(
        aspect_ratio=aspect_ratio,
        circularity=circularity,
        solidity=solidity,
        is_circular=circularity > 0.7,
        is_elongated=aspect_ratio > 2.0
    )


def compute_centroid_moments(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Compute centroid using image moments (center of mass).

    This is the most accurate method for finding the geometric center,
    but can be affected by noise or irregular shapes.

    Args:
        mask: Binary mask [H, W] (bool or uint8)

    Returns:
        (cx, cy): Centroid coordinates in pixels, or None if mask is empty
    """
    mask_uint8 = (mask > 0).astype(np.uint8)
    M = cv2.moments(mask_uint8)

    if M["m00"] == 0:  # Empty mask
        return None

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    return (cx, cy)


def compute_centroid_median(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Compute centroid using median of pixel coordinates.

    More robust to outliers and noise than moment-based method.
    Recommended for masks with irregular shapes or holes.

    Args:
        mask: Binary mask [H, W] (bool or uint8)

    Returns:
        (cx, cy): Centroid coordinates in pixels, or None if mask is empty
    """
    coords = np.where(mask > 0)

    if len(coords[0]) == 0:  # Empty mask
        return None

    cy = np.median(coords[0])  # row = y
    cx = np.median(coords[1])  # col = x

    return (cx, cy)


def compute_centroid_bbox(bbox: np.ndarray) -> Tuple[float, float]:
    """
    Compute center from bounding box coordinates.

    Most reliable for robot picking - gives geometric center of object extent.

    Args:
        bbox: Bounding box [x1, y1, x2, y2]

    Returns:
        (cx, cy): Center coordinates in pixels
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    return (cx, cy)


def compute_centroid_mask_bbox(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Compute center from the bounding box of the mask (not the predicted bbox).

    This finds the actual extent of the mask pixels and returns the center.
    More accurate than predicted bbox when mask is well-segmented.

    Args:
        mask: Binary mask [H, W] (bool or uint8)

    Returns:
        (cx, cy): Center coordinates in pixels, or None if mask is empty
    """
    coords = np.where(mask > 0)

    if len(coords[0]) == 0:  # Empty mask
        return None

    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()

    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2

    return (cx, cy)


def compute_centroid_distance_transform(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Compute centroid using distance transform - finds the point maximally
    distant from all edges.

    BEST FOR: Suction grippers on flat surfaces, finding the "deepest" stable point.

    Args:
        mask: Binary mask [H, W] (bool or uint8)

    Returns:
        (cx, cy): Center coordinates in pixels, or None if mask is empty
    """
    mask_uint8 = (mask > 0).astype(np.uint8)

    if np.sum(mask_uint8) == 0:
        return None

    # Compute distance transform - each pixel's value is its distance to nearest edge
    dist_transform = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)

    # Find the maximum distance point (furthest from edges)
    _, _, _, max_loc = cv2.minMaxLoc(dist_transform)

    cx, cy = max_loc  # OpenCV returns (x, y)

    return (float(cx), float(cy))


def compute_centroid_top_center(mask: np.ndarray, top_percent: float = 30.0) -> Optional[Tuple[float, float]]:
    """
    Compute centroid at the TOP CENTER of the object.

    BEST FOR: Vacuum suction cup grippers picking from above.
    Finds the center of the topmost region of the object (e.g., bottle cap/lid).

    How it works:
    1. Find the topmost row of the mask (y_min)
    2. Take the top X% of the object height as the "lid region"
    3. Use distance transform within this region to find the most stable suction point
       (point furthest from edges = center of the lid)

    Args:
        mask: Binary mask [H, W] (bool or uint8)
        top_percent: Percentage of object height to consider as "top" (default 30%)

    Returns:
        (cx, cy): Center coordinates of top region in pixels, or None if mask is empty
    """
    mask_uint8 = (mask > 0).astype(np.uint8)
    coords = np.where(mask_uint8 > 0)

    if len(coords[0]) == 0:
        return None

    # Find bounding box of the mask
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()

    # Calculate the height and width of the object
    obj_height = y_max - y_min + 1
    obj_width = x_max - x_min + 1

    # For objects that are wider than tall (like round lids viewed from above),
    # use a larger top_percent to capture more of the visible surface
    aspect_ratio = obj_width / max(obj_height, 1)
    if aspect_ratio > 1.5:
        # Wide object (likely a lid or flat object) - use most of it
        effective_top_percent = 80.0
    elif aspect_ratio > 0.8:
        # Square-ish object - use more of the top
        effective_top_percent = max(top_percent, 40.0)
    else:
        # Tall object (like a bottle) - use specified top_percent
        effective_top_percent = top_percent

    # Define the "top region" (top X% of the object)
    top_region_height = max(int(obj_height * effective_top_percent / 100.0), 10)  # At least 10 pixels
    top_y_max = y_min + top_region_height

    # Create mask for top region only
    top_mask = np.zeros_like(mask_uint8)
    top_mask[y_min:top_y_max, :] = mask_uint8[y_min:top_y_max, :]

    # Find pixels in the top region
    top_coords = np.where(top_mask > 0)

    if len(top_coords[0]) == 0:
        # Fallback to regular mask_bbox if top region is empty
        return compute_centroid_mask_bbox(mask)

    # Use distance transform to find the CENTER of the top region
    # The point with maximum distance from all edges is the most central/stable point
    dist_transform = cv2.distanceTransform(top_mask, cv2.DIST_L2, 5)

    # Find the point with maximum distance from edges in top region
    _, max_dist, _, max_loc = cv2.minMaxLoc(dist_transform)

    if max_dist > 3:  # Need at least 3 pixels of clearance for a meaningful center
        # Use the distance transform result - this is the CENTER of the top region
        cx, cy = max_loc
    else:
        # Fallback: use geometric center of top region bounding box
        top_x_min, top_x_max = top_coords[1].min(), top_coords[1].max()
        top_y_min, top_y_max_actual = top_coords[0].min(), top_coords[0].max()
        cx = (top_x_min + top_x_max) / 2
        cy = (top_y_min + top_y_max_actual) / 2

    return (float(cx), float(cy))


def compute_centroid_skeleton_dt(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Compute centroid using medial axis skeleton combined with distance transform.

    Based on the 2D Skeleton-Based Keypoint Generation method (ICRoM 2023, 98% success).
    Finds the skeleton point with the maximum distance transform value — this is the
    structural center of the object that is also maximally distant from all edges.

    BEST FOR: Elongated objects, irregular shapes, objects with complex topology.
    Also works well for circular objects (skeleton center = geometric center).

    Algorithm:
        1. Compute medial axis skeleton (captures object topology/symmetry)
        2. Compute distance transform of the original mask (boundary distance)
        3. Mask the distance transform with the skeleton
        4. The skeleton point with highest DT value = optimal grasp point

    Args:
        mask: Binary mask [H, W] (bool or uint8)

    Returns:
        (cx, cy): Centroid coordinates in pixels, or None if mask is empty
    """
    mask_bool = mask > 0

    if np.sum(mask_bool) < 10:  # Too small for skeleton
        return compute_centroid_distance_transform(mask)

    # Compute medial axis skeleton
    skeleton = medial_axis(mask_bool)

    if np.sum(skeleton) == 0:
        # Skeleton empty — fall back to distance transform
        return compute_centroid_distance_transform(mask)

    # Compute distance transform of the original mask
    mask_uint8 = mask_bool.astype(np.uint8)
    dist_transform = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)

    # Mask the distance transform with the skeleton:
    # only consider skeleton points, scored by their distance from edges
    skeleton_dt = dist_transform * skeleton.astype(np.float32)

    # Find the skeleton point with maximum boundary distance
    _, _, _, max_loc = cv2.minMaxLoc(skeleton_dt)
    cx, cy = max_loc  # OpenCV returns (x, y)

    return (float(cx), float(cy))


def compute_centroid_adaptive(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Shape-adaptive centroid computation that finds the true geometric center
    of an object while ensuring the point is inside the mask and away from edges.

    Key insight: distance_transform finds the point furthest from edges, which
    for asymmetric shapes (bottles) biases toward the WIDEST part, not the
    geometric center. Moments (center of mass) gives the true geometric center
    but may fall outside concave masks.

    Strategy:
    - Circular objects → distance_transform (DT center = geometric center)
    - All other shapes → moments (geometric center), validated against mask
      If moments point is outside mask or too close to edge, blend toward DT max

    Based on literature:
    - DYNAMO-GRASP (CoRL 2023): center-of-mass proximity matters for transport
    - Cartman (ARC 2017): boundary distance for seal quality
    - Combined: geometric center + edge clearance check

    Args:
        mask: Binary mask [H, W] (bool or uint8)

    Returns:
        (cx, cy): Optimal centroid coordinates in pixels, or None if mask is empty
    """
    mask_uint8 = (mask > 0).astype(np.uint8)

    if np.sum(mask_uint8) == 0:
        return None

    # Analyze shape properties
    shape = analyze_mask_shape(mask)

    if shape is None:
        # Mask too small for shape analysis — use moments
        return compute_centroid_moments(mask)

    # For circular objects, DT center = geometric center (no asymmetry bias)
    if shape.is_circular:
        result = compute_centroid_distance_transform(mask)
        if result is not None:
            return result

    # For all other shapes: use moments (geometric center) as primary
    moments_result = compute_centroid_moments(mask)
    if moments_result is None:
        return compute_centroid_distance_transform(mask)

    mx, my = moments_result
    mx_int, my_int = int(round(mx)), int(round(my))

    # Compute distance transform for edge clearance check
    dist_transform = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
    max_dt = np.max(dist_transform)

    if max_dt == 0:
        return moments_result

    # Check if moments centroid is inside the mask and has reasonable edge clearance
    H, W = mask.shape
    mx_clamped = np.clip(mx_int, 0, W - 1)
    my_clamped = np.clip(my_int, 0, H - 1)
    moments_inside_mask = mask_uint8[my_clamped, mx_clamped] > 0
    moments_edge_dist = dist_transform[my_clamped, mx_clamped] if moments_inside_mask else 0.0

    # Minimum edge clearance: at least 20% of the max inscribed radius
    min_clearance = max_dt * 0.20

    if moments_inside_mask and moments_edge_dist >= min_clearance:
        # Moments centroid is inside mask with good edge clearance — use it
        return moments_result

    # Moments centroid is outside mask or too close to edge
    # Find the DT maximum (furthest from edges, guaranteed inside mask)
    _, _, _, dt_max_loc = cv2.minMaxLoc(dist_transform)
    dt_cx, dt_cy = float(dt_max_loc[0]), float(dt_max_loc[1])

    if not moments_inside_mask:
        # For concave shapes where moments falls outside the mask,
        # find the closest mask point to the moments centroid, then
        # blend it with DT max for edge safety
        mask_coords = np.where(mask_uint8 > 0)
        if len(mask_coords[0]) == 0:
            return (dt_cx, dt_cy)
        distances = (mask_coords[1] - mx)**2 + (mask_coords[0] - my)**2
        nearest_idx = np.argmin(distances)
        nearest_x = float(mask_coords[1][nearest_idx])
        nearest_y = float(mask_coords[0][nearest_idx])
        # Blend: 50% nearest-to-moments + 50% DT max
        cx = 0.5 * nearest_x + 0.5 * dt_cx
        cy = 0.5 * nearest_y + 0.5 * dt_cy
    else:
        # Inside mask but too close to edge — blend toward DT max
        # Blend ratio based on how close to edge (closer to edge = more DT influence)
        edge_ratio = moments_edge_dist / max_dt  # 0.0 = at edge, 1.0 = at DT max
        dt_weight = 1.0 - (edge_ratio / 0.20)  # More DT influence when closer to edge
        dt_weight = np.clip(dt_weight, 0.0, 0.7)
        cx = (1.0 - dt_weight) * mx + dt_weight * dt_cx
        cy = (1.0 - dt_weight) * my + dt_weight * dt_cy

    # Final safety check: ensure the result is inside the mask
    cx_int, cy_int = int(round(cx)), int(round(cy))
    cx_int = np.clip(cx_int, 0, W - 1)
    cy_int = np.clip(cy_int, 0, H - 1)
    if mask_uint8[cy_int, cx_int] == 0:
        # Blended point fell outside mask — use DT max as safe fallback
        return (dt_cx, dt_cy)

    return (float(cx), float(cy))


def compute_surface_normals(depth_image: np.ndarray) -> np.ndarray:
    """
    Compute surface normals from a depth image.

    Based on the cross-product method from:
    "Fast and Accurate Computation of Surface Normals from Range Images"
    (Badino et al.)

    Args:
        depth_image: Depth image [H, W] in any unit (mm or m)

    Returns:
        normals: Surface normal map [H, W, 3] with (nx, ny, nz) per pixel
    """
    # Ensure float type
    depth = depth_image.astype(np.float32)

    # Compute gradients using Sobel operators for smoothing
    # dz/dx and dz/dy
    dzdx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=5)
    dzdy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=5)

    # Normal vector: (-dz/dx, -dz/dy, 1) then normalize
    # The normal points "outward" from the surface (toward camera for top surfaces)
    normals = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.float32)
    normals[:, :, 0] = -dzdx
    normals[:, :, 1] = -dzdy
    normals[:, :, 2] = 1.0

    # Normalize to unit vectors
    magnitude = np.sqrt(np.sum(normals ** 2, axis=2, keepdims=True))
    magnitude[magnitude == 0] = 1  # Avoid division by zero
    normals = normals / magnitude

    return normals


def compute_normal_variance(normals: np.ndarray, mask: np.ndarray,
                           window_size: int = 15) -> np.ndarray:
    """
    Compute local variance of surface normals within a window.

    Lower variance = flatter surface = better for suction.

    Based on grasp quality metric from:
    "Learning Suction Graspability Considering Grasp Quality and Robot
    Reachability for Bin-Picking" (Frontiers in Neurorobotics, 2022)

    Formula: var(ns) = Σ||ns,i - n̄s||² / (N-1)

    Args:
        normals: Surface normal map [H, W, 3]
        mask: Binary object mask [H, W]
        window_size: Size of local window for variance computation

    Returns:
        variance_map: Normal variance at each pixel [H, W] (lower = flatter)
    """
    H, W = mask.shape
    half_win = window_size // 2

    # Compute variance for each normal component
    variance_map = np.zeros((H, W), dtype=np.float32)

    # Use box filter to compute local mean of each normal component
    kernel = np.ones((window_size, window_size), np.float32) / (window_size ** 2)

    for i in range(3):
        ni = normals[:, :, i]
        # Local mean
        mean_ni = cv2.filter2D(ni, -1, kernel)
        # Local variance: E[X²] - E[X]²
        mean_ni_sq = cv2.filter2D(ni ** 2, -1, kernel)
        var_ni = mean_ni_sq - mean_ni ** 2
        variance_map += var_ni

    # Apply mask - set variance to high value outside object
    variance_map[mask == 0] = 999.0

    return variance_map


def compute_suction_grasp_point(
    mask: np.ndarray,
    depth_image: np.ndarray,
    suction_cup_radius: int = 15,
    top_k_candidates: int = 10,
    prefer_center: bool = True,
    center_weight: float = 0.3,
    approach_direction: str = "top"
) -> Optional[Tuple[float, float]]:
    """
    Compute optimal suction grasp point using surface normal analysis.

    Based on research from:
    - Dex-Net 3.0 (Mahler et al., ICRA 2018): Seal formation and wrench resistance
    - Learning Suction Graspability (Frontiers, 2022): Normal variance for flatness
    - Amazon Robotics Challenge approaches: Centroid + surface normal combination

    Algorithm:
    1. Determine the graspable region based on approach direction (top surface for top-down picking)
    2. Compute surface normals from depth image
    3. Compute local normal variance (flatness measure)
    4. Find regions large enough for suction cup (using distance transform)
    5. Score candidates by: flatness + distance from edges + proximity to region center
    6. Return the best grasp point

    Args:
        mask: Binary object mask [H, W]
        depth_image: Depth image [H, W] (raw depth values)
        suction_cup_radius: Radius of suction cup in pixels (for contact area check)
        top_k_candidates: Number of top candidates to consider
        prefer_center: If True, prefer points closer to object center
        center_weight: Weight for center proximity (0-1)
        approach_direction: Direction robot approaches from:
                           - "top": Robot picks from above (default for vacuum suction)
                                   Uses DEPTH to find the highest surface (closest to camera)
                           - "any": Consider the entire mask

    Returns:
        (cx, cy): Optimal suction grasp point, or None if no valid point found
    """
    mask_uint8 = (mask > 0).astype(np.uint8)

    if np.sum(mask_uint8) == 0:
        return None

    # Step 0: For top-down picking, identify the TOP SURFACE using depth
    # The top surface is the part of the object closest to the camera (smallest depth values)
    if approach_direction == "top":
        # Get depth values within the mask
        masked_depth = depth_image.copy().astype(np.float32)
        masked_depth[mask_uint8 == 0] = np.inf  # Ignore non-mask areas

        # Find valid depth values (non-zero, non-inf)
        valid_mask = (masked_depth > 0) & (masked_depth < np.inf)
        if not np.any(valid_mask):
            # No valid depth, fall back to shape-adaptive method
            return compute_centroid_adaptive(mask)

        # Find the minimum depth (closest to camera = top surface)
        min_depth = np.min(masked_depth[valid_mask])
        max_depth = np.max(masked_depth[valid_mask])
        depth_range = max_depth - min_depth

        # Check if depth has meaningful variation (not dummy/uniform depth)
        # For real depth, bottles typically have 50-200mm depth range
        # For dummy depth, all values are the same (range = 0 or near 0)
        min_meaningful_depth_range = 10.0  # 10mm minimum to consider depth meaningful

        if depth_range > min_meaningful_depth_range:
            # Real depth data - use depth-based top surface detection
            # Create a mask for the TOP SURFACE: points within 20% of the depth range from the top
            # This captures the lid/cap area of bottles and top surfaces of objects
            top_surface_threshold = min_depth + depth_range * 0.20
            top_surface_mask = ((masked_depth <= top_surface_threshold) & valid_mask).astype(np.uint8)

            # Ensure top surface mask has enough pixels for suction cup
            top_surface_area = np.sum(top_surface_mask)
            min_required_area = np.pi * suction_cup_radius * suction_cup_radius * 0.5

            if top_surface_area >= min_required_area:
                # Use top surface mask for further processing
                working_mask = top_surface_mask
            else:
                # Top surface too small, gradually increase threshold
                for threshold_pct in [0.30, 0.40, 0.50]:
                    threshold = min_depth + depth_range * threshold_pct
                    top_surface_mask = ((masked_depth <= threshold) & valid_mask).astype(np.uint8)
                    if np.sum(top_surface_mask) >= min_required_area:
                        working_mask = top_surface_mask
                        break
                else:
                    # Still not enough, use full mask
                    working_mask = mask_uint8
        else:
            # No meaningful depth variation (dummy/uniform depth)
            # Fall back to shape-adaptive method (distance_transform for circular, skeleton for complex)
            return compute_centroid_adaptive(mask)
    else:
        working_mask = mask_uint8

    # Step 1: Compute surface normals from depth
    normals = compute_surface_normals(depth_image)

    # Step 2: Compute normal variance (flatness measure)
    # Lower variance = flatter surface = better for suction
    variance_map = compute_normal_variance(normals, working_mask, window_size=suction_cup_radius * 2)

    # Step 3: Distance transform on the WORKING MASK to find points with enough clearance
    dist_transform = cv2.distanceTransform(working_mask, cv2.DIST_L2, 5)

    # Step 4: Compute center of the WORKING region (not full object) for center preference
    coords = np.where(working_mask > 0)
    if len(coords[0]) == 0:
        return compute_centroid_adaptive(mask)

    center_y = np.mean(coords[0])
    center_x = np.mean(coords[1])

    # Create distance-from-center map (normalized to 0-1)
    y_coords, x_coords = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
    dist_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    max_dist = np.max(dist_from_center[working_mask > 0]) if np.any(working_mask > 0) else 1.0
    if max_dist > 0:
        dist_from_center_norm = dist_from_center / max_dist
    else:
        dist_from_center_norm = np.zeros_like(dist_from_center)

    # Step 5: Combine scores
    # Normalize variance to 0-1 range within the working mask
    var_in_mask = variance_map[working_mask > 0]
    if len(var_in_mask) > 0 and np.max(var_in_mask) > np.min(var_in_mask):
        var_min, var_max = np.min(var_in_mask), np.max(var_in_mask)
        variance_norm = (variance_map - var_min) / (var_max - var_min + 1e-6)
    else:
        variance_norm = np.zeros_like(variance_map)

    # Normalize distance transform
    max_dist_transform = np.max(dist_transform)
    if max_dist_transform > 0:
        dist_norm = dist_transform / max_dist_transform
    else:
        dist_norm = np.zeros_like(dist_transform)

    # Combined score (higher is better)
    flatness_score = 1.0 - variance_norm
    edge_score = dist_norm
    center_score = 1.0 - dist_from_center_norm

    # Final score with weights
    if prefer_center:
        combined_score = (
            0.4 * flatness_score +
            0.3 * edge_score +
            center_weight * center_score
        )
    else:
        combined_score = 0.5 * flatness_score + 0.5 * edge_score

    # Apply working mask and minimum clearance requirement
    combined_score[working_mask == 0] = -999
    combined_score[dist_transform < suction_cup_radius * 0.3] = -999  # Need some clearance

    # Find the best point
    best_loc = np.unravel_index(np.argmax(combined_score), combined_score.shape)
    cy, cx = best_loc

    # Verify it's a valid point
    if combined_score[cy, cx] <= -999:
        # Fallback to shape-adaptive method
        return compute_centroid_adaptive(mask)

    return (float(cx), float(cy))


def compute_centroid_min_enclosing_circle(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Compute centroid as the center of the minimum enclosing circle.

    BEST FOR: Roughly circular or compact objects.

    Args:
        mask: Binary mask [H, W] (bool or uint8)

    Returns:
        (cx, cy): Center coordinates in pixels, or None if mask is empty
    """
    mask_uint8 = (mask > 0).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    if len(largest_contour) < 3:
        return None

    # Fit minimum enclosing circle
    (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)

    return (float(cx), float(cy))


def compute_centroid_ellipse_fit(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Compute centroid by fitting an ellipse to the mask contour.

    BEST FOR: Elongated objects (bottles, tools, etc.)
    Returns the center of the fitted ellipse.

    Args:
        mask: Binary mask [H, W] (bool or uint8)

    Returns:
        (cx, cy): Center coordinates in pixels, or None if mask is empty or too small
    """
    mask_uint8 = (mask > 0).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Need at least 5 points to fit ellipse
    if len(largest_contour) < 5:
        return compute_centroid_moments(mask)  # Fallback

    # Fit ellipse
    ellipse = cv2.fitEllipse(largest_contour)
    (cx, cy), (MA, ma), angle = ellipse

    return (float(cx), float(cy))


def compute_centroid_3d(
    centroid_2d: Tuple[float, float],
    depth_image: np.ndarray,
    camera_intrinsics: Optional[Dict[str, float]] = None,
    depth_scale: float = 0.001,  # Convert mm to meters
    window_size: int = 5
) -> Optional[Tuple[float, float, float]]:
    """
    Compute 3D centroid in camera coordinates using depth information.

    Projects the 2D centroid to 3D space using camera intrinsics and depth.
    Uses a window around the centroid to get a more stable depth reading.

    Args:
        centroid_2d: (cx, cy) pixel coordinates
        depth_image: Depth image [H, W] in millimeters (raw depth, not normalized)
        camera_intrinsics: Dict with 'fx', 'fy', 'cx', 'cy' (focal lengths and principal point)
                          If None, uses default RealSense D435 intrinsics for 640x480
        depth_scale: Scale factor to convert depth to meters (default: 0.001 for mm to m)
        window_size: Size of window around centroid for depth averaging

    Returns:
        (X, Y, Z): 3D coordinates in camera frame (meters), or None if depth invalid
    """
    # Default camera intrinsics for RealSense D435 at 640x480
    if camera_intrinsics is None:
        camera_intrinsics = {
            'fx': 612.0,  # Focal length x
            'fy': 612.0,  # Focal length y
            'cx': 320.0,  # Principal point x
            'cy': 240.0   # Principal point y
        }

    cx_pixel, cy_pixel = centroid_2d
    cx_pixel, cy_pixel = int(round(cx_pixel)), int(round(cy_pixel))

    H, W = depth_image.shape[:2]

    # Ensure coordinates are within bounds
    cx_pixel = np.clip(cx_pixel, 0, W - 1)
    cy_pixel = np.clip(cy_pixel, 0, H - 1)

    # Get depth value with windowed averaging for stability
    half_win = window_size // 2
    y_min = max(0, cy_pixel - half_win)
    y_max = min(H, cy_pixel + half_win + 1)
    x_min = max(0, cx_pixel - half_win)
    x_max = min(W, cx_pixel + half_win + 1)

    depth_window = depth_image[y_min:y_max, x_min:x_max]

    # Filter out invalid depth values (0 or very large)
    valid_depths = depth_window[(depth_window > 0) & (depth_window < 10000)]

    if len(valid_depths) == 0:
        return None

    # Use median for robustness
    Z = np.median(valid_depths) * depth_scale  # Convert to meters

    if Z <= 0:
        return None

    # Back-project to 3D using pinhole camera model
    fx = camera_intrinsics['fx']
    fy = camera_intrinsics['fy']
    cx_cam = camera_intrinsics['cx']
    cy_cam = camera_intrinsics['cy']

    X = (cx_pixel - cx_cam) * Z / fx
    Y = (cy_pixel - cy_cam) * Z / fy

    return (X, Y, Z)


def compute_all_centroids(
    pred_masks: np.ndarray,
    pred_visible_masks: np.ndarray,
    pred_boxes: np.ndarray,
    pred_occlusions: np.ndarray,
    scores: np.ndarray,
    depth_image: Optional[np.ndarray] = None,
    camera_intrinsics: Optional[Dict[str, float]] = None,
    method: str = "suction",
    use_amodal_for_centroid: bool = True,
    suction_cup_radius: int = 15
) -> List[ObjectCentroid]:
    """
    Compute centroids for all detected objects.

    Args:
        pred_masks: Amodal masks [N, H, W] (complete object including occluded parts)
        pred_visible_masks: Visible masks [N, H, W] (only visible parts)
        pred_boxes: Bounding boxes [N, 4] (x1, y1, x2, y2)
        pred_occlusions: Occlusion flags [N] (0=not occluded, 1=occluded)
        scores: Confidence scores [N]
        depth_image: Optional raw depth image [H, W] in mm for 3D computation
        camera_intrinsics: Optional camera parameters for 3D projection
        method: Centroid computation method:
                - "suction": Surface-normal based optimal suction point (BEST for vacuum grippers)
                             Uses flatness + edge clearance + center proximity
                             Based on Dex-Net 3.0 and Frontiers 2022 research
                - "top_center": Center of top region (for picking lids/caps from above)
                - "distance_transform": Point furthest from edges
                - "ellipse": Center of fitted ellipse (for elongated objects)
                - "circle": Center of minimum enclosing circle (for round objects)
                - "mask_bbox": Center of mask's bounding box
                - "bbox": Center of predicted bounding box
                - "moments": Center of mass
                - "median": Median of coordinates
        use_amodal_for_centroid: If True, compute grasp centroid from amodal (complete) mask.
                                 If False, use visible mask only. Default True for better picking.
        suction_cup_radius: Radius of suction cup in pixels (for "suction" method)

    Returns:
        List of ObjectCentroid objects with all centroid information
    """
    centroids = []
    num_objects = len(pred_masks)

    # Select centroid computation function based on method
    method_functions = {
        "distance_transform": compute_centroid_distance_transform,
        "top_center": compute_centroid_top_center,
        "adaptive": compute_centroid_adaptive,
        "skeleton_dt": compute_centroid_skeleton_dt,
        "ellipse": compute_centroid_ellipse_fit,
        "circle": compute_centroid_min_enclosing_circle,
        "mask_bbox": compute_centroid_mask_bbox,
        "moments": compute_centroid_moments,
        "median": compute_centroid_median,
    }

    for i in range(num_objects):
        amodal_mask = pred_masks[i]
        visible_mask = pred_visible_masks[i]
        bbox = pred_boxes[i]
        is_occluded = bool(pred_occlusions[i])
        confidence = float(scores[i]) if i < len(scores) else 0.0

        # Compute predicted bbox center (always available as fallback)
        centroid_bbox = compute_centroid_bbox(bbox)

        # --- GEOMETRIC CENTER (where the object IS) ---
        # Always compute using adaptive method (moments-based, shape-aware)
        centroid_amodal = compute_centroid_adaptive(amodal_mask)
        centroid_visible = compute_centroid_adaptive(visible_mask)

        # If a non-default method is explicitly chosen, override
        if method == "bbox":
            centroid_amodal = centroid_bbox
            centroid_visible = centroid_bbox
        elif method not in ("suction", "adaptive"):
            compute_fn = method_functions.get(method, compute_centroid_distance_transform)
            centroid_amodal = compute_fn(amodal_mask)
            centroid_visible = compute_fn(visible_mask)

        # Fallback chain: visible -> amodal -> bbox
        if centroid_amodal is None:
            centroid_amodal = centroid_bbox
        if centroid_visible is None:
            centroid_visible = centroid_amodal

        # --- SUCTION GRASP POINT (where to GRIP) ---
        # Separate from geometric center — uses depth-based surface analysis
        grasp_point = None
        if depth_image is not None:
            pick_mask = amodal_mask if use_amodal_for_centroid else visible_mask
            grasp_point = compute_suction_grasp_point(
                pick_mask, depth_image,
                suction_cup_radius=suction_cup_radius,
                prefer_center=True
            )
        # Fallback: if no depth or suction failed, use geometric center
        if grasp_point is None:
            grasp_point = centroid_amodal if use_amodal_for_centroid else centroid_visible

        # Select the primary centroid for the object
        primary_centroid = centroid_amodal if use_amodal_for_centroid else centroid_visible

        # Compute 3D positions if depth is available
        centroid_3d = None
        grasp_point_3d = None
        if depth_image is not None:
            centroid_3d = compute_centroid_3d(
                primary_centroid,
                depth_image,
                camera_intrinsics
            )
            if grasp_point != primary_centroid:
                grasp_point_3d = compute_centroid_3d(
                    grasp_point,
                    depth_image,
                    camera_intrinsics
                )
            else:
                grasp_point_3d = centroid_3d

        # Create ObjectCentroid instance
        obj_centroid = ObjectCentroid(
            object_id=i,
            centroid_amodal=centroid_amodal,
            centroid_visible=centroid_visible,
            centroid_bbox=centroid_bbox,
            grasp_point=grasp_point,
            centroid_3d=centroid_3d,
            grasp_point_3d=grasp_point_3d,
            bbox=tuple(bbox.astype(int)),
            is_occluded=is_occluded,
            confidence=confidence,
            mask_area=int(np.sum(amodal_mask)),
            visible_area=int(np.sum(visible_mask))
        )

        centroids.append(obj_centroid)

    return centroids


def draw_centroids(
    image: np.ndarray,
    centroids: List[ObjectCentroid],
    draw_amodal: bool = True,
    draw_grasp: bool = True,
    draw_bbox_center: bool = False,
    draw_bbox: bool = False,
    draw_labels: bool = True,
    draw_visible: bool = True,
    center_color: Tuple[int, int, int] = (0, 0, 255),    # Red - geometric center
    grasp_color: Tuple[int, int, int] = (0, 255, 0),     # Green - suction grasp point
    bbox_color: Tuple[int, int, int] = (255, 255, 0),    # Cyan
    radius: int = 6,
    thickness: int = 2
) -> np.ndarray:
    """
    Draw centroids on an image for visualization.

    Shows TWO points per object:
    - RED filled circle: Geometric center (where the object IS)
    - GREEN crosshair: Suction grasp point (where to GRIP)
    When no depth is available, both points coincide.

    Args:
        image: Input image [H, W, 3] (BGR)
        centroids: List of ObjectCentroid objects
        draw_amodal: Draw geometric center marker
        draw_grasp: Draw suction grasp point marker
        draw_bbox_center: Draw bounding box center
        draw_bbox: Draw bounding box rectangle
        draw_labels: Draw object ID and info labels
        draw_visible: (legacy, ignored — use draw_amodal/draw_grasp)
        center_color: Color for geometric center (BGR)
        grasp_color: Color for grasp point (BGR)
        bbox_color: Color for bbox and bbox center (BGR)
        radius: Circle radius for centroid markers
        thickness: Line thickness for markers

    Returns:
        Image with centroids drawn
    """
    vis_image = image.copy()

    for obj in centroids:
        # Draw bounding box (only if enabled)
        if draw_bbox and obj.bbox is not None:
            x1, y1, x2, y2 = obj.bbox
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), bbox_color, 1)

        # Draw bbox center
        if draw_bbox_center and obj.centroid_bbox is not None:
            cx, cy = int(obj.centroid_bbox[0]), int(obj.centroid_bbox[1])
            cv2.circle(vis_image, (cx, cy), radius, bbox_color, thickness)

        # Draw line connecting center to grasp point (if they differ)
        if draw_amodal and draw_grasp and obj.centroid_amodal is not None and obj.grasp_point is not None:
            c_cx, c_cy = int(obj.centroid_amodal[0]), int(obj.centroid_amodal[1])
            g_cx, g_cy = int(obj.grasp_point[0]), int(obj.grasp_point[1])
            dist = np.sqrt((c_cx - g_cx)**2 + (c_cy - g_cy)**2)
            if dist > 5:  # Only draw line if points are far enough apart
                cv2.line(vis_image, (c_cx, c_cy), (g_cx, g_cy), (255, 255, 255), 1, cv2.LINE_AA)

        # Draw geometric center (RED filled circle)
        if draw_amodal and obj.centroid_amodal is not None:
            cx, cy = int(obj.centroid_amodal[0]), int(obj.centroid_amodal[1])
            cv2.circle(vis_image, (cx, cy), radius, center_color, -1)  # Filled
            cv2.circle(vis_image, (cx, cy), radius, (255, 255, 255), 1)  # White border

        # Draw suction grasp point (GREEN crosshair)
        if draw_grasp and obj.grasp_point is not None:
            gx, gy = int(obj.grasp_point[0]), int(obj.grasp_point[1])
            cv2.drawMarker(vis_image, (gx, gy), grasp_color, cv2.MARKER_CROSS, radius*3, 2)
            cv2.circle(vis_image, (gx, gy), radius - 1, grasp_color, -1)

        # Draw labels (anchored to geometric center)
        if draw_labels and obj.centroid_amodal is not None:
            cx, cy = int(obj.centroid_amodal[0]), int(obj.centroid_amodal[1])

            # Object ID
            label = f"#{obj.object_id}"
            if obj.is_occluded:
                label += " (occ)"

            # 3D position if available
            if obj.centroid_3d is not None:
                X, Y, Z = obj.centroid_3d
                label += f" Z:{Z:.2f}m"

            # Draw label background
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_image, (cx - 2, cy - text_h - 8), (cx + text_w + 2, cy - 4), (0, 0, 0), -1)
            cv2.putText(vis_image, label, (cx, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return vis_image


def get_best_grasp_point(
    centroids: List[ObjectCentroid],
    prefer_unoccluded: bool = True,
    min_confidence: float = 0.5,
    min_visible_ratio: float = 0.3
) -> Optional[ObjectCentroid]:
    """
    Select the best object to grasp based on various criteria.

    Args:
        centroids: List of ObjectCentroid objects
        prefer_unoccluded: Prefer objects that are not occluded
        min_confidence: Minimum confidence threshold
        min_visible_ratio: Minimum ratio of visible to total area

    Returns:
        Best ObjectCentroid for grasping, or None if no valid candidates
    """
    valid_candidates = []

    for obj in centroids:
        # Filter by confidence
        if obj.confidence < min_confidence:
            continue

        # Filter by visible ratio
        if obj.mask_area > 0:
            visible_ratio = obj.visible_area / obj.mask_area
            if visible_ratio < min_visible_ratio:
                continue

        valid_candidates.append(obj)

    if not valid_candidates:
        return None

    # Sort by preference
    if prefer_unoccluded:
        # First priority: not occluded
        # Second priority: larger visible area
        # Third priority: higher confidence
        valid_candidates.sort(
            key=lambda x: (not x.is_occluded, x.visible_area, x.confidence),
            reverse=True
        )
    else:
        # Sort by visible area and confidence
        valid_candidates.sort(
            key=lambda x: (x.visible_area, x.confidence),
            reverse=True
        )

    return valid_candidates[0]


def _to_json_serializable(obj):
    """Convert numpy scalars / arrays and nested tuples to JSON-safe Python types."""
    if obj is None:
        return None
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return [_to_json_serializable(x) for x in obj]
    if isinstance(obj, list):
        return [_to_json_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    return obj


def centroids_to_dict(centroids: List[ObjectCentroid]) -> List[Dict]:
    """
    Convert list of ObjectCentroid to list of dictionaries for JSON serialization.

    Args:
        centroids: List of ObjectCentroid objects

    Returns:
        List of dictionaries
    """
    result = []
    for obj in centroids:
        d = {
            'object_id': obj.object_id,
            'centroid_amodal': obj.centroid_amodal,
            'centroid_visible': obj.centroid_visible,
            'centroid_bbox': obj.centroid_bbox,
            'centroid_3d': obj.centroid_3d,
            'bbox': obj.bbox,
            'is_occluded': obj.is_occluded,
            'confidence': obj.confidence,
            'mask_area': obj.mask_area,
            'visible_area': obj.visible_area
        }
        result.append(_to_json_serializable(d))
    return result


# Convenience function for quick centroid extraction
def extract_centroids_from_instances(instances, depth_image=None, camera_intrinsics=None):
    """
    Extract centroids directly from UOAIS instances output.

    Args:
        instances: Detectron2 Instances object from UOAIS prediction
        depth_image: Optional raw depth image [H, W] in mm
        camera_intrinsics: Optional camera parameters

    Returns:
        List of ObjectCentroid objects
    """
    pred_masks = instances.pred_masks.cpu().numpy()
    pred_visible_masks = instances.pred_visible_masks.cpu().numpy()
    pred_boxes = instances.pred_boxes.tensor.cpu().numpy()
    pred_occlusions = instances.pred_occlusions.cpu().numpy()
    scores = instances.scores.cpu().numpy() if hasattr(instances, 'scores') else np.ones(len(pred_masks))

    return compute_all_centroids(
        pred_masks=pred_masks,
        pred_visible_masks=pred_visible_masks,
        pred_boxes=pred_boxes,
        pred_occlusions=pred_occlusions,
        scores=scores,
        depth_image=depth_image,
        camera_intrinsics=camera_intrinsics
    )
