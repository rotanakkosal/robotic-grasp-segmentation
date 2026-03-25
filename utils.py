import cv2
import numpy as np


def normalize_depth(depth, min_val=250.0, max_val=1500.0):
    """Normalize depth (mm) to uint8 [H, W, 3] in 0–255."""
    depth = depth.copy()
    depth[depth < min_val] = min_val
    depth[depth > max_val] = max_val
    depth = (depth - min_val) / (max_val - min_val) * 255
    depth = np.expand_dims(depth, -1)
    depth = np.uint8(np.repeat(depth, 3, -1))
    return depth


def unnormalize_depth(depth, min_val=250.0, max_val=1500.0):
    """Invert normalize_depth: uint8 [H, W, 3] → mm (uses first channel)."""
    depth = np.float32(depth[..., 0]) / 255
    depth = depth * (max_val - min_val) + min_val
    return depth


def inpaint_depth(depth, factor=1, kernel_size=3, dilate=False):
    """Inpaint zero regions in normalized depth [H, W, 3] uint8."""
    H, W, _ = depth.shape
    resized_depth = cv2.resize(depth, (W // factor, H // factor))
    mask = np.all(resized_depth == 0, axis=2).astype(np.uint8)
    if dilate:
        mask = cv2.dilate(mask, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
    inpainted_data = cv2.inpaint(resized_depth, mask, kernel_size, cv2.INPAINT_TELEA)
    inpainted_data = cv2.resize(inpainted_data, (W, H))
    depth = np.where(depth == 0, inpainted_data, depth)
    return depth
