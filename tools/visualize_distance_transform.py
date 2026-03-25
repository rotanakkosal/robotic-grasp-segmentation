"""
Visualize Distance Transform on a 2D image.

Loads an image, extracts a binary mask via thresholding,
applies cv2.distanceTransform, and shows the results side-by-side.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ── Input image path ──
IMG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "sample_data",
    "Top-Notch-Reflection-600-Ml-20-Oz-Stainless-Steel-Bottle-DW306-6.webp",
)

def main():
    # 1. Load image
    img_bgr = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot load image: {IMG_PATH}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2. Convert to grayscale and threshold to get a binary mask
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # The bottle is dark on a white background → invert so object = 255
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Clean up small noise with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # 3. Apply Distance Transform
    dt = cv2.distanceTransform(binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    # 4. Find the DT maximum (point furthest from all edges)
    _, max_val, _, max_loc = cv2.minMaxLoc(dt)
    dt_cx, dt_cy = max_loc  # (x, y)

    # 5. Normalize DT for visualization
    dt_norm = dt / max_val if max_val > 0 else dt

    # 6. Draw the DT-max point on the original image
    img_marked = img_rgb.copy()
    cv2.circle(img_marked, (dt_cx, dt_cy), 10, (255, 0, 0), -1)
    cv2.circle(img_marked, (dt_cx, dt_cy), 12, (255, 255, 255), 2)

    # ── Visualization ──
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(binary, cmap="gray")
    axes[1].set_title("Binary Mask (threshold)")
    axes[1].axis("off")

    im = axes[2].imshow(dt_norm, cmap="hot")
    axes[2].plot(dt_cx, dt_cy, "co", markersize=10, markeredgecolor="white", markeredgewidth=2)
    axes[2].set_title(f"Distance Transform\nmax={max_val:.1f}px at ({dt_cx}, {dt_cy})")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(img_marked)
    axes[3].set_title("DT Max Point on Image")
    axes[3].axis("off")

    plt.suptitle("Distance Transform Visualization", fontsize=16, fontweight="bold")
    plt.tight_layout()

    # Save output
    out_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "distance_transform_visualization.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to: {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
