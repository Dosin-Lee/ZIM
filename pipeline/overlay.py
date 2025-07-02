"""Utilities for overlaying segmentation masks onto images."""

from pathlib import Path

import cv2
import numpy as np


def overlay_mask(image_path: Path, mask_path: Path, output_path: Path, alpha: float = 0.5) -> None:
    """Overlay an RGBA mask onto an RGB image and save the result.

    Args:
        image_path: Path to the source image.
        mask_path: Path to the mask image (single channel or RGBA).
        output_path: Where to save the overlay image.
        alpha: Blending factor between 0 and 1.
    """
    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

    if mask is None or image is None:
        raise FileNotFoundError("Image or mask not found")

    if mask.ndim == 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    elif mask.shape[2] == 4:
        mask = mask[:, :, :3]

    mask = mask.astype(float) / 255.0
    image = image.astype(float) / 255.0
    blended = (1 - alpha) * image + alpha * mask
    blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(output_path), blended)

