"""Utilities for extracting bounding boxes from binary masks."""

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def bbox_from_mask(mask_path: Path) -> Tuple[int, int, int, int]:
    """Compute tight bounding box coordinates from a mask image.

    Args:
        mask_path: Path to a binary mask image where non-zero pixels indicate the object.

    Returns:
        Bounding box as (x1, y1, x2, y2).
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, 0, 0
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return x1, y1, x2, y2

