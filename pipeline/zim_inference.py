"""Wrapper for running ZIM ONNX inference."""

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import onnxruntime as ort


class ZimInferencer:
    """Load ZIM ONNX models and run inference on frames."""

    def __init__(self, encoder_path: Path, decoder_path: Path):
        self.encoder_session = ort.InferenceSession(str(encoder_path))
        self.decoder_session = ort.InferenceSession(str(decoder_path))

    def infer(self, image_path: Path, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Run ZIM inference on an image with a bounding box prompt.

        Args:
            image_path: Path to the RGB image.
            bbox: Bounding box (x1, y1, x2, y2).

        Returns:
            Binary mask as a NumPy array.
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)

        # Placeholder pre-processing. Actual ZIM pre-process should be implemented.
        input_image = image.transpose(2, 0, 1)[None].astype(np.float32)
        input_bbox = np.array(bbox, dtype=np.float32)[None]

        latent = self.encoder_session.run(None, {"image": input_image, "bbox": input_bbox})[0]
        mask = self.decoder_session.run(None, {"latent": latent})[0]
        mask = (mask.squeeze() > 0.5).astype(np.uint8) * 255
        return mask

