"""Utilities for splitting videos into individual frames."""

from pathlib import Path
from typing import List

import cv2


def split_video_to_frames(video_path: Path, output_dir: Path, prefix: str = "frame_", ext: str = ".jpg") -> List[Path]:
    """Split an mp4 video into frame images.

    Args:
        video_path: Path to the input mp4 video.
        output_dir: Directory to save frame images.
        prefix: Filename prefix for saved frames.
        ext: Image extension.

    Returns:
        List of saved frame paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    frame_paths = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fname = f"{prefix}{idx:04d}{ext}"
        fpath = output_dir / fname
        cv2.imwrite(str(fpath), frame)
        frame_paths.append(fpath)
        idx += 1
    cap.release()
    return frame_paths

