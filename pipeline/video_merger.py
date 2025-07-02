"""Functions for merging overlayed frames into a video."""

from pathlib import Path
from typing import List

import cv2


def merge_frames_to_video(frame_paths: List[Path], output_path: Path, fps: int) -> None:
    """Merge image frames into a single mp4 video.

    Args:
        frame_paths: Ordered list of frame image paths.
        output_path: Path to save the mp4 video.
        fps: Target frames per second.
    """
    if not frame_paths:
        raise ValueError("No frames to merge")

    first = cv2.imread(str(frame_paths[0]))
    height, width = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for frame_path in frame_paths:
        img = cv2.imread(str(frame_path))
        writer.write(img)
    writer.release()

