"""Lightweight wrapper for SAMURAI video segmentation."""

from pathlib import Path
from typing import Tuple


class SamuraiWrapper:
    """A thin wrapper around SAMURAI for video object segmentation."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        # In a real implementation, SAMURAI models would be loaded here.

    def propagate(self, frames_dir: Path, first_frame_idx: int, bbox: Tuple[int, int, int, int], output_dir: Path) -> None:
        """Run SAMURAI propagation over a sequence of frames.

        This placeholder simply copies the first frame bbox mask to all frames.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        first_mask = output_dir / f"frame_{first_frame_idx:04d}.png"
        # Placeholder: create empty mask for demonstration.
        first_mask.write_bytes(b"")
        # Real implementation would call SAMURAI library here.

