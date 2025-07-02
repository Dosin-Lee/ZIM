from dataclasses import dataclass
from pathlib import Path

@dataclass
class PipelineConfig:
    """Configuration parameters for the video segmentation pipeline."""

    video_path: Path
    frames_dir: Path
    prompt_file: Path
    samurai_masks_dir: Path
    zim_masks_dir: Path
    overlay_dir: Path
    output_video: Path
    encoder_path: Path
    decoder_path: Path
    fps: int = 30

