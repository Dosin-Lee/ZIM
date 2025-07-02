"""Main script that ties all modules together."""

from pathlib import Path
import cv2

from .config import PipelineConfig
from .frame_splitter import split_video_to_frames
from .samurai_wrapper import SamuraiWrapper
from .bbox_extractor import bbox_from_mask
from .zim_inference import ZimInferencer
from .overlay import overlay_mask
from .video_merger import merge_frames_to_video


def run_pipeline(cfg: PipelineConfig) -> None:
    """Execute the video segmentation pipeline using SAMURAI and ZIM."""

    frame_paths = split_video_to_frames(cfg.video_path, cfg.frames_dir)
    first_frame = frame_paths[0]

    # Load bbox prompt from file
    bbox = tuple(map(int, Path(cfg.prompt_file).read_text().split()))  # x1 y1 x2 y2

    zim = ZimInferencer(cfg.encoder_path, cfg.decoder_path)

    # High-res mask for the first frame
    first_mask = cfg.zim_masks_dir / first_frame.name.replace(".jpg", ".png")
    cfg.zim_masks_dir.mkdir(parents=True, exist_ok=True)
    mask = zim.infer(first_frame, bbox)
    cv2.imwrite(str(first_mask), mask)

    # Run SAMURAI to propagate coarse masks
    samurai = SamuraiWrapper(cfg.samurai_masks_dir)
    samurai.propagate(cfg.frames_dir, 0, bbox, cfg.samurai_masks_dir)

    # Iterate over frames
    overlay_paths = []
    cfg.overlay_dir.mkdir(parents=True, exist_ok=True)
    for frame_path in frame_paths:
        samurai_mask = cfg.samurai_masks_dir / frame_path.name.replace(".jpg", ".png")
        bbox = bbox_from_mask(samurai_mask)
        mask = zim.infer(frame_path, bbox)
        zim_mask_path = cfg.zim_masks_dir / frame_path.name.replace(".jpg", ".png")
        cv2.imwrite(str(zim_mask_path), mask)

        overlay_path = cfg.overlay_dir / frame_path.name
        overlay_mask(frame_path, zim_mask_path, overlay_path)
        overlay_paths.append(overlay_path)

    merge_frames_to_video(overlay_paths, cfg.output_video, cfg.fps)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run video segmentation pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to config json")
    args = parser.parse_args()

    import json

    cfg_dict = json.loads(Path(args.config).read_text())
    cfg = PipelineConfig(**{k: Path(v) if "path" in k or "dir" in k else v for k, v in cfg_dict.items()})
    run_pipeline(cfg)

