"""
extract_proposals_sam3_mvsam3d_v2.py

Stage 1 (SAM3.1 variant) of the MV-SAM3D pipeline using the inference.py logic:
  1. Detect & track one object across all frames with SAM3.1 (text prompt → argmax
     confidence, largest-connected-component cleanup per frame).
  2. Save debug PNGs: binary_mask/, boxes/, masks_overlay/.
  3. Prepare MV-SAM3D input structure (images + masks).
  4. Save proposals JSON.

Debug outputs → data/results/mvsam3d/<video>/01_detection_tracking/
  binary_mask/     frame{N:05d}_obj{id}.png
  boxes/           frame{N:05d}.png
  masks_overlay/   frame{N:05d}.png

MV-SAM3D input → data/results/mvsam3d/<video>/mvsam3d_input/
  images/           — N evenly-spaced video frames
  {prompt_slug}/    — corresponding binary masks ({name}_mask.png)
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import torch
from loguru import logger
from sam3.inference.inference import run_inference, save_results
from sam3.train.masks_ops import rle_encode

_FREEPOSE_ROOT = Path(__file__).resolve().parent.parent

random.seed(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",     type=str, required=True)
    parser.add_argument("--prompt",    type=str, default="objects.")
    parser.add_argument("--num_views", type=int, default=6,
                        help="Number of evenly-spaced frames to select for MV-SAM3D (default: 6)")
    args = parser.parse_args()

    video_dir   = _FREEPOSE_ROOT / "data" / "datasets" / "videos" / args.video
    video_mp4   = _FREEPOSE_ROOT / "data" / "datasets" / "videos" / f"{args.video}.mp4"
    frame_paths = sorted(video_dir.glob("*.png")) if video_dir.exists() else []

    # Prefer the folder of extracted PNG frames if available; fall back to the mp4.
    if frame_paths:
        video_source = str(video_dir)
    elif video_mp4.exists():
        video_source = str(video_mp4)
    else:
        sys.exit(f"No frames in {video_dir} and no mp4 at {video_mp4}")

    results_dir = _FREEPOSE_ROOT / "data" / "results" / "mvsam3d" / args.video
    results_dir.mkdir(parents=True, exist_ok=True)
    prompt_slug = args.prompt.rstrip(".").strip().replace(" ", "_").replace("/", "_") or "objects"
    output_file = results_dir / f"mvsam3d_{args.video}_{prompt_slug}.json"

    debug_dir = results_dir / "01_detection_tracking"

    # ── SAM3.1 detection + tracking (single-session, argmax confidence) ────────
    text_prompt = args.prompt.rstrip(".")
    logger.info(f"Running SAM3.1 inference: source={video_source}, prompt='{text_prompt}'")

    results, video_frames = run_inference(video_source, text_prompt)

    if not results:
        sys.exit("SAM3.1 returned no results — check the prompt or video source.")

    # ── Debug visualizations (binary_mask/, boxes/, masks_overlay/) ────────────
    save_results(results, str(debug_dir), video_frames)
    logger.info(f"Debug PNGs saved → {debug_dir}")

    # ── Build proposals JSON ──────────────────────────────────────────────────
    props = []
    for frame_idx in sorted(results):
        for obj_id, mask in results[frame_idx]["binary_masks"].items():
            bbox = results[frame_idx]["bboxes"].get(obj_id)
            if bbox is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            mask_t = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)
            props.append({
                "bbox":         [x1, y1, x2 - x1, y2 - y1],
                "segmentation": rle_encode(mask_t)[0],
                "mesh":         None,
                "score":        1.0,
                "scene_id":     0,
                "image_id":     int(frame_idx),
                "time":         0.01,
            })

    with open(output_file, "w") as f:
        json.dump(props, f)
    logger.info(f"Proposals saved → {output_file} ({len(props)} entries)")

    # ── Prepare MV-SAM3D input structure ─────────────────────────────────────
    valid_idxs    = [idx for idx in sorted(results) if results[idx]["binary_masks"]]
    num_views     = min(args.num_views, len(valid_idxs))
    selected_idxs = random.sample(valid_idxs, num_views)

    mvsam3d_input_dir = results_dir / "mvsam3d_input"
    images_dir        = mvsam3d_input_dir / "images"
    masks_dest        = mvsam3d_input_dir / prompt_slug
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dest.mkdir(parents=True, exist_ok=True)

    selected_names = []
    for idx in selected_idxs:
        name = f"{idx:06d}"

        # Copy frame image (prefer pre-extracted PNGs, fall back to video_frames)
        dst_img = images_dir / f"{name}.png"
        src_img = video_dir / f"{name}.png"
        if src_img.exists() and not dst_img.exists():
            shutil.copy2(str(src_img), str(dst_img))

        # Copy binary mask from debug output
        obj_id   = next(iter(results[idx]["binary_masks"]))
        src_mask = debug_dir / "binary_mask" / f"frame{idx:05d}_obj{obj_id}.png"
        dst_mask = masks_dest / f"{name}_mask.png"
        if src_mask.exists() and not dst_mask.exists():
            shutil.copy2(str(src_mask), str(dst_mask))

        selected_names.append(name)

    names_file = results_dir / f"mvsam3d_{args.video}_{prompt_slug}_selected_frames.txt"
    names_file.write_text("\n".join(selected_names))
    logger.info(f"MV-SAM3D input prepared → {mvsam3d_input_dir}")
    logger.info(f"Selected {num_views} frames: {selected_names}")
    logger.info(f"Frame names saved → {names_file}")
