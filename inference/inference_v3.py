#!/usr/bin/env python3
"""
inference_v3.py — End-to-end 6D object pose trajectory from an arbitrary video,
                  using SAM-3D-Objects for per-object Gaussian splat generation
                  instead of mesh retrieval.

Usage:
    python inference/inference_v3.py --video /path/to/video.mp4
    python inference/inference_v3.py --video /path/to/video.mp4 --track_object "cup"

Pipeline:
    0. Extract frames → data/datasets/videos/<video_name>/
    1. Detect & track objects (Grounding DINO + SAM2, highest-confidence object only)
       + generate Gaussian splat per object (SAM-3D-Objects) from the first frame
    2. Estimate object scale via ZoeDepth + CLIP
    3. Per-frame 6D pose estimation via DINOv2 patch-feature matching (600 templates
       rendered from the Gaussian splat)
    4. Temporal refinement (2D-3D point tracking) + pose smoothing (R, t)

Output:
    data/results/sam3d/<video_name>/<video_name>-tracked-sam3d.csv
    Columns: scene_id, im_id, obj_id, score, R (3x3 flattened), t (mm), bbox_visib, scale

Debug outputs (per stage):
    data/results/sam3d/<video_name>/01_detection/
    data/results/sam3d/<video_name>/02_tracking/
    data/results/sam3d/<video_name>/03_splats/
    data/results/sam3d/<video_name>/04_coarse_poses/
    data/results/sam3d/<video_name>/05_tracked/
    data/results/sam3d/<video_name>/06_vis/
"""

import argparse
import subprocess
import sys
from pathlib import Path

import cv2

FREEPOSE_ROOT = Path(__file__).resolve().parent.parent


def extract_frames(video_path: Path, out_dir: Path) -> int:
    """Decode video to JPEG frames with zero-padded names for correct sort order."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {video_path}")
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(out_dir / f"{str(idx).zfill(6)}.jpg"), frame)
        idx += 1
    cap.release()
    print(f"Extracted {idx} frames → {out_dir}")
    return idx


def run(cmd: list[str]) -> None:
    print(f"\n[>>] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=FREEPOSE_ROOT)
    if result.returncode != 0:
        sys.exit(f"Pipeline failed at: {' '.join(cmd)}")


def main():
    parser = argparse.ArgumentParser(description="6D object pose trajectory from video (SAM-3D mesh generation)")
    parser.add_argument("--video", required=True, help="Path to input video file (e.g. /path/to/clip.mp4)")
    parser.add_argument("--track_object", default=None,
                        help="If set, only detect and track this object class (e.g. 'cup'). "
                             "Passed as the text prompt to Grounding DINO. Default: None (track all objects).")
    parser.add_argument("--skip_completed", action="store_true", default=True,
                        help="Skip pipeline stages whose output already exists (default: True). "
                             "Pass --no-skip_completed to rerun all stages.")
    parser.add_argument("--no-skip_completed", dest="skip_completed", action="store_false")
    parser.add_argument("--num_templates", type=int, default=600,
                        help="Number of Hammersley template views rendered for pose matching (default: 600)")
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        sys.exit(f"Video not found: {video_path}")

    video = video_path.stem  # use filename without extension as video name

    skip = args.skip_completed

    # Stage 0: Extract frames into the directory the pipeline expects
    frames_dir = FREEPOSE_ROOT / "data" / "datasets" / "videos" / video
    if skip and frames_dir.exists() and any(frames_dir.iterdir()):
        print(f"Stage 0: frames already exist at {frames_dir}, skipping.")
    else:
        extract_frames(video_path, frames_dir)

    # Derived file names — use "sam3d" prefix throughout (no BASE_PROPS mesh-retrieval prefix)
    props     = f"sam3d_{video}.json"
    scaled    = f"sam3d_{video}_gpt4_scaled.json"
    poses_csv = f"sam3d_{video}_gpt4_scaled_dinopose_layer_22_bbext_0.05_depth_zoedepth.csv"
    output    = FREEPOSE_ROOT / "data" / "results" / "sam3d" / video / f"{video}-tracked-sam3d.csv"

    results_dir = FREEPOSE_ROOT / "data" / "results" / "sam3d" / video
    scaled_path = results_dir / scaled

    # Stage 1: Zero-shot detection (Grounding DINO) + SAM2 tracking + SAM-3D splat generation
    if skip and (results_dir / props).exists():
        print(f"Stage 1: output already exists ({props}), skipping.")
    else:
        stage1_cmd = ["python", "-m", "scripts.extract_proposals_ground_video_sam3d", "--video", video]
        if args.track_object is not None:
            prompt = args.track_object if args.track_object.endswith(".") else args.track_object + "."
            stage1_cmd += ["--prompt", prompt]
        run(stage1_cmd)

    # Stage 2: Per-object scale estimation via ZoeDepth metric depth + CLIP
    if skip and scaled_path.exists():
        print(f"Stage 2: output already exists ({scaled}), skipping.")
    else:
        run(["python", "-m", "scripts.compute_scale_video", "--video", video, "--proposals", props])

    # Stage 3: Coarse 6D pose via DINOv2 patch-feature matching to 600 splat-rendered templates
    poses_path = results_dir / poses_csv
    if skip and poses_path.exists():
        print(f"Stage 3: output already exists ({poses_csv}), skipping.")
    else:
        run(["python", "-m", "scripts.dino_inference_video_sam3d", "--video", video, "--proposals", scaled,
             "--num_templates", str(args.num_templates)])

    # Stage 4: Refine via 2D-3D point tracking + PnP, then smooth R and t temporally
    if skip and output.exists():
        print(f"Stage 4: output already exists ({output.name}), skipping.")
    else:
        run([
            "python", "-m", "scripts.smooth_poses_video_sam3d",
            "--video", video,
            "--proposals", scaled,
            "--poses", poses_csv,
            "--vis"
        ])

    print(f"\nDone. 6D pose trajectory written to:\n  {output}")


if __name__ == "__main__":
    main()
