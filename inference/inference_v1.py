#!/usr/bin/env python3
"""
inference_v1.py — End-to-end 6D object pose trajectory from an arbitrary video.

Usage:
    python inference/inference_v1.py --video /path/to/video.mp4

Pipeline:
    0. Extract frames → data/datasets/videos/<video_name>/
    1. Detect & track objects (Grounding DINO + SAM2) + mesh retrieval (DINOv2)
    2. Estimate object scale via ZoeDepth + CLIP
    3. Filter to best-matching object track
    4. Per-frame 6D pose estimation via DINOv2 patch-feature matching (600 templates)
    5. Temporal refinement (2D-3D PnP tracking) + pose smoothing (R, t)

Output:
    data/results/videos/<video_name>/<video_name>-tracked.csv
    Columns: scene_id, im_id, obj_id, score, R (3x3 flattened), t (mm), bbox_visib, scale
"""

import argparse
import json
import subprocess
import sys
from itertools import takewhile
from pathlib import Path

import cv2
import numpy as np

FREEPOSE_ROOT = Path(__file__).resolve().parent.parent
BASE_PROPS = "props-ground-box-0.2-text-0.2-ffa-22-top-25"


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


def filter_by_score(proposals_path: Path) -> None:
    """Fallback for filter_predictions when no GT exists: pick object with highest mean score."""
    with open(proposals_path) as f:
        proposals = json.load(f)

    N = len(list(takewhile(lambda x: x["image_id"] == 0, proposals)))
    object_proposals = [proposals[i::N] for i in range(N)]

    mean_scores = [np.mean([p["score"] for p in obj]) for obj in object_proposals]
    idx = int(np.argmax(mean_scores))
    print(f"No GT found — selecting object {idx} with mean score {mean_scores[idx]:.3f} "
          f"(out of {N} tracked objects)")

    best_path = proposals_path.with_name(proposals_path.stem + "_best_object.json")
    with open(best_path, "w") as f:
        json.dump(object_proposals[idx], f)


def run(cmd: list[str]) -> None:
    print(f"\n[>>] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=FREEPOSE_ROOT)
    if result.returncode != 0:
        sys.exit(f"Pipeline failed at: {' '.join(cmd)}")


def main():
    parser = argparse.ArgumentParser(description="6D object pose trajectory from video")
    parser.add_argument("--video", required=True, help="Path to input video file (e.g. /path/to/clip.mp4)")
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        sys.exit(f"Video not found: {video_path}")

    video = video_path.stem  # use filename without extension as video name

    # Stage 0: Extract frames into the directory the pipeline expects
    frames_dir = FREEPOSE_ROOT / "data" / "datasets" / "videos" / video
    if frames_dir.exists() and any(frames_dir.iterdir()):
        print(f"Frames already exist at {frames_dir}, skipping extraction.")
    else:
        extract_frames(video_path, frames_dir)

    # Derived file names following pipeline naming convention
    props     = f"{BASE_PROPS}_{video}.json"
    scaled    = f"{BASE_PROPS}_{video}_gpt4_scaled.json"
    best      = f"{BASE_PROPS}_{video}_gpt4_scaled_best_object.json"
    poses_csv = f"{BASE_PROPS}_{video}_gpt4_scaled_best_object_dinopose_layer_22_bbext_0.05_depth_zoedepth.csv"
    output    = FREEPOSE_ROOT / "data" / "results" / "videos" / video / f"{video}-tracked.csv"

    # Stage 1: Zero-shot detection (Grounding DINO) + SAM2 tracking + mesh retrieval
    run(["python", "-m", "scripts.extract_proposals_ground_video", "--video", video])

    # Stage 2: Per-object scale estimation via ZoeDepth metric depth + CLIP
    run(["python", "-m", "scripts.compute_scale_video", "--video", video, "--proposals", props])

    # Stage 3: Select best object track (by IoU vs GT if available, else by mean score)
    gt_path = FREEPOSE_ROOT / "data" / "video_gt" / f"{video}_poses_id1.npy"
    scaled_path = FREEPOSE_ROOT / "data" / "results" / "videos" / video / scaled
    if gt_path.exists():
        run(["python", "-m", "scripts.filter_predictions", "--video", video, "--proposals", scaled])
    else:
        filter_by_score(scaled_path)

    # Stage 4: Coarse 6D pose via DINOv2 patch-feature matching to 600 rendered templates
    run(["python", "-m", "scripts.dino_inference_video", "--video", video, "--proposals", best])

    # Stage 5: Refine via 2D-3D point tracking + PnP, then smooth R and t temporally
    run([
        "python", "-m", "scripts.smooth_poses_video",
        "--video", video,
        "--proposals", best,
        "--poses", poses_csv,
    ])

    print(f"\nDone. 6D pose trajectory written to:\n  {output}")


if __name__ == "__main__":
    main()
