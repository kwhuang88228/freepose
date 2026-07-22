#!/usr/bin/env python3
"""
inference_v6.py — End-to-end 6D object pose trajectory from an arbitrary video.

Uses Depth Anything 3 + MV-SAM3D weighted inference to reconstruct a 3D mesh
from multiple video frames.
All results → data/results/mvsam3d/<video_name>/ (override with --results_dir)

Pipeline:
    0. Extract frames
    1. Detect & track (GroundingDINO + SAM2) + prepare MV-SAM3D input structure
    2. Depth Anything 3 (DA3) on selected frames
    3. MV-SAM3D weighted inference → 3D mesh / Gaussian asset, then visualize GLB outputs
    4. Scale estimation (DA3 depth + CLIP)
    5. Per-frame 6D pose estimation (DINOv2 patch-feature matching)
    6. Run HaMeR on the stage-0 frames → wrist trajectory → relative poses, then
       apply relative-pose orientation: keep stage-5 translation, override
       orientation with the HaMeR relative-rotation chain anchored at frame 0

Usage:
    python inference/inference_v6.py --video /path/to/video.mp4
    python inference/inference_v6.py --video /path/to/video.mp4 --track_object "cup"

Output:
    data/results/mvsam3d/<video_name>/02_da3/da3_output.npz
    data/results/mvsam3d/<video_name>/03_mvsam3d/   (mesh, splat, logs)
    data/results/mvsam3d/<video_name>/mvsam3d_<video_name>_<prompt_slug>_..._relpose.csv

Debug outputs (per stage):
    data/results/mvsam3d/<video_name>/01_detection_tracking/detection/
    data/results/mvsam3d/<video_name>/01_detection_tracking/tracking/
    data/results/mvsam3d/<video_name>/02_da3/
    data/results/mvsam3d/<video_name>/03_mvsam3d/
"""

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

FREEPOSE_ROOT    = Path(__file__).resolve().parent.parent
MVSAM3D_ROOT     = FREEPOSE_ROOT / "MV-SAM3D"
HAMER_ROOT       = FREEPOSE_ROOT / "hamer"


def extract_frames(video_path: Path, out_dir: Path) -> int:
    """Decode video to PNG frames with zero-padded names for correct sort order."""
    # Clear any stale frames (e.g. from a previous, uncropped version of the same video).
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {video_path}")
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(out_dir / f"{str(idx).zfill(6)}.png"), frame)
        idx += 1
    cap.release()
    print(f"Extracted {idx} frames → {out_dir}")
    return idx


def save_depth_maps(da3_npz: Path, out_dir: Path) -> None:
    """Save a colorized per-frame depth map from the DA3 output, for debugging.

    Mirrors the per-frame PNG debug dumps in 01_detection_tracking/tracking.
    Depth is normalized globally (2nd/98th percentile across all frames) so the
    globally-scale-consistent DA3 depth is directly comparable across frames.
    """
    data = np.load(da3_npz, allow_pickle=True)
    depth = data["depth"]  # (N, H, W)
    image_files = data["image_files"] if "image_files" in data.files else None

    out_dir.mkdir(parents=True, exist_ok=True)
    valid = np.isfinite(depth) & (depth > 0)
    lo, hi = np.percentile(depth[valid], [2, 98]) if valid.any() else (0.0, 1.0)
    denom = max(hi - lo, 1e-6)

    for i in range(depth.shape[0]):
        stem = Path(str(image_files[i])).stem if image_files is not None else f"{i:06d}"
        norm = np.clip((depth[i] - lo) / denom, 0.0, 1.0)
        vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        cv2.imwrite(str(out_dir / f"{stem}.png"), vis)

    print(f"Saved {depth.shape[0]} depth maps → {out_dir}")


def run(cmd: list[str], cwd: Path = None) -> None:
    print(f"\n[>>] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd or FREEPOSE_ROOT)
    if result.returncode != 0:
        sys.exit(f"Pipeline failed at: {' '.join(cmd)}")


def log_command(results_dir: Path) -> None:
    """Write the invoking shell command to results_dir/command.txt."""
    cmd_line = shlex.join([sys.executable, *sys.argv])
    (results_dir / "command.txt").write_text(cmd_line + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="6D object pose trajectory from video via MV-SAM3D mesh generation"
    )
    parser.add_argument("--video", required=True,
                        help="Path to input video file (e.g. /path/to/clip.mp4)")
    parser.add_argument("--track_object", required=True,
                        help="Object class to detect and track (e.g. 'cup'). ")

    # ── Common options ────────────────────────────────────────────────────────
    parser.add_argument("--results_dir", default=None,
                        help="Directory for all results/debug output "
                             "(default: <FREEPOSE_ROOT>/data/results/mvsam3d/<video_name>).")

    # ── Pose estimation options ───────────────────────────────────────────────
    parser.add_argument("--num_templates", type=int, default=3600,
                        help="Hopf-Hammersley SO(3) template views rendered for pose matching (default: 3600). "
                             "Yaw, pitch, and roll are sampled jointly from one low-discrepancy SO(3) sequence.")
    parser.add_argument("--mask_query", action="store_true", default=True,
                        help="Mask query crop to object foreground (default: True).")
    parser.add_argument("--no-mask_query", dest="mask_query", action="store_false")
    parser.add_argument("--mask_template", action="store_true", default=False,
                        help="Zero out background in rendered templates (default: False).")
    parser.add_argument("--no-mask_template", dest="mask_template", action="store_false")
    parser.add_argument("--query_fg_patches", action="store_true", default=True,
                        help="Average similarity over query foreground patches only (default: True).")
    parser.add_argument("--no-query_fg_patches", dest="query_fg_patches", action="store_false")
    parser.add_argument("--template_fg_patches", action="store_true", default=False,
                        help="Average similarity over template foreground patches only (default: False).")
    parser.add_argument("--no-template_fg_patches", dest="template_fg_patches", action="store_false")
    parser.add_argument("--dino_layer", type=int, default=22,
                        help="DINOv2 layer for patch-feature extraction (default: 22)")
    parser.add_argument("--top_n_candidates", type=int, default=5,
                        help="Number of top pose candidates to consider during "
                             "symmetry canonicalization (default: 5).")

    # ── MV-SAM3D options ──────────────────────────────────────────────────────
    parser.add_argument("--num_views_mvsam3d", type=int, default=6,
                        help="Number of randomly selected frames to use for reconstruction "
                             "(default: 6)")

    # ── DA3 options ───────────────────────────────────────────────────────────
    # Depth is estimated two ways depending on clip length:
    #   * short clips  → run_da3.py: one joint, seam-free pass over all frames. It
    #                    holds every frame's pointmaps in memory, so it OOMs on long
    #                    clips.
    #   * long clips   → run_da3_streaming.py: overlapping chunks processed one at a
    #                    time, offloaded to disk, and Sim(3)-stitched into one
    #                    frame/scale, so GPU memory stays bounded by a single chunk.
    # Both emit the identical da3_output.npz, so all later stages are unchanged.
    parser.add_argument("--da3_streaming_threshold", type=int, default=128,
                        help="Use DA3-Streaming when the clip has more than this many "
                             "frames (default: 128); shorter clips run the single "
                             "joint run_da3 pass.")
    parser.add_argument("--da3_chunk_size", type=int, default=120,
                        help="Frames per DA3-Streaming chunk (default: 120). Only used "
                             "on the streaming path.")
    parser.add_argument("--da3_overlap", type=int, default=60,
                        help="Frames shared between consecutive streaming chunks for "
                             "Sim(3) alignment (default: 60). Must be < --da3_chunk_size.")

    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        sys.exit(f"Video not found: {video_path}")

    video = video_path.stem

    # ── Stage 0: Extract frames ───────────────────────────────────────────────
    frames_dir = FREEPOSE_ROOT / "data" / "datasets" / "videos" / video
    n_frames = extract_frames(video_path, frames_dir)

    prompt_slug = args.track_object.rstrip(".").strip().replace(" ", "_").replace("/", "_")

    results_dir     = (Path(args.results_dir).resolve() if args.results_dir
                       else FREEPOSE_ROOT / "data" / "results" / "mvsam3d" / video)
    results_dir.mkdir(parents=True, exist_ok=True)
    log_command(results_dir)

    props_file      = results_dir / f"mvsam3d_{video}_{prompt_slug}.json"
    names_file      = results_dir / f"mvsam3d_{video}_{prompt_slug}_selected_frames.txt"
    input_dir        = results_dir / "mvsam3d_input"
    da3_output_dir   = results_dir / "02_da3"
    mvsam3d_out_dir  = results_dir / "03_mvsam3d"

    # Stage 1: Detection + tracking (GroundingDINO + SAM2) + prepare MV-SAM3D input structure
    stage1_cmd = [
        "python", "-m", "scripts.extract_proposals_groundingdino-sam2_mvsam3d",
        "--video", video,
        "--num_views", str(args.num_views_mvsam3d),
        "--prompt", args.track_object if args.track_object.endswith(".") else args.track_object + ".",
        "--results_dir", str(results_dir),
    ]

    run(stage1_cmd)

    # Read the selected frame names produced by stage 1
    if not names_file.exists():
        sys.exit(f"Stage 1 did not produce {names_file}")
    selected_names  = [n.strip() for n in names_file.read_text().splitlines() if n.strip()]
    image_names_arg = ",".join(selected_names)

    # Stage 2: Depth Anything 3 → globally-scale-consistent depth for the clip.
    # Run with this interpreter (freepose); both scripts add DA3/src to sys.path.
    # Short clips take the single joint run_da3 pass; long clips stream (see the
    # --da3_streaming_threshold help above). Both write da3_output.npz.
    da3_npz = da3_output_dir / "da3_output.npz"
    if n_frames > args.da3_streaming_threshold:
        print(f"Stage 2: {n_frames} frames > {args.da3_streaming_threshold}; "
              f"using DA3-Streaming (chunk_size={args.da3_chunk_size}, "
              f"overlap={args.da3_overlap}).")
        run([
            sys.executable, str(MVSAM3D_ROOT / "scripts" / "run_da3_streaming.py"),
            "--image_dir", str(frames_dir),
            "--output_dir", str(da3_output_dir),
            "--output_npz", str(da3_npz),
            "--chunk_size", str(args.da3_chunk_size),
            "--overlap", str(args.da3_overlap),
        ])
    else:
        print(f"Stage 2: {n_frames} frames <= {args.da3_streaming_threshold}; "
              f"using single joint run_da3 pass.")
        run([
            sys.executable, str(MVSAM3D_ROOT / "scripts" / "run_da3.py"),
            "--image_dir", str(frames_dir),
            "--output_dir", str(da3_output_dir),
            "--output_npz", str(da3_npz),
            "--no_vis",
        ])

    # Debug: dump a colorized per-frame depth map 
    save_depth_maps(da3_npz, da3_output_dir / "depth_maps")

    # Stage 3: MV-SAM3D weighted inference → 3D mesh / Gaussian asset, then visualize GLB outputs
    # Run from MVSAM3D_ROOT so its relative imports (sys.path.append("notebook")) work.
    # Pass --visualization_dir as absolute path to redirect output into results_dir.
    mvsam3d_cmd = [
        "python", "run_inference_weighted.py",
        "--input_path", str(input_dir),
        "--mask_prompt", prompt_slug,
        "--image_names", image_names_arg,
        "--da3_output", str(da3_npz),
        "--visualization_dir", str(mvsam3d_out_dir),
    ]
    run(mvsam3d_cmd, cwd=MVSAM3D_ROOT)

    # Visualize the GLB outputs produced by the inference above.
    view_glb_script = MVSAM3D_ROOT / "scratch" / "view_glb.py"
    glb_files = sorted(mvsam3d_out_dir.rglob("*.glb"))
    if not glb_files:
        print("Stage 3 (GLB viz): no .glb files found, skipping.")
    else:
        for glb_path in glb_files:
            png_path = glb_path.with_name(glb_path.stem + "_views.png")
            run([
                "python", str(view_glb_script),
                str(glb_path),
                "--output", str(png_path),
            ])

    # Locate the Gaussian splat / mesh produced by MV-SAM3D. Passed directly to the
    # pose stage below via --mesh instead of being backfilled into the proposals JSON.
    ply_files = sorted(mvsam3d_out_dir.rglob("result.ply"))
    if not ply_files:
        sys.exit(f"No result.ply found under {mvsam3d_out_dir}. Run MV-SAM3D inference first.")
    mesh_path = str(ply_files[0])
    print(f"Stage 3: mesh asset → {mesh_path}")

    # Stage 4: Scale estimation
    scaled_props      = props_file.name.replace(".json", "_gpt4_scaled.json")
    run(["python", "-m", "scripts.compute_scale_video",
         "--video", video,
         "--proposals", props_file.name,
         "--backend", "mvsam3d",
         "--da3_depth", str(da3_npz),
         "--results_dir", str(results_dir)])

    # Stage 5: Per-frame 6D pose estimation (DINOv2)
    poses_csv = (
        scaled_props.replace(".json", "")
        + f"_dinopose_layer_{args.dino_layer}_bbext_0.05_depth_da3"
        f"_qimg{'m' if args.mask_query else 'u'}"
        f"_timg{'m' if args.mask_template else 'u'}"
        f"_qpatch{'fg' if args.query_fg_patches else 'all'}"
        f"_tpatch{'fg' if args.template_fg_patches else 'all'}"
        f"_n{args.num_templates}"
        f".csv"
    )
    poses_path = results_dir / poses_csv
    stage5_cmd = [
        "python", "-m", "scripts.dino_inference_video_mvsam3d",
        "--video", video,
        "--proposals", scaled_props,
        "--mesh", mesh_path,
        "--num_templates", str(args.num_templates),
        "--layer", str(args.dino_layer),
        "--da3_depth", str(da3_npz),
        "--results_dir", str(results_dir),
    ]
    if not args.mask_query:
        stage5_cmd += ["--no-mask_query"]
    if args.mask_template:
        stage5_cmd += ["--mask_template"]
    if not args.query_fg_patches:
        stage5_cmd += ["--no-query_fg_patches"]
    if args.template_fg_patches:
        stage5_cmd += ["--template_fg_patches"]
    stage5_cmd += ["--top_n_candidates", str(args.top_n_candidates)]
    run(stage5_cmd)

    # # Stage 7: Apply relative-pose orientation
    # # Run HaMeR on the stage-0 frames to get the right-wrist trajectory
    # # (wrist_poses.json), convert it to frame-to-frame relative poses (rel_poses.json),
    # # then keep the stage-6 per-frame translation (position estimation is already
    # # reliable) but override the orientation with the relative-rotation chain, anchored
    # # at frame 0's estimated orientation. Only the rotation is propagated; the
    # # translation is untouched. Downstream symmetry canonicalization and Kalman
    # # smoothing are intentionally not run.
    # hamer_out_dir = results_dir / "04_hamer"
    # run([
    #     "python", "demo.py",
    #     "--img_folder", str(frames_dir),
    #     "--out_folder", str(hamer_out_dir),
    #     "--batch_size=48", "--side_view", "--save_mesh", "--full_frame",
    # ], cwd=HAMER_ROOT)

    # wrist_poses_file = hamer_out_dir / "wrist_poses.json"
    # if not wrist_poses_file.exists():
    #     sys.exit(f"Stage 7: HaMeR did not produce {wrist_poses_file}")
    # rel_poses_file = hamer_out_dir / "rel_poses.json"
    # run([
    #     "python", str(HAMER_ROOT / "inference" / "wrist_pose_to_rel_pose.py"),
    #     str(wrist_poses_file),
    #     "-o", str(rel_poses_file),
    # ])

    # relpose_csv  = poses_csv.replace(".csv", "_relpose.csv")
    # relpose_path = results_dir / relpose_csv
    # run(["python", "-m", "scripts.apply_rel_pose_orientation",
    #      "--video", video,
    #      "--poses", poses_csv,
    #      "--backend", "mvsam3d",
    #      "--rel_poses", str(rel_poses_file)])

    print(f"\nDone. 3D assets written to:\n  {mvsam3d_out_dir}")
    # print(f"  DA3 depth/pointmaps: {da3_npz}")
    print(f"  6D pose trajectory:  {poses_path}")
    # print(f"  Rel-pose orientation:{relpose_path}")


if __name__ == "__main__":
    main()
