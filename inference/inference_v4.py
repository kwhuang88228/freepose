#!/usr/bin/env python3
"""
inference_v4.py — End-to-end 6D object pose trajectory from an arbitrary video.

Uses Depth Anything 3 + MV-SAM3D weighted inference to reconstruct a 3D mesh
from multiple video frames.
All results → data/results/mvsam3d/<video_name>/

Pipeline:
    0. Extract frames
    1. Detect & track (GroundingDINO + SAM2, or SAM3.1) + prepare MV-SAM3D input structure
    2. Depth Anything 3 (DA3) on selected frames
    3. Mask-restricted dense 3D tracking (DenseTrack3DV2 + UniDepth) over the full
       video, keeping only points inside the stage-1 binary mask
    4. MV-SAM3D weighted inference → 3D mesh / Gaussian asset
    5. Visualize GLB outputs
    6. Backfill mesh path in proposals JSON
    7. Scale estimation (ZoeDepth + CLIP)
    8. Per-frame 6D pose estimation (DINOv2 patch-feature matching)
    8.5 Symmetry canonicalization: greedy temporal selection from top-N candidates
    9. Kalman filter smoothing of pose trajectory

NOTE: Stage 3 (dense tracking) was inserted between DA3 and MV-SAM3D. All stages
that previously followed DA3 were renumbered +1 (MV-SAM3D 3→4, GLB viz 4→5,
backfill 5→6, scale 6→7, pose 7→8, canonicalize 7.5→8.5, Kalman 8→9). The
MV-SAM3D output directory was likewise renamed 03_mvsam3d → 04_mvsam3d.

Usage:
    python inference/inference_v4.py --video /path/to/video.mp4
    python inference/inference_v4.py --video /path/to/video.mp4 --track_object "cup"
    python inference/inference_v4.py --video /path/to/video.mp4 --track_object "cup" --tracker sam3.1

Output:
    data/results/mvsam3d/<video_name>/02_da3/da3_output.npz
    data/results/mvsam3d/<video_name>/03_dense_track/dense_3d_track_masked.pkl
    data/results/mvsam3d/<video_name>/04_mvsam3d/   (mesh, splat, logs)
    data/results/mvsam3d/<video_name>/mvsam3d_<video_name>_<prompt_slug>_..._kalman.csv

Debug outputs (per stage):
    data/results/mvsam3d/<video_name>/01_detection_tracking/detection/
    data/results/mvsam3d/<video_name>/01_detection_tracking/tracking/
    data/results/mvsam3d/<video_name>/02_da3/
    data/results/mvsam3d/<video_name>/03_dense_track/
    data/results/mvsam3d/<video_name>/04_mvsam3d/
"""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import cv2

FREEPOSE_ROOT    = Path(__file__).resolve().parent.parent
MVSAM3D_ROOT     = FREEPOSE_ROOT / "MV-SAM3D"
# Dense tracking has its own dependency stack (UniDepth, etc.); run it in its env.
DENSETRACK_PYTHON = "/home/kh775/.conda/envs/densetrack3d/bin/python"


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
        cv2.imwrite(str(out_dir / f"{str(idx).zfill(6)}.png"), frame)
        idx += 1
    cap.release()
    print(f"Extracted {idx} frames → {out_dir}")
    return idx


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
    parser.add_argument("--track_object", default=None,
                        help="Object class to detect and track (e.g. 'cup'). "
                             "Default: None (track all objects).")
    parser.add_argument("--tracker", choices=["gdino_sam2", "sam3.1"], default="gdino_sam2",
                        help="Stage 1 tracker: 'gdino_sam2' (GroundingDINO + SAM2, default) "
                             "or 'sam3.1' (SAM3.1).")

    # ── Common options ────────────────────────────────────────────────────────
    parser.add_argument("--skip_completed", action="store_true", default=True,
                        help="Skip pipeline stages whose output already exists (default: True).")
    parser.add_argument("--no-skip_completed", dest="skip_completed", action="store_false")
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
                        help="Number of evenly-spaced frames to use for reconstruction "
                             "(default: 6)")

    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        sys.exit(f"Video not found: {video_path}")

    video = video_path.stem
    skip  = args.skip_completed

    print(f"\n[inference_v3] video={video}")

    # ── Stage 0: Extract frames ───────────────────────────────────────────────
    frames_dir = FREEPOSE_ROOT / "data" / "datasets" / "videos" / video
    if skip and frames_dir.exists() and any(frames_dir.iterdir()):
        print(f"Stage 0: frames already exist at {frames_dir}, skipping.")
    else:
        extract_frames(video_path, frames_dir)

    prompt_slug = (
        args.track_object.strip().replace(" ", "_").replace("/", "_")
        if args.track_object else "objects"
    )

    results_dir     = (Path(args.results_dir).resolve() if args.results_dir
                       else FREEPOSE_ROOT / "data" / "results" / "mvsam3d" / video)
    results_dir.mkdir(parents=True, exist_ok=True)
    log_command(results_dir)

    props_file      = results_dir / f"mvsam3d_{video}_{prompt_slug}.json"
    names_file      = results_dir / f"mvsam3d_{video}_{prompt_slug}_selected_frames.txt"
    input_dir        = results_dir / "mvsam3d_input"
    da3_output_dir   = results_dir / "02_da3"
    densetrack_dir   = results_dir / "03_dense_track"
    mvsam3d_out_dir  = results_dir / "04_mvsam3d"

    # Stage 1: Detection + tracking + prepare MV-SAM3D input structure
    if skip and props_file.exists() and names_file.exists():
        print(f"Stage 1: output already exists ({props_file.name}), skipping.")
    else:
        if args.tracker == "sam3.1":
            stage1_cmd = [
                "python", "-m", "scripts.extract_proposals_sam3_mvsam3d_v2",
                "--video", video,
                "--num_views", str(args.num_views_mvsam3d),
            ]
            if args.track_object is not None:
                prompt = args.track_object if args.track_object.endswith(".") else args.track_object + "."
                stage1_cmd += ["--prompt", prompt]
        else:
            stage1_cmd = [
                "python", "-m", "scripts.extract_proposals_groundingdino-sam2_mvsam3d",
                "--video", video,
                "--num_views", str(args.num_views_mvsam3d),
            ]
            if args.track_object is not None:
                prompt = args.track_object if args.track_object.endswith(".") else args.track_object + "."
                stage1_cmd += ["--prompt", prompt]

        run(stage1_cmd)

    # Read the selected frame names produced by stage 1
    if not names_file.exists():
        sys.exit(f"Stage 1 did not produce {names_file}")
    selected_names  = [n.strip() for n in names_file.read_text().splitlines() if n.strip()]
    image_names_arg = ",".join(selected_names)

    # Stage 2: Depth Anything 3
    da3_npz = da3_output_dir / "da3_output.npz"
    if skip and da3_npz.exists():
        print(f"Stage 2 (DA3): output already exists ({da3_npz}), skipping.")
    else:
        run([
            "python", str(MVSAM3D_ROOT / "scripts" / "run_da3.py"),
            "--image_dir", str(input_dir / "images"),
            "--output_dir", str(da3_output_dir),
        ])

    # Stage 3: Mask-restricted dense 3D tracking (DenseTrack3DV2 + UniDepth)
    # Tracks the full video and keeps only points inside the stage-1 binary mask.
    # scripts/dense_track_masked.py resolves DENSETRACK_ROOT itself (cwd-independent);
    # run it with the densetrack3d env python. Additive output only — does not feed
    # into MV-SAM3D below.
    binary_masks_dir = results_dir / "01_detection_tracking" / "tracking" / "binary_masks"
    dense_track_pkl  = densetrack_dir / "dense_3d_track_masked.pkl"
    if skip and dense_track_pkl.exists():
        print(f"Stage 3 (dense track): output already exists ({dense_track_pkl}), skipping.")
    elif not binary_masks_dir.exists():
        print(f"Stage 3 (dense track): no binary masks at {binary_masks_dir}, skipping.")
    else:
        run([
            DENSETRACK_PYTHON, str(FREEPOSE_ROOT / "scripts" / "dense_track_masked.py"),
            "--frames_dir", str(frames_dir),
            "--mask_dir", str(binary_masks_dir),
            "--output_dir", str(densetrack_dir),
        ])

    # Stage 4: MV-SAM3D weighted inference
    # Run from MVSAM3D_ROOT so its relative imports (sys.path.append("notebook")) work.
    # Pass --visualization_dir as absolute path to redirect output into results_dir.
    if skip and mvsam3d_out_dir.exists() and any(mvsam3d_out_dir.iterdir()):
        print(f"Stage 4: output already exists at {mvsam3d_out_dir}, skipping.")
    else:
        mvsam3d_cmd = [
            "python", "run_inference_weighted.py",
            "--input_path", str(input_dir),
            "--mask_prompt", prompt_slug,
            "--image_names", image_names_arg,
            "--da3_output", str(da3_npz),
            "--visualization_dir", str(mvsam3d_out_dir),
        ]
        run(mvsam3d_cmd, cwd=MVSAM3D_ROOT)

    # Stage 5: Visualize GLB outputs
    view_glb_script = MVSAM3D_ROOT / "scratch" / "view_glb.py"
    glb_files = sorted(mvsam3d_out_dir.rglob("*.glb"))
    if not glb_files:
        print("Stage 5 (GLB viz): no .glb files found, skipping.")
    else:
        for glb_path in glb_files:
            png_path = glb_path.with_name(glb_path.stem + "_views.png")
            if skip and png_path.exists():
                print(f"Stage 5 (GLB viz): {png_path.name} already exists, skipping.")
                continue
            run([
                "python", str(view_glb_script),
                str(glb_path),
                "--output", str(png_path),
            ])

    # Stage 6: Backfill mesh path in proposals JSON
    meshed_props      = f"mvsam3d_{video}_{prompt_slug}_meshed.json"
    meshed_props_file = results_dir / meshed_props
    if skip and meshed_props_file.exists():
        print(f"Stage 6: proposals already backfilled ({meshed_props}), skipping.")
    else:
        ply_files = sorted(mvsam3d_out_dir.rglob("result.ply"))
        if not ply_files:
            sys.exit(f"No result.ply found under {mvsam3d_out_dir}. Run MV-SAM3D inference first.")
        ply_rel = str(ply_files[0].relative_to(FREEPOSE_ROOT))
        with open(props_file) as _f:
            _props = json.load(_f)
        for _p in _props:
            _p["mesh"] = ply_rel
        with open(meshed_props_file, "w") as _f:
            json.dump(_props, _f)
        print(f"Stage 6: mesh path backfilled → {meshed_props}")

    # Stage 7: Scale estimation
    scaled_props      = meshed_props.replace(".json", "_gpt4_scaled.json")
    scaled_props_file = results_dir / scaled_props
    if skip and scaled_props_file.exists():
        print(f"Stage 7: scale output already exists ({scaled_props}), skipping.")
    else:
        run(["python", "-m", "scripts.compute_scale_video",
             "--video", video,
             "--proposals", meshed_props,
             "--backend", "mvsam3d"])

    # Stage 8: Per-frame 6D pose estimation (DINOv2)
    poses_csv = (
        scaled_props.replace(".json", "")
        + f"_dinopose_layer_{args.dino_layer}_bbext_0.05_depth_zoedepth"
        f"_qimg{'m' if args.mask_query else 'u'}"
        f"_timg{'m' if args.mask_template else 'u'}"
        f"_qpatch{'fg' if args.query_fg_patches else 'all'}"
        f"_tpatch{'fg' if args.template_fg_patches else 'all'}"
        f"_n{args.num_templates}"
        f".csv"
    )
    poses_path = results_dir / poses_csv
    if skip and poses_path.exists():
        print(f"Stage 8: pose output already exists ({poses_csv}), skipping.")
    else:
        stage8_cmd = [
            "python", "-m", "scripts.dino_inference_video_mvsam3d",
            "--video", video,
            "--proposals", scaled_props,
            "--num_templates", str(args.num_templates),
            "--layer", str(args.dino_layer),
        ]
        if not args.mask_query:
            stage8_cmd += ["--no-mask_query"]
        if args.mask_template:
            stage8_cmd += ["--mask_template"]
        if not args.query_fg_patches:
            stage8_cmd += ["--no-query_fg_patches"]
        if args.template_fg_patches:
            stage8_cmd += ["--template_fg_patches"]
        stage8_cmd += ["--top_n_candidates", str(args.top_n_candidates)]
        run(stage8_cmd)

    # Stage 8.5: Symmetry canonicalization (greedy temporal selection from top-5)
    canonical_csv  = poses_csv.replace(".csv", "_canonical.csv")
    canonical_path = results_dir / canonical_csv
    if skip and canonical_path.exists():
        print(f"Stage 8.5: canonical output already exists ({canonical_csv}), skipping.")
    else:
        run(["python", "-m", "scripts.canonicalize_symmetry",
             "--video", video,
             "--poses", poses_csv,
             "--backend", "mvsam3d",
             "--top_n", str(args.top_n_candidates)])

    # Stage 9: Kalman filter smoothing (consumes canonicalized poses)
    kalman_csv  = canonical_csv.replace(".csv", "_kalman.csv")
    kalman_path = results_dir / kalman_csv
    if skip and kalman_path.exists():
        print(f"Stage 9: Kalman output already exists ({kalman_csv}), skipping.")
    else:
        run(["python", "-m", "scripts.kalman_smooth_poses",
             "--video", video,
             "--poses", canonical_csv,
             "--backend", "mvsam3d",
             "--viz"])

    print(f"\nDone. 3D assets written to:\n  {mvsam3d_out_dir}")
    print(f"  DA3 depth/pointmaps: {da3_npz}")
    print(f"  Dense 3D tracks:     {dense_track_pkl}")
    print(f"  6D pose trajectory:  {poses_path}")
    print(f"  Canonicalized:       {canonical_path}")
    print(f"  Kalman-smoothed:     {kalman_path}")


if __name__ == "__main__":
    main()
