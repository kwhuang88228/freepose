#!/usr/bin/env python3
"""
apply_rel_pose_orientation.py

Override the per-frame object orientation with a relative-rotation chain anchored
at the first frame's estimated pose, while keeping the per-frame translation from
the pose estimator unchanged (position estimation is already reliable).

Only the initial object orientation is taken from the estimator (frame 0, top-1).
Every later frame's orientation is propagated by composing the wrist's frame-to-frame
relative rotations from HaMeR's rel_poses.json:

    R_obj[k] = R_obj[0] @ R_rel[1] @ R_rel[2] @ ... @ R_rel[k]      (rotation only)

rel_poses.json stores rel[i] = inv(wrist_pose[i-1]) @ wrist_pose[i], i.e. the wrist
motion expressed in the previous frame's local frame (right-multiply composition,
matching how the wrist trajectory itself is rebuilt). For an object rigidly held in
the hand this right-multiplied chain reproduces the object's rigid rotation in the
camera frame; the constant grip offset cancels because it is folded into R_obj[0].

Translation is copied verbatim from the input CSV. The top-N candidate columns
(R_top5 / t_top5 / score_top5) are left untouched but are stale after this step
(orientation no longer comes from candidate selection); downstream symmetry
canonicalization / Kalman stages are not run.

With --viz (default), the 3D bounding box and RGB coordinate axes of each detected
object are rendered onto every frame using the final orientation, for debugging,
under results/<backend>/<video>/05_relpose_viz/.

Usage:
    python -m scripts.apply_rel_pose_orientation \
        --video <video> --poses <stage8_csv> --backend mvsam3d \
        --rel_poses /path/to/rel_poses.json
"""

import itertools
import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

_FREEPOSE_ROOT = Path(__file__).resolve().parent.parent


def build_accumulated_rotations(rel_poses: list) -> dict:
    """Return {frame_idx: A} where A = R_rel[first+1] @ ... @ R_rel[frame_idx].

    The anchor frame (the earliest prev_frame) maps to identity. Entries are
    accumulated in frame order following the prev_frame -> frame chain.
    """
    rel_sorted = sorted(rel_poses, key=lambda e: int(e["frame"]))
    anchor = int(rel_sorted[0]["prev_frame"])
    accum = {anchor: np.eye(3)}
    for e in rel_sorted:
        prev_f, cur_f = int(e["prev_frame"]), int(e["frame"])
        if prev_f not in accum:
            raise SystemExit(
                f"rel_poses chain is not contiguous: frame {cur_f} references "
                f"prev_frame {prev_f}, which has no accumulated rotation yet."
            )
        R_rel = np.asarray(e["pose"], dtype=float)[:3, :3]
        accum[cur_f] = accum[prev_f] @ R_rel
    return accum


def render_debug_viz(df: pd.DataFrame, video: str, results_dir: Path) -> None:
    """Overlay the 3D bounding box and RGB coordinate axes of each detected object
    onto every frame using the final (rel-pose) orientation, and stitch videos.
    Also renders an axes-only overlay (no bbox) under 05_relpose_viz/axis/.

    Reuses the stage-8 drawing helpers so the box/axes match the pose convention.
    Both helpers re-derive the translation from the bbox + scale internally (the
    reliable position estimate), so only R / scale / bbox from this CSV are used.
    """
    import cv2  # noqa: local import; keeps the transform path free of heavy deps
    import torch
    from scripts.dino_inference_video_mvsam3d import (
        _dir_to_video, _save_bbox_3d, _save_bbox3d_axis, _save_axis,
    )
    from src.pipeline.retrieval.renderer_sam3d import load_gaussian

    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames_dir  = _FREEPOSE_ROOT / "data" / "datasets" / "videos" / video
    frame_names = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() == ".png")
    if not frame_names:
        print(f"[warn] no frames at {frames_dir}, skipping debug viz.")
        return

    img0 = cv2.imread(str(frame_names[0]))
    h, w = img0.shape[:2]
    f_px = np.sqrt(h ** 2 + w ** 2)
    K = np.array([[f_px, 0, w / 2.0], [0, f_px, h / 2.0], [0, 0, 1]], dtype=float)

    viz_dir       = results_dir / "05_relpose_viz"
    bbox_dir      = viz_dir / "bbox3d"
    axis_dir      = viz_dir / "bbox3d_axis"
    axis_only_dir = viz_dir / "axis"
    bbox_dir.mkdir(parents=True, exist_ok=True)
    axis_dir.mkdir(parents=True, exist_ok=True)
    axis_only_dir.mkdir(parents=True, exist_ok=True)

    # Load each unique object's Gaussian splat once.
    gs_cache = {}
    for obj_id in df["obj_id"].unique():
        gs_cache[obj_id] = load_gaussian(_FREEPOSE_ROOT / obj_id, device=device)

    for frame_idx, frame_name in enumerate(frame_names):
        rows = df[df.im_id == frame_idx]
        if rows.empty:
            continue
        img = cv2.cvtColor(cv2.imread(str(frame_name)), cv2.COLOR_BGR2RGB).astype(np.uint8)
        for obj_idx, (_, row) in enumerate(rows.iterrows()):
            R = np.array([float(x) for x in row["R"].split()]).reshape(3, 3)
            t = np.array([float(x) for x in row["t"].split()])
            x, y, bw, bh = [float(v) for v in str(row["bbox_visib"]).split()]
            box = np.array([x, y, x + bw, y + bh], dtype=float)   # xywh -> xyxy
            gs = gs_cache[row["obj_id"]]
            scale = float(row["scale"])
            tag = f"{frame_idx:06d}_obj{obj_idx}.png"
            try:
                _save_bbox_3d(img, gs, K, R, t, scale, box, bbox_dir / tag)
                _save_bbox3d_axis(img, gs, K, R, t, scale, box, axis_dir / tag)
                _save_axis(img, gs, K, R, t, scale, box, axis_only_dir / tag)
            except Exception as exc:
                print(f"[warn] debug viz failed at frame {frame_idx} obj {obj_idx}: {exc}")

    _dir_to_video(bbox_dir, viz_dir / "bbox3d.mp4")
    _dir_to_video(axis_dir, viz_dir / "bbox3d_axis.mp4")
    _dir_to_video(axis_only_dir, viz_dir / "axis.mp4")
    print(f"Debug viz (bbox + RGB axes) → {viz_dir}")


def main(args):
    results_dir = Path("data") / "results" / args.backend / args.video
    csv_path = results_dir / args.poses

    df = pd.read_csv(csv_path)

    with open(args.rel_poses) as f:
        rel_poses = json.load(f)
    accum = build_accumulated_rotations(rel_poses)

    n_objects = len(list(itertools.takewhile(
        lambda x: x == df.iloc[0]["im_id"], df["im_id"]
    )))

    out_df = df.copy()
    n_held = 0

    for obj_idx in range(n_objects):
        sub_idx = list(range(obj_idx, len(df), n_objects))
        frames = [int(df.iloc[i]["im_id"]) for i in sub_idx]

        R0 = np.array([float(x) for x in df.iloc[sub_idx[0]]["R"].split()]).reshape(3, 3)

        last_A = np.eye(3)
        for i, f in zip(sub_idx, frames):
            if f in accum:
                last_A = accum[f]
            else:
                # rel_poses chain does not reach this frame; hold the last rotation.
                n_held += 1
            R_f = R0 @ last_A
            out_df.at[i, "R"] = " ".join(str(x) for x in R_f.flatten())

    out_csv = results_dir / (csv_path.stem + "_relpose.csv")
    out_df.to_csv(out_csv, index=False)

    print(f"Relative-pose orientation applied → {out_csv}")
    print(f"  Objects: {n_objects}, frames/object: {len(df) // n_objects}")
    if n_held:
        print(f"  [warn] {n_held} row(s) had no rel_pose and held the last orientation.")

    if args.viz:
        render_debug_viz(out_df, args.video, results_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--poses", type=str, required=True,
                        help="CSV filename inside results/<backend>/<video>/ from stage 8.")
    parser.add_argument("--backend", type=str, choices=["sam3d", "mvsam3d"], required=True)
    parser.add_argument("--rel_poses", type=str, required=True,
                        help="Path to HaMeR rel_poses.json for this video.")
    parser.add_argument("--viz", action="store_true", default=True,
                        help="Render 3D bbox + RGB axes debug overlays (default: True).")
    parser.add_argument("--no-viz", dest="viz", action="store_false")
    args = parser.parse_args()
    main(args)
