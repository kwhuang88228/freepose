"""
smooth_poses_video_sam3d.py

Stage 5 of the SAM-3D pipeline:
  1. Load Gaussian splat for each tracked object (path from CSV obj_id).
  2. Find the highest-confidence frame using DINOv2 feature similarity.
  3. Run CoTracker-based 2D-3D tracking to refine poses interval by interval.
  4. Smooth R and t temporally.
  5. Write refined poses to <video>-tracked-sam3d.csv.

Debug outputs → data/results/sam3d/<video>/05_tracked/
  Per-frame centroid overlay PNGs (from visualize()).
"""

import itertools
import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import tqdm
from loguru import logger
from PIL import Image, ImageDraw
from sam2.utils.amg import rle_to_mask

# ── 3-D bounding-box helpers ───────────────────────────────────────────────────
_AXIS_COLORS_BGR = {"x": (0, 0, 255), "y": (0, 255, 0), "z": (255, 0, 0)}
_BBOX_EDGES = [
    (0, 1, "x"), (3, 2, "x"), (4, 5, "x"), (7, 6, "x"),
    (0, 3, "y"), (1, 2, "y"), (4, 7, "y"), (5, 6, "y"),
    (0, 4, "z"), (1, 5, "z"), (2, 6, "z"), (3, 7, "z"),
]


def _get_bbox_corners_gs(gs) -> np.ndarray:
    """Return the 8 AABB corners of the splat's centroid point cloud."""
    xyz = gs.get_xyz.detach().cpu().numpy()
    mn  = xyz.min(axis=0)
    mx  = xyz.max(axis=0)
    xmin, ymin, zmin = mn
    xmax, ymax, zmax = mx
    return np.array([
        [xmin, ymin, zmin], [xmax, ymin, zmin],
        [xmax, ymax, zmin], [xmin, ymax, zmin],
        [xmin, ymin, zmax], [xmax, ymin, zmax],
        [xmax, ymax, zmax], [xmin, ymax, zmax],
    ])


def _project_pts(pts_3d: np.ndarray, K: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Project Nx3 object-space points to Nx2 pixel coords via pose T."""
    R, t = T[:3, :3], T[:3, 3]
    pts_cam = (R @ pts_3d.T).T + t
    pts_h   = (K @ pts_cam.T).T
    return pts_h[:, :2] / pts_h[:, 2:3]


def _draw_bbox_3d(img_bgr: np.ndarray, corners_2d: np.ndarray, thickness: int = 2):
    for i, j, axis in _BBOX_EDGES:
        p1 = tuple(corners_2d[i].astype(int))
        p2 = tuple(corners_2d[j].astype(int))
        cv2.line(img_bgr, p1, p2, _AXIS_COLORS_BGR[axis], thickness, cv2.LINE_AA)

from src.pipeline.estimators.tracking_refiner_sam3d import TrackingRefinerSam3d
from src.pipeline.refiner_utils import smooth_transforms
from src.pipeline.retrieval.renderer_sam3d import load_gaussian, scale_gaussian

_FREEPOSE_ROOT = Path(__file__).resolve().parent.parent
os.environ["PYOPENGL_PLATFORM"] = "egl"


# ── Centroid projection helpers ────────────────────────────────────────────────

def _project_gs(gs, K: np.ndarray, T: np.ndarray, n_pts: int = 3000):
    """Project at most n_pts Gaussian centroids through pose T into the image.

    Returns:
        uv:    (M, 2) float pixel coords of visible centroids.
        depth: (M,)  z-depth in camera frame.
    """
    xyz = gs.get_xyz.detach().cpu().numpy()
    if len(xyz) > n_pts:
        idx = np.random.choice(len(xyz), n_pts, replace=False)
        xyz = xyz[idx]
    xyz_h   = np.concatenate([xyz, np.ones((len(xyz), 1))], axis=1).T  # (4, N)
    xyz_cam = (T @ xyz_h)[:3].T                                         # (N, 3)
    valid   = xyz_cam[:, 2] > 0
    if not valid.any():
        return np.empty((0, 2)), np.empty(0)
    proj  = (K @ xyz_cam[valid].T)                                      # (3, M)
    uv    = (proj[:2] / proj[2]).T                                       # (M, 2)
    return uv, xyz_cam[valid, 2]


def _make_centroid_mask(gs, w: int, h: int, K: np.ndarray, T: np.ndarray):
    """Rasterise Gaussian centroids into (rgb HxWx3, depth HxW)."""
    uv, z = _project_gs(gs, K, T, n_pts=6000)
    rgb   = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.zeros((h, w), dtype=np.float32)
    if len(uv):
        uv_int = np.clip(uv.astype(int), [0, 0], [w - 1, h - 1])
        depth[uv_int[:, 1], uv_int[:, 0]] = z
        rgb[uv_int[:, 1], uv_int[:, 0]]   = [200, 120, 50]
        kernel = np.ones((5, 5), np.uint8)
        depth  = cv2.dilate(depth, kernel, iterations=1)
        rgb    = cv2.dilate(rgb,   kernel, iterations=1)
    return rgb, depth


def create_outline(rgb_hw3, depth_hw):
    kernel   = np.ones((3, 3), np.uint8)
    mask_rgb = np.stack([np.uint8(depth_hw > 0)] * 3, 2) * 255
    mask_rgb = cv2.dilate(mask_rgb, kernel, iterations=2)
    canny    = cv2.Canny(mask_rgb, 30, 100)
    canny    = cv2.dilate(canny, kernel, iterations=2)
    outline  = np.clip(
        np.stack([canny] * 3, 2).astype(np.float32) * np.array([[[0.21, 0.49, 0.74]]]),
        0, 255,
    ).astype(np.uint8)
    return np.concatenate([outline, canny[:, :, np.newaxis]], 2)


# ── Tracking helpers (mirrors smooth_poses_video.py) ──────────────────────────

def predict_transforms_from_tracks(tracks, K):
    """Estimate per-frame poses from 2D-3D tracked correspondences via RANSAC PnP.

    Only CoTracker-visible points are used — invisible (extrapolated) tracks are
    excluded because their 2D positions are unreliable and corrupt the PnP solution.
    When RANSAC fails or too few visible points exist, the last successfully solved
    transform is carried forward so the interval never produces a degenerate pose.
    """
    _, _, p3d, p2d, pvis = tracks
    K64 = K.astype(np.float64)

    transforms  = []
    last_good_T = None

    for i in range(len(p2d)):
        vis_mask  = pvis[i].astype(bool)
        n_visible = int(vis_mask.sum())

        T = None
        if n_visible >= 4:
            p3d_i = p3d[vis_mask].astype(np.float64)
            p2d_i = p2d[i][vis_mask].astype(np.float64)
            success, rot, trans, inliers = cv2.solvePnPRansac(
                p3d_i, p2d_i, K64, np.array([]),
                flags=cv2.SOLVEPNP_EPNP,
                reprojectionError=8.0,
                confidence=0.99,
            )
            if success and inliers is not None and len(inliers) >= 4:
                T = np.eye(4)
                T[:3, :3] = cv2.Rodrigues(rot)[0]
                T[:3, 3]  = trans.reshape(-1)

        if T is None:
            if last_good_T is not None:
                logger.warning(
                    f"Frame offset {i}: PnP failed ({n_visible} visible pts); "
                    "carrying forward last good transform."
                )
                T = last_good_T.copy()
            else:
                logger.warning(
                    f"Frame offset {i}: PnP failed and no previous transform available; "
                    "using identity (this should not happen in normal operation)."
                )
                T = np.eye(4)
        else:
            last_good_T = T

        transforms.append(T)

    transforms = np.array(transforms)
    if len(transforms) == 0:
        raise RuntimeError("Got 0 poses from PnP!")
    return transforms


def predict_transforms_at_interval(
    frames, gs, K, masks, track_interval, out_interval,
    init_index, init_transform, tracref
):
    points2d, points3d = tracref.compute_2d3d_correspondences(
        gs, Image.fromarray(frames[init_index]), K, init_transform,
        mask=masks[init_index],
    )
    query_points = np.pad(
        points2d, [(0, 0), (1, 0)],
        constant_values=init_index - track_interval[0],
    )
    pred_tracks, pred_visibility = tracref._track_frames(
        frames[track_interval[0] : track_interval[1]], query_points
    )
    trackinfo = [init_index, out_interval, points3d, pred_tracks, pred_visibility]
    pred_transforms = predict_transforms_from_tracks(trackinfo, K)

    _from_ = out_interval[0] - track_interval[0]
    _to_   = out_interval[1] - track_interval[0]
    pred_transforms = pred_transforms[_from_:_to_]
    for ii in range(2, 5):
        trackinfo[ii] = trackinfo[ii][_from_:_to_]
    return pred_transforms, trackinfo


def predict_transforms(frames, transforms, gs, K, masks):
    tracref = TrackingRefinerSam3d(dino_device="cuda", cotracker_device="cuda")

    n_inliers, _ = tracref.n_inliers_per_pose(gs, frames, K, transforms)
    start_frame  = int(np.argmax(n_inliers))
    interval_len = 12

    boundaries     = np.round(np.linspace(0, len(frames), len(frames) // interval_len)).astype(int)
    out_intervals  = np.array(list(zip(boundaries[:-1], boundaries[1:])))
    track_intervals = np.clip(out_intervals.copy(), 0, len(frames))

    start_interval = int(np.where(
        (start_frame >= out_intervals[:, 0]) & (start_frame < out_intervals[:, 1])
    )[0][0])

    interval_indices    = [start_interval]
    interval_directions = [0]
    interval_indices   += list(range(start_interval + 1, track_intervals.shape[0]))
    interval_directions += [1] * (track_intervals.shape[0] - start_interval - 1)
    interval_indices   += list(range(start_interval))[::-1]
    interval_directions += [-1] * start_interval

    pred_transforms  = [None] * len(interval_indices)
    computed_tracks  = [None] * len(interval_indices)

    for i, direction in tqdm.tqdm(list(zip(interval_indices, interval_directions))):
        i0, i1 = track_intervals[i]
        if direction == 0:
            # Anchor the start interval at the highest-confidence stage-3 frame.
            init_index = start_frame
        elif direction == 1:
            # Use the first frame of this interval as the anchor.
            init_index = int(out_intervals[i][0])
        else:
            # Use the last frame of this interval as the anchor.
            init_index = int(out_intervals[i][1]) - 1

        # Always initialise from the stage-3 pose (with bbox-derived t) rather
        # than chaining from the previous interval's last PnP result.
        # Chaining propagates rotation errors: one bad interval corrupts every
        # subsequent one.  Stage-3 DINOv2 poses are accurate per-frame and give
        # a stable crop/correspondence estimate for every interval independently.
        init_T = transforms[init_index]

        pt, ct = predict_transforms_at_interval(
            frames=frames, gs=gs, K=K, masks=masks,
            track_interval=track_intervals[i], out_interval=out_intervals[i],
            init_index=init_index, init_transform=init_T, tracref=tracref,
        )
        pred_transforms[i] = pt
        computed_tracks[i]  = ct

    return np.concatenate(pred_transforms, axis=0), computed_tracks


# ── Visualisation ──────────────────────────────────────────────────────────────

def draw_line(rgb, coord_y, coord_x, color, linewidth):
    draw = ImageDraw.Draw(rgb)
    draw.line((coord_y[0], coord_y[1], coord_x[0], coord_x[1]),
              fill=tuple(color), width=linewidth)
    return rgb


def multiply_alpha(pil_image, alpha):
    return Image.fromarray(
        (np.array(pil_image) * np.array([1.0, 1.0, 1.0, alpha])).astype(np.uint8)
    )


def create_viz_imgs(frames, transforms_all, splats_all, K, computed_tracks_all, track_colors_all, t):
    h, w = frames[0].shape[:2]
    outline_img_all = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ren_img_all     = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    track_img_all   = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    for transforms, gs, computed_tracks, track_colors in zip(
        transforms_all, splats_all, computed_tracks_all, track_colors_all
    ):
        rgb_arr, depth_arr = _make_centroid_mask(gs, w, h, K, transforms[t])
        outline_img = Image.fromarray(create_outline(rgb_arr, depth_arr))
        ren_img     = Image.fromarray(
            np.concatenate([rgb_arr, np.uint8(depth_arr > 0)[:, :, np.newaxis] * 255], 2)
        )
        track_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))

        for track_idx, tracks in enumerate(computed_tracks):
            query_index, frame_interval, _, p2d, pvis = tracks
            if t < frame_interval[0] or t >= frame_interval[1]:
                continue
            n_points = p2d.shape[1]
            colors   = track_colors[track_idx]
            t_rel    = t - frame_interval[0]
            pts_cur  = p2d[t_rel]
            pts_prev = p2d[t_rel - 1] if t_rel > 0 else pts_cur + 1

            for i in list(range(n_points))[::4]:
                cy = (int(pts_prev[i, 0]), int(pts_prev[i, 1]))
                cx = (int(pts_cur[i, 0]),  int(pts_cur[i, 1]))
                if cy[0] != 0 and cy[1] != 0:
                    track_img = draw_line(track_img, cy, cx, colors[i], 3)

        ren_img_all.paste(ren_img,     (0, 0), mask=ren_img)
        outline_img_all.paste(outline_img, (0, 0), mask=outline_img)
        track_img_all.paste(track_img, (0, 0), mask=track_img)

    return ren_img_all, outline_img_all, track_img_all


def visualize(frames, transforms_all, splats_all, K, computed_tracks_all, output_path: Path):
    output_path.mkdir(parents=True, exist_ok=True)
    cmap = matplotlib.colormaps["gist_rainbow"]
    track_colors_all = []
    for computed_tracks in computed_tracks_all:
        track_colors = []
        for ti, tracks in enumerate(computed_tracks):
            _, _, _, p2d, _ = tracks
            n_pts = p2d.shape[1]
            track_colors.append(
                (cmap(np.linspace(0, 1 + 1e-10, n_pts)) * 255).astype(np.uint8)[
                    ::-1 if ti % 2 == 1 else 1
                ]
            )
        track_colors_all.append(track_colors)

    h, w = frames[0].shape[:2]
    track_img_history = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    for t in tqdm.tqdm(range(len(frames))):
        frame_img = Image.fromarray(frames[t])
        ren_img, outline_img, track_img = create_viz_imgs(
            frames, transforms_all, splats_all, K, computed_tracks_all, track_colors_all, t
        )
        track_img_history = multiply_alpha(track_img_history, 0.66)
        track_img_history.paste(track_img, (0, 0), mask=track_img)
        ren_img.putalpha(140)

        frame_img.paste(ren_img,          (0, 0), mask=ren_img)
        frame_img.paste(outline_img,      (0, 0), mask=outline_img)
        frame_img.paste(track_img_history,(0, 0), mask=track_img_history)
        frame_img.save(output_path / f"{t:06d}.png")


def visualize_bbox3d(frames, transforms_all, splats_all, K, output_path: Path):
    """Render only the 3-D RGB bounding box for each object onto each frame."""
    output_path.mkdir(parents=True, exist_ok=True)
    bbox_corners_all = [_get_bbox_corners_gs(gs) for gs in splats_all]

    for t in tqdm.tqdm(range(len(frames))):
        img_bgr = cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR)
        for transforms, corners in zip(transforms_all, bbox_corners_all):
            corners_2d = _project_pts(corners, K, transforms[t])
            _draw_bbox_3d(img_bgr, corners_2d)
        Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).save(
            output_path / f"{t:06d}.png"
        )


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    data_dir   = Path("data")
    frames_dir = data_dir / "datasets" / "videos" / args.video
    results_dir = data_dir / "results" / "sam3d" / args.video

    csv_path    = results_dir / args.poses
    frame_paths = sorted(p for p in frames_dir.iterdir()
                         if p.suffix.lower() in [".jpg", ".jpeg", ".png"])
    logger.info(f"Processing video '{args.video}' — {len(frame_paths)} frames")

    # Intrinsics
    K_file = results_dir / "K.txt"
    if K_file.exists():
        K = np.loadtxt(K_file)
    else:
        frame0 = Image.open(frame_paths[0])
        fw, fh = frame0.size
        f_px   = np.sqrt(fw**2 + fh**2)
        K      = np.array([[f_px, 0, fw / 2.0], [0, f_px, fh / 2.0], [0, 0, 1]])

    df_all   = pd.read_csv(csv_path)
    n_objects = len(list(itertools.takewhile(
        lambda x: x == df_all.iloc[0]["im_id"], df_all["im_id"]
    )))

    if args.obj_idxs is None:
        args.obj_idxs = list(range(n_objects))

    with open(results_dir / args.proposals) as f:
        proposals_all = json.load(f)

    # Debug directory
    debug_dir = _FREEPOSE_ROOT / "data" / "results" / "sam3d" / args.video / "05_tracked"
    debug_dir.mkdir(parents=True, exist_ok=True)

    pred_transforms_all = []
    splats_all          = []
    computed_tracks_all = []
    out_dfs             = []

    for obj_idx in args.obj_idxs:
        assert obj_idx < n_objects
        df        = df_all.iloc[list(range(len(df_all)))[obj_idx::n_objects]]
        proposals = proposals_all[obj_idx::n_objects]
        assert len(frame_paths) == len(df) == len(proposals)

        masks  = [rle_to_mask(p["segmentation"]) for p in proposals]
        frames = []
        transforms = []
        scale  = None

        for idx, row in enumerate(df.itertuples()):
            if idx == 0:
                scale = row.scale
            assert scale == row.scale

            T = np.eye(4)
            T[:3, :3] = np.array([float(x) for x in row.R.split()]).reshape(3, 3)
            T[:3,  3] = np.array([float(x) for x in row.t.split()])
            transforms.append(T)
            frames.append(np.array(Image.open(frame_paths[idx])))

        frames     = np.stack(frames)
        transforms = np.stack(transforms)

        # Load and scale Gaussian splat
        splat_path = df.iloc[0]["obj_id"]           # relative path stored in CSV
        ply_abs    = _FREEPOSE_ROOT / splat_path
        logger.info(f"Loading Gaussian splat for object {obj_idx}: {ply_abs}")
        gs = load_gaussian(ply_abs, device="cuda")
        gs = scale_gaussian(gs, scale)              # scale positions + kernels

        # The t stored in stage-3 CSV is in SAM-3D world units (z ≈ 2.0; unreliable).
        # Re-derive a metric t from the bbox to use as the INITIAL pose for the
        # tracking+PnP refinement.  The final output t comes from PnP (see below),
        # not from this approximation.
        gs_xyz  = gs.get_xyz.detach().cpu().numpy()       # already scaled → metric
        gs_diam = (gs_xyz.max(0) - gs_xyz.min(0)).max()
        for idx, prop in enumerate(proposals):
            x, y, w, h = prop["bbox"]                     # [x y w h] pixel coords
            bbox_px     = max(w, h) + 1.0
            z           = K[0, 0] * gs_diam / bbox_px
            cx, cy      = x + w / 2.0, y + h / 2.0
            transforms[idx][:3, 3] = [
                (cx - K[0, 2]) * z / K[0, 0],
                (cy - K[1, 2]) * z / K[1, 1],
                z,
            ]

        pred_transforms, computed_tracks = predict_transforms(
            frames, transforms, gs, K, masks=masks
        )
        # Keep PnP rotation (refined from stage-3 via 2D-3D tracking), but
        # replace translation with the per-frame bbox-derived estimate stored in
        # `transforms`.  PnP translation accumulates tracking errors across
        # chained intervals and drifts, whereas the bbox estimate is a fresh
        # pinhole observation for every frame that can't drift.
        pred_transforms[:, :3, 3] = transforms[:, :3, 3]
        pred_transforms = smooth_transforms(pred_transforms)

        df_out = df.copy()
        R_flat = pred_transforms[:, :3, :3].reshape(-1, 9)
        df_out["R"] = [" ".join(map(str, r)) for r in R_flat]
        t_flat = pred_transforms[:, :3, 3]
        df_out["t"] = [" ".join(map(str, t)) for t in t_flat]

        out_dfs.append(df_out)
        splats_all.append(gs)
        pred_transforms_all.append(pred_transforms)
        computed_tracks_all.append(computed_tracks)

    for i, df in enumerate(out_dfs):
        df.index = df.index * n_objects + i
    df_all = pd.concat(out_dfs).sort_index()

    out_csv = results_dir / f"{args.video}-tracked-sam3d.csv"
    df_all.to_csv(out_csv, index=False)
    logger.info(f"Saved refined poses → {out_csv}")

    if args.vis:
        cotracker_dir = debug_dir / "cotracker"
        visualize(
            frames, pred_transforms_all, splats_all, K,
            computed_tracks_all,
            output_path=cotracker_dir,
        )
        logger.info(f"Saved tracking visualisation → {cotracker_dir}")

        bbox3d_dir = debug_dir / "bbox3d"
        visualize_bbox3d(frames, pred_transforms_all, splats_all, K, output_path=bbox3d_dir)
        logger.info(f"Saved 3D bounding-box visualisation → {bbox3d_dir}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video",    type=str, required=True)
    parser.add_argument("--obj-idxs", type=int, default=None, nargs="+")
    parser.add_argument("--poses",    type=str, default=None)
    parser.add_argument("--proposals",type=str, default=None)
    parser.add_argument("--vis",      action="store_true")
    args = parser.parse_args()

    if args.poses is None and args.proposals is None:
        args.poses     = (f"sam3d_{args.video}_gpt4_scaled_best_object"
                          f"_dinopose_layer_22_bbext_0.05_depth_zoedepth.csv")
        args.proposals = f"sam3d_{args.video}_gpt4_scaled_best_object.json"

    main(args)
