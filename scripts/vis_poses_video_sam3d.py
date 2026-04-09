"""
vis_poses_video_sam3d.py

Stage 6 of the SAM-3D pipeline:
  1. Load per-frame 6D poses from the tracked CSV.
  2. Project Gaussian centroid positions onto each video frame.
  3. Optionally draw a 3D bounding-box overlay (--bbox).
  4. Write visualisation frames to viz_<stem>/ beside the CSV.

Debug outputs → data/results/sam3d/<video>/06_vis/
  Same frames mirrored for quick debugging access.
"""

import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

_FREEPOSE_ROOT = Path(__file__).resolve().parent.parent

os.environ["PYOPENGL_PLATFORM"] = "egl"

from src.pipeline.retrieval.renderer_sam3d import load_gaussian, scale_gaussian

DATA_DIR = Path("data")

# ── Axis/edge colours (BGR for cv2, same convention as vis_poses_video.py) ────
_AXIS_COLORS = {"x": (0, 0, 255), "y": (0, 255, 0), "z": (255, 0, 0)}
_BBOX_EDGES  = [
    (0, 1, "x"), (3, 2, "x"), (4, 5, "x"), (7, 6, "x"),
    (0, 3, "y"), (1, 2, "y"), (4, 7, "y"), (5, 6, "y"),
    (0, 4, "z"), (1, 5, "z"), (2, 6, "z"), (3, 7, "z"),
]


# ── Gaussian centroid helpers ──────────────────────────────────────────────────

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


def _project(pts_3d: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Project Nx3 object-space points to Nx2 pixel coords."""
    pts_cam = (R @ pts_3d.T).T + t
    pts_h   = (K @ pts_cam.T).T
    return pts_h[:, :2] / pts_h[:, 2:3]


def _draw_bbox_3d(img: np.ndarray, corners_2d: np.ndarray, thickness: int = 2):
    for i, j, axis in _BBOX_EDGES:
        p1 = tuple(corners_2d[i].astype(int))
        p2 = tuple(corners_2d[j].astype(int))
        cv2.line(img, p1, p2, _AXIS_COLORS[axis], thickness, cv2.LINE_AA)


def create_outline(mask: np.ndarray, color=(0.21, 0.49, 0.74)) -> np.ndarray:
    kernel   = np.ones((3, 3), np.uint8)
    mask_rgb = np.stack([np.uint8(mask)] * 3, 2) * 255
    mask_rgb = cv2.dilate(mask_rgb, kernel, iterations=2)
    canny    = cv2.Canny(mask_rgb, 30, 100)
    canny    = cv2.dilate(canny, kernel, iterations=2)
    outline  = np.clip(
        np.stack([canny] * 3, 2).astype(np.float32) * np.array([[color]]), 0, 255
    ).astype(np.uint8)
    return np.concatenate([outline, canny[:, :, np.newaxis]], 2)


def _render_centroid_overlay(gs, w: int, h: int, K: np.ndarray, T: np.ndarray,
                              n_pts: int = 5000) -> tuple:
    """Project Gaussian centroids → filled silhouette RGB + depth mask."""
    xyz = gs.get_xyz.detach().cpu().numpy()
    if len(xyz) > n_pts:
        idx = np.random.choice(len(xyz), n_pts, replace=False)
        xyz = xyz[idx]

    xyz_h   = np.concatenate([xyz, np.ones((len(xyz), 1))], axis=1).T  # (4,N)
    xyz_cam = (T @ xyz_h)[:3].T                                          # (N,3)
    valid   = xyz_cam[:, 2] > 0

    rgb   = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.zeros((h, w),    dtype=np.float32)

    if valid.any():
        proj    = (K @ xyz_cam[valid].T)            # (3,M)
        uv      = (proj[:2] / proj[2]).T.astype(int) # (M,2)
        z_vals  = xyz_cam[valid, 2]
        in_bounds = (
            (uv[:, 0] >= 0) & (uv[:, 0] < w) &
            (uv[:, 1] >= 0) & (uv[:, 1] < h)
        )
        uv_ib = uv[in_bounds]
        z_ib  = z_vals[in_bounds]
        depth[uv_ib[:, 1], uv_ib[:, 0]] = z_ib
        rgb[uv_ib[:, 1], uv_ib[:, 0]]   = [200, 120, 50]

        kernel = np.ones((11, 11), np.uint8)
        depth  = cv2.dilate(depth, kernel, iterations=2)
        rgb    = cv2.dilate(rgb,   kernel, iterations=2)

    return rgb, depth


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    pred_path = DATA_DIR / "results" / "sam3d" / args.video / args.predictions
    pred      = pd.read_csv(pred_path)
    viz_dir   = pred_path.parent / f"viz_{pred_path.stem}"
    viz_dir.mkdir(exist_ok=True, parents=True)

    # Debug dirs
    debug_mesh_dir = _FREEPOSE_ROOT / "data" / "results" / "sam3d" / args.video / "06_vis" / "mesh"
    debug_mesh_dir.mkdir(parents=True, exist_ok=True)
    debug_bbox_dir = _FREEPOSE_ROOT / "data" / "results" / "sam3d" / args.video / "06_vis" / "bbox3d"
    debug_bbox_dir.mkdir(parents=True, exist_ok=True)

    N = int((pred["im_id"] == 0).sum())
    objects_pred = [pred.iloc[i::N] for i in range(N)]

    splat_paths = [x.iloc[0]["obj_id"]   for x in objects_pred]
    scales      = [x.iloc[0]["scale"].item() for x in objects_pred]

    # Load Gaussian splats (already scaled)
    gaussian_splats = []
    bbox_corners_all = []
    for splat_path, scale in zip(splat_paths, scales):
        ply_abs = _FREEPOSE_ROOT / splat_path
        gs = load_gaussian(ply_abs, device="cuda")
        gs = scale_gaussian(gs, scale)
        gaussian_splats.append(gs)
        bbox_corners_all.append(_get_bbox_corners_gs(gs))

    video_path  = DATA_DIR / "datasets" / "videos" / args.video
    frame_paths = sorted(p for p in video_path.iterdir()
                         if p.suffix.lower() in [".jpg", ".jpeg", ".png"])

    h, w = np.array(Image.open(frame_paths[0])).shape[:2]
    f_px = np.sqrt(h**2 + w**2)
    K    = np.array([[f_px, 0, w / 2.0], [0, f_px, h / 2.0], [0, 0, 1]])


    for frame_idx, frame_path in tqdm(enumerate(frame_paths), ncols=100, total=len(frame_paths)):
        frame = Image.open(frame_path)

        # Collect poses; sort by z-distance (render far objects first)
        Ts = []
        for obj_idx in range(N):
            row = objects_pred[obj_idx].iloc[frame_idx]
            R   = np.array([float(x) for x in row["R"].split()]).reshape(3, 3)
            t   = np.array([float(x) for x in row["t"].split()])
            T   = np.eye(4)
            T[:3, :3] = R
            T[:3, 3]  = t
            Ts.append(T)

        distances = [np.linalg.norm(T[:3, 3]) for T in Ts]
        order     = np.argsort(distances)[::-1]

        # ── Centroid overlay ───────────────────────────────────────────────────
        for i in order:
            T  = Ts[i]
            gs = gaussian_splats[i]

            rgb_arr, depth_arr = _render_centroid_overlay(gs, w, h, K, T)
            mask    = depth_arr > 0
            outline = Image.fromarray(create_outline(mask))
            ren_img = Image.fromarray(
                np.concatenate([rgb_arr, np.uint8(mask)[:, :, np.newaxis] * 255], 2)
            )
            frame.paste(ren_img, (0, 0), ren_img)
            frame.paste(outline, (0, 0), outline)

        frame.save(viz_dir / f"{frame_idx:06d}.png")
        frame.save(debug_mesh_dir / f"{frame_idx:06d}.png")

        # ── 3D bounding-box overlay (--bbox) ──────────────────────────────────
        if args.bbox:
            img_bgr = cv2.cvtColor(np.array(Image.open(frame_path)), cv2.COLOR_RGB2BGR)
            for obj_idx, (gs, corners) in enumerate(zip(gaussian_splats, bbox_corners_all)):
                row = objects_pred[obj_idx].iloc[frame_idx]
                R   = np.array([float(x) for x in row["R"].split()]).reshape(3, 3)
                t   = np.array([float(x) for x in row["t"].split()])

                corners_2d = _project(corners, K, R, t)
                _draw_bbox_3d(img_bgr, corners_2d)

            Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).save(
                debug_bbox_dir / f"{frame_idx:06d}.png"
            )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video",       type=str, required=True)
    parser.add_argument("--predictions", "-p", type=str, required=True)
    parser.add_argument("--bbox",        action="store_true",
                        help="Also render a 3D bounding-box overlay showing 6DoF pose.")
    args = parser.parse_args()
    main(args)
