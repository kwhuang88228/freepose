#!/usr/bin/env python3
"""
kalman_smooth_poses.py

Apply a Kalman filter to smooth 6D pose trajectories stored in a BOP-format CSV.

For each tracked object:
  - Translation (x, y, z): independent constant-velocity Kalman filter per coordinate.
  - Rotation: constant-velocity Kalman filter on the axis-angle (Rodrigues vector)
    representation; result is re-normalised back to SO(3) via Rotation.from_rotvec.

The approach is valid when inter-frame rotation changes are well under π rad, which
holds for any video-rate pose tracking sequence.

Usage:
    python -m scripts.kalman_smooth_poses \
        --video <video_name> \
        --poses <poses_filename.csv> \
        --backend {sam3d|mvsam3d}
    python -m scripts.kalman_smooth_poses \
        --video <video_name> \
        --poses <poses_filename.csv> \
        --backend {sam3d|mvsam3d} \
        --viz

Debug outputs (with --viz):
    data/results/<backend>/<video>/05_kalman_poses/bbox3d/
    data/results/<backend>/<video>/05_kalman_poses/bbox3d_axis/
"""

import itertools
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
from scipy.spatial.transform import Rotation

_FREEPOSE_ROOT = Path(__file__).resolve().parent.parent


def _make_cv_kf(dim: int, process_noise: float, meas_noise: float) -> KalmanFilter:
    """Constant-velocity Kalman filter for `dim` observed coordinates."""
    kf = KalmanFilter(dim_x=dim * 2, dim_z=dim)
    # State transition: position advances by velocity each step
    kf.F = np.eye(dim * 2)
    for i in range(dim):
        kf.F[i, dim + i] = 1.0
    # Observation: measure position only
    kf.H = np.zeros((dim, dim * 2))
    for i in range(dim):
        kf.H[i, i] = 1.0
    kf.R = np.eye(dim) * meas_noise
    kf.Q = np.eye(dim * 2) * process_noise
    kf.P = np.eye(dim * 2)
    return kf


def _kalman_smooth(seq: np.ndarray, process_noise: float, meas_noise: float) -> np.ndarray:
    """Forward-pass Kalman filter on an (N, D) sequence. Returns smoothed (N, D)."""
    N, D = seq.shape
    kf = _make_cv_kf(D, process_noise=process_noise, meas_noise=meas_noise)
    kf.x[:D] = seq[0].reshape(-1, 1)
    kf.x[D:] = 0.0
    out = np.empty_like(seq)
    for i in range(N):
        kf.predict()
        kf.update(seq[i])
        out[i] = kf.x[:D, 0]
    return out


def kalman_smooth_transforms(
    TCOs: np.ndarray,
    trans_process_noise: float = 1e-4,
    trans_meas_noise: float = 1e-2,
    rot_process_noise: float = 1e-4,
    rot_meas_noise: float = 1e-1,
) -> np.ndarray:
    """Kalman-smooth an (N, 4, 4) pose trajectory in-place copy."""
    TCOs = TCOs.copy()
    TCOs[:, :3, 3] = _kalman_smooth(
        TCOs[:, :3, 3], trans_process_noise, trans_meas_noise
    )
    rotvecs = Rotation.from_matrix(TCOs[:, :3, :3]).as_rotvec()
    TCOs[:, :3, :3] = Rotation.from_rotvec(
        _kalman_smooth(rotvecs, rot_process_noise, rot_meas_noise)
    ).as_matrix()
    return TCOs


def _save_bbox3d(img_rgb, gs_xyz, scale, K, R, box_xyxy, out_path):
    """Draw 3D bounding box edges on image. Red=X, Green=Y, Blue=Z."""
    from matplotlib import pyplot as plt

    xyz_min = gs_xyz.min(0) * scale
    xyz_max = gs_xyz.max(0) * scale
    gs_diameter_m = (gs_xyz.max(0) - gs_xyz.min(0)).max() * scale
    bbox_px = max(box_xyxy[2] - box_xyxy[0], box_xyxy[3] - box_xyxy[1]) + 1.0
    z_correct = K[0, 0] * gs_diameter_m / bbox_px
    bb_center = np.array([(box_xyxy[0] + box_xyxy[2]) / 2.0, (box_xyxy[1] + box_xyxy[3]) / 2.0])
    t = np.array([
        (bb_center[0] - K[0, 2]) * z_correct / K[0, 0],
        (bb_center[1] - K[1, 2]) * z_correct / K[1, 1],
        z_correct,
    ])

    xn, yn, zn = xyz_min
    xx, yx, zx = xyz_max
    corners = np.array([
        [xn, yn, zn], [xx, yn, zn], [xx, yx, zn], [xn, yx, zn],
        [xn, yn, zx], [xx, yn, zx], [xx, yx, zx], [xn, yx, zx],
    ])
    edges = {
        "r": [(0, 1), (3, 2), (4, 5), (7, 6)],
        "g": [(0, 3), (1, 2), (4, 7), (5, 6)],
        "b": [(0, 4), (1, 5), (2, 6), (3, 7)],
    }
    corners_cam = (R @ corners.T + t[:, None]).T
    proj = K @ corners_cam.T
    uv = (proj[:2] / proj[2]).T
    in_front = corners_cam[:, 2] > 0

    fig, ax = plt.subplots(1, 1, figsize=(img_rgb.shape[1] / 100, img_rgb.shape[0] / 100))
    ax.set_axis_off()
    ax.imshow(img_rgb)
    color_map = {"r": "red", "g": "lime", "b": "blue"}
    for axis_key, edge_list in edges.items():
        for i, j in edge_list:
            if not (in_front[i] and in_front[j]):
                continue
            ax.plot([uv[i, 0], uv[j, 0]], [uv[i, 1], uv[j, 1]],
                    color=color_map[axis_key], linewidth=1.5)
    ax.set_xlim(0, img_rgb.shape[1])
    ax.set_ylim(img_rgb.shape[0], 0)
    plt.savefig(str(out_path), bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _save_bbox3d_axis(img_rgb, gs_xyz, scale, K, R, box_xyxy, out_path):
    """Draw 3D bounding box + XYZ coordinate axes. Red=X, Green=Y, Blue=Z."""
    from matplotlib import pyplot as plt

    xyz_min = gs_xyz.min(0) * scale
    xyz_max = gs_xyz.max(0) * scale
    gs_diameter_m = (gs_xyz.max(0) - gs_xyz.min(0)).max() * scale
    bbox_px = max(box_xyxy[2] - box_xyxy[0], box_xyxy[3] - box_xyxy[1]) + 1.0
    z_correct = K[0, 0] * gs_diameter_m / bbox_px
    bb_center = np.array([(box_xyxy[0] + box_xyxy[2]) / 2.0, (box_xyxy[1] + box_xyxy[3]) / 2.0])
    t = np.array([
        (bb_center[0] - K[0, 2]) * z_correct / K[0, 0],
        (bb_center[1] - K[1, 2]) * z_correct / K[1, 1],
        z_correct,
    ])
    axis_len = gs_diameter_m * 0.5

    def _proj(pt_obj):
        cam = R @ pt_obj + t
        if cam[2] <= 0:
            return None
        p = K @ cam
        return p[:2] / p[2]

    origin_uv = _proj(np.zeros(3))
    x_tip_uv  = _proj(np.array([axis_len, 0.0, 0.0]))
    y_tip_uv  = _proj(np.array([0.0, axis_len, 0.0]))
    z_tip_uv  = _proj(np.array([0.0, 0.0, axis_len]))

    if origin_uv is None:
        import cv2
        cv2.imwrite(str(out_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        return

    xn, yn, zn = xyz_min
    xx, yx, zx = xyz_max
    corners = np.array([
        [xn, yn, zn], [xx, yn, zn], [xx, yx, zn], [xn, yx, zn],
        [xn, yn, zx], [xx, yn, zx], [xx, yx, zx], [xn, yx, zx],
    ])
    edges = {
        "r": [(0, 1), (3, 2), (4, 5), (7, 6)],
        "g": [(0, 3), (1, 2), (4, 7), (5, 6)],
        "b": [(0, 4), (1, 5), (2, 6), (3, 7)],
    }
    corners_cam = (R @ corners.T + t[:, None]).T
    proj_c = K @ corners_cam.T
    uv = (proj_c[:2] / proj_c[2]).T
    in_front = corners_cam[:, 2] > 0

    fig, ax = plt.subplots(1, 1, figsize=(img_rgb.shape[1] / 100, img_rgb.shape[0] / 100))
    ax.set_axis_off()
    ax.imshow(img_rgb)
    color_map = {"r": "red", "g": "lime", "b": "blue"}
    for axis_key, edge_list in edges.items():
        for i, j in edge_list:
            if not (in_front[i] and in_front[j]):
                continue
            ax.plot([uv[i, 0], uv[j, 0]], [uv[i, 1], uv[j, 1]],
                    color=color_map[axis_key], linewidth=1.5)
    for tip_uv, color in [(x_tip_uv, "red"), (y_tip_uv, "lime"), (z_tip_uv, "blue")]:
        if tip_uv is None:
            continue
        ax.annotate("", xy=(tip_uv[0], tip_uv[1]), xytext=(origin_uv[0], origin_uv[1]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.0))
    ax.scatter([origin_uv[0]], [origin_uv[1]], s=20, c="white", zorder=5)
    ax.set_xlim(0, img_rgb.shape[1])
    ax.set_ylim(img_rgb.shape[0], 0)
    plt.savefig(str(out_path), bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _run_viz(df_all, video_dir, viz_dir):
    """Render bbox3d and bbox3d_axis overlays for every row in the Kalman-smoothed CSV."""
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import torch
    from src.pipeline.retrieval.renderer_sam3d import load_gaussian

    device = "cuda" if torch.cuda.is_available() else "cpu"

    bbox3d_dir      = viz_dir / "bbox3d"
    bbox3d_axis_dir = viz_dir / "bbox3d_axis"
    bbox3d_dir.mkdir(parents=True, exist_ok=True)
    bbox3d_axis_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(video_dir.glob("*.png"))
    img0 = cv2.imread(str(frame_paths[0]))
    h, w = img0.shape[:2]
    f_px = float(np.sqrt(h ** 2 + w ** 2))
    K = np.array([[f_px, 0, w / 2.0], [0, f_px, h / 2.0], [0, 0, 1.0]])

    gs_xyz_cache = {}
    for obj_id in df_all["obj_id"].unique():
        ply_path = _FREEPOSE_ROOT / obj_id
        gs = load_gaussian(ply_path, device=device)
        gs_xyz_cache[obj_id] = gs.get_xyz.detach().cpu().numpy()

    for row in df_all.itertuples():
        R = np.array([float(x) for x in str(row.R).split()]).reshape(3, 3)
        bbox_xywh = np.array([float(x) for x in str(row.bbox_visib).split()])
        box_xyxy  = np.array([bbox_xywh[0], bbox_xywh[1],
                               bbox_xywh[0] + bbox_xywh[2], bbox_xywh[1] + bbox_xywh[3]])
        scale  = float(row.scale)
        gs_xyz = gs_xyz_cache[row.obj_id]

        img_bgr = cv2.imread(str(frame_paths[row.im_id]))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        obj_tag = Path(row.obj_id).stem
        tag = f"{row.im_id:06d}_{obj_tag}"

        try:
            _save_bbox3d(img_rgb, gs_xyz, scale, K, R, box_xyxy,
                         bbox3d_dir / f"{tag}.png")
        except Exception as exc:
            print(f"[warn] bbox3d failed at frame {row.im_id}: {exc}")

        try:
            _save_bbox3d_axis(img_rgb, gs_xyz, scale, K, R, box_xyxy,
                              bbox3d_axis_dir / f"{tag}.png")
        except Exception as exc:
            print(f"[warn] bbox3d_axis failed at frame {row.im_id}: {exc}")

    _dir_to_video(bbox3d_dir,      viz_dir / "bbox3d.mp4")
    _dir_to_video(bbox3d_axis_dir, viz_dir / "bbox3d_axis.mp4")
    print(f"Kalman viz → {viz_dir}")


def _dir_to_video(frames_dir: Path, out_path: Path, fps: int = 10) -> None:
    import cv2
    frames = sorted(frames_dir.glob("*.png"))
    if not frames:
        print(f"[warn] No frames in {frames_dir}, skipping video.")
        return
    first = cv2.imread(str(frames[0]))
    h_v, w_v = first.shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w_v, h_v))
    for fp in frames:
        writer.write(cv2.imread(str(fp)))
    writer.release()
    print(f"Video saved → {out_path}")


def main(args):
    data_dir = Path("data")
    results_dir = data_dir / "results" / args.backend / args.video
    csv_path = results_dir / args.poses

    df_all = pd.read_csv(csv_path)
    n_objects = len(list(itertools.takewhile(
        lambda x: x == df_all.iloc[0]["im_id"], df_all["im_id"]
    )))

    out_dfs = []
    for obj_idx in range(n_objects):
        df = df_all.iloc[list(range(len(df_all)))[obj_idx::n_objects]]

        transforms = []
        for row in df.itertuples():
            T = np.eye(4)
            T[:3, :3] = np.array([float(x) for x in row.R.split()]).reshape(3, 3)
            T[:3, 3]  = np.array([float(x) for x in row.t.split()])
            transforms.append(T)
        transforms = np.stack(transforms)

        transforms = kalman_smooth_transforms(transforms)

        df_out = df.copy()
        df_out["R"] = [" ".join(map(str, r)) for r in transforms[:, :3, :3].reshape(-1, 9)]
        df_out["t"] = [" ".join(map(str, t)) for t in transforms[:, :3, 3]]
        out_dfs.append(df_out)

    for i, df in enumerate(out_dfs):
        df.index = df.index * n_objects + i
    df_out_all = pd.concat(out_dfs).sort_index()

    out_csv = results_dir / (csv_path.stem + "_kalman.csv")
    df_out_all.to_csv(out_csv, index=False)
    print(f"Kalman-smoothed poses → {out_csv}")

    if args.viz:
        video_dir = (data_dir / "datasets" / "videos" / args.video).resolve()
        viz_dir   = results_dir / "05_kalman_poses"
        _run_viz(df_out_all, video_dir, viz_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video",   type=str, required=True)
    parser.add_argument("--poses",   type=str, required=True,
                        help="CSV filename (not full path) inside the results/<backend>/<video>/ dir.")
    parser.add_argument("--backend", type=str, choices=["sam3d", "mvsam3d"], required=True)
    parser.add_argument("--viz", action="store_true", default=False,
                        help="Save bbox3d and bbox3d_axis debug images to 05_kalman_poses/.")
    args = parser.parse_args()
    main(args)