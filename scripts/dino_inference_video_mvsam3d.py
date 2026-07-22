"""
dino_inference_video_mvsam3d.py

Stage 4 of the MV-SAM3D pipeline:
  1. Load each object's Gaussian splat (path passed via the --mesh argument).
  2. Pre-render Hopf-Hammersley SO(3) views of the splat → coarse template bank.
  3. Run DinoOnlinePoseEstimatorSam3d per frame: coarse DINOv2 matching +
     fine-pose neighbourhood refinement.
  4. Write per-frame 6D poses to CSV.

Debug outputs → data/results/mvsam3d/<video>/04_coarse_poses/
  Centroid scatter projections for all frames.
"""

import argparse
import functools
import json
import os
import sys
from itertools import takewhile
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch
from loguru import logger
from matplotlib import pyplot as plt
from sam2.utils.amg import rle_to_mask
from tqdm import tqdm

from src.pipeline.estimators.online_pose_estimator_sam3d import DinoOnlinePoseEstimatorSam3d
from src.pipeline.estimators.pose_estimator import DinoPoseEstimator
from src.pipeline.retrieval.renderer_sam3d import (
    K_SAM3D,
    SplatRenderer,
    load_gaussian,
)
from src.pipeline.utils import Proposals

_FREEPOSE_ROOT = Path(__file__).resolve().parent.parent

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
os.environ["PYOPENGL_PLATFORM"] = "egl"


# ── Template dict builder ──────────────────────────────────────────────────────

def build_template_dict(gs, model_name: str, n_views: int = 3600,
                        bbox_extend: float = 0.05,
                        mask_template: bool = False, debug_dir=None):
    """Render Hopf-Hammersley SO(3) views of *gs* and package them as a template_dict.

    The format mirrors what WebTemplateDataset returns so DinoPoseEstimatorSam3d
    can use the same feature-extraction + caching machinery.

    Returns:
        dict with keys:
          'model_name'  str
          'templates'   Tensor [N, 3, H, W]  float, normalised 0-1
          'depths'      list of Tensor [H, W] float (metric depth)
          'intrinsic'   Tensor [3, 3] float  (K_SAM3D pixel-space)
    """
    logger.info(f"Rendering {n_views} Hopf-Hammersley SO(3) template views for {model_name}")
    renderer = SplatRenderer(n_views=n_views)
    renders  = renderer.render(gs)                                    # list of (rgb, depth, tcoinit)
    templates_cropped, tcoinits, masks_cropped = renderer.generate_proposals(renders, bbox_extend=bbox_extend, debug_dir=debug_dir)

    depths = [torch.from_numpy(renders[i][1]) for i in range(len(renders))]

    # Optionally zero out background pixels in template images before feature extraction.
    if mask_template:
        masks_tensor = torch.from_numpy(np.stack(masks_cropped).astype(np.float32)).unsqueeze(1)  # [N,1,H,W]
        templates_cropped = templates_cropped * masks_tensor

    # Pre-compute patch-level foreground masks from the CROPPED template masks.
    # masks_cropped matches the spatial layout of templates_cropped (what DINOv2 sees).
    # Shape: [N, num_patches] bool — used for masked mean pooling in the estimator.
    patch_masks = DinoPoseEstimator._to_patch_mask(np.stack(masks_cropped))   # [N, num_patches]

    if debug_dir is not None:
        _pm_dir = Path(debug_dir) / "patch_masks"
        _pm_dir.mkdir(parents=True, exist_ok=True)
        _patch_size = 14
        _H, _W = masks_cropped[0].shape
        _Ph, _Pw = _H // _patch_size, _W // _patch_size
        for _i, _pm in enumerate(patch_masks):
            _grid = _pm.reshape(_Ph, _Pw).numpy().astype(np.uint8) * 255
            _img_up = cv2.resize(_grid, (_W, _H), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(_pm_dir / f"{_i:04d}.png"), _img_up)

    # Mask flag + sampler tag + view count in the cache key:
    #   - mask flag avoids masked/unmasked feature collision
    #   - _hopf invalidates caches built with the older S²×S¹ product sampler
    #   - _n{N} prevents collision between runs at different total view counts
    #   - stem only: the key is joined onto cache_dir as a path, so separators in an
    #     absolute/relative mesh path would write the cache outside cache_dir
    cache_name = f"{Path(model_name).stem}_tmpl{'m' if mask_template else 'u'}_hopf_n{n_views}"
    # Store SAM-3D TCO_init matrices in the renderer for get_z_from_pointcloud
    return {
        "model_name":    cache_name,
        "templates":     templates_cropped,                            # [N,3,H,W]
        "depths":        depths,                                       # list of [H,W] Tensor
        "intrinsic":     torch.from_numpy(K_SAM3D).float(),
        "patch_masks":   patch_masks,                                  # [N, num_patches] bool
        "_tcoinits":     tcoinits,                                     # list of (4,4) np arrays
        "_renderer":     renderer,                                     # kept for fine-pose use
        "_masks_pixel":  np.stack(masks_cropped),                     # [N, H, W] bool
        "_xyzs":         renderer._xyz,                                # (N, 3) Hopf base-point unit vectors on S²
        "_rolls":        np.array(renderer._rolls),                    # (N,) in-plane roll angles (rad)
    }


# ── Debug visualisation ────────────────────────────────────────────────────────

def _save_bbox_3d(img, gs, K, R, t, scale, box, out_path: Path):
    """Draw the 3D bounding box of the Gaussian splat on the image.

    Edges parallel to X are drawn in Red, Y in Green, Z in Blue.
    Translation t is supplied by the caller (DA3-based when available; see main()).
    """
    gs_xyz_world = gs.get_xyz.detach().cpu().numpy()
    xyz_min = gs_xyz_world.min(0) * scale
    xyz_max = gs_xyz_world.max(0) * scale

    # 8 corners of the 3D bounding box in object space
    xn, yn, zn = xyz_min
    xx, yx, zx = xyz_max
    corners = np.array([
        [xn, yn, zn],  # 0
        [xx, yn, zn],  # 1
        [xx, yx, zn],  # 2
        [xn, yx, zn],  # 3
        [xn, yn, zx],  # 4
        [xx, yn, zx],  # 5
        [xx, yx, zx],  # 6
        [xn, yx, zx],  # 7
    ])

    # 12 edges grouped by axis (R=X, G=Y, B=Z)
    edges = {
        "r": [(0, 1), (3, 2), (4, 5), (7, 6)],   # X-parallel → Red
        "g": [(0, 3), (1, 2), (4, 7), (5, 6)],   # Y-parallel → Green
        "b": [(0, 4), (1, 5), (2, 6), (3, 7)],   # Z-parallel → Blue
    }

    # Project corners to image
    corners_cam = (R @ corners.T + t[:, None]).T   # (8, 3)
    proj = K @ corners_cam.T                        # (3, 8)
    uv = (proj[:2] / proj[2]).T                    # (8, 2)
    in_front = corners_cam[:, 2] > 0

    fig, ax = plt.subplots(1, 1, figsize=(img.shape[1] / 100, img.shape[0] / 100))
    ax.set_axis_off()
    ax.imshow(img)

    color_map = {"r": "red", "g": "lime", "b": "blue"}
    for axis_key, edge_list in edges.items():
        color = color_map[axis_key]
        for i, j in edge_list:
            if not (in_front[i] and in_front[j]):
                continue
            ax.plot([uv[i, 0], uv[j, 0]], [uv[i, 1], uv[j, 1]],
                    color=color, linewidth=1.5)

    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    plt.savefig(str(out_path), bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _save_bbox3d_axis(img, gs, K, R, t, scale, box, out_path: Path):
    """Draw XYZ coordinate axes and 3D bounding box at the projected object centroid.

    Red=X, Green=Y, Blue=Z. Axis length = half the object diameter.
    Translation t is supplied by the caller (DA3-based when available; see main()).
    """
    gs_xyz_world  = gs.get_xyz.detach().cpu().numpy()
    xyz_min = gs_xyz_world.min(0) * scale
    xyz_max = gs_xyz_world.max(0) * scale
    gs_diameter_m = (gs_xyz_world.max(0) - gs_xyz_world.min(0)).max() * scale
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
        plt.imsave(str(out_path), img)
        return

    # 8 corners of the 3D bounding box in object space
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

    fig, ax = plt.subplots(1, 1, figsize=(img.shape[1] / 100, img.shape[0] / 100))
    ax.set_axis_off()
    ax.imshow(img)

    color_map = {"r": "red", "g": "lime", "b": "blue"}
    for axis_key, edge_list in edges.items():
        color = color_map[axis_key]
        for i, j in edge_list:
            if not (in_front[i] and in_front[j]):
                continue
            ax.plot([uv[i, 0], uv[j, 0]], [uv[i, 1], uv[j, 1]], color=color, linewidth=1.5)

    for tip_uv, color in [
        (x_tip_uv, "red"),
        (y_tip_uv, "lime"),
        (z_tip_uv, "blue"),
    ]:
        if tip_uv is None:
            continue
        ax.annotate("", xy=(tip_uv[0], tip_uv[1]), xytext=(origin_uv[0], origin_uv[1]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.0))
    ax.scatter([origin_uv[0]], [origin_uv[1]], s=20, c="white", zorder=5)
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    plt.savefig(str(out_path), bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _save_axis(img, gs, K, R, t, scale, box, out_path: Path):
    """Draw only the XYZ coordinate axes at the projected object centroid.

    Red=X, Green=Y, Blue=Z. Axis length = half the object diameter. Same t
    re-derivation as _save_bbox3d_axis, but without the 3D bounding box.
    """
    gs_xyz_world  = gs.get_xyz.detach().cpu().numpy()
    gs_diameter_m = (gs_xyz_world.max(0) - gs_xyz_world.min(0)).max() * scale
    bbox_px       = max(box[2] - box[0], box[3] - box[1]) + 1.0
    z_correct     = K[0, 0] * gs_diameter_m / bbox_px
    bb_center     = np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])
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

    fig, ax = plt.subplots(1, 1, figsize=(img.shape[1] / 100, img.shape[0] / 100))
    ax.set_axis_off()
    ax.imshow(img)

    if origin_uv is not None:
        for tip_uv, color in [
            (x_tip_uv, "red"),
            (y_tip_uv, "lime"),
            (z_tip_uv, "blue"),
        ]:
            if tip_uv is None:
                continue
            ax.annotate("", xy=(tip_uv[0], tip_uv[1]), xytext=(origin_uv[0], origin_uv[1]),
                        arrowprops=dict(arrowstyle="->", color=color, lw=2.0))
        ax.scatter([origin_uv[0]], [origin_uv[1]], s=20, c="white", zorder=5)

    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    plt.savefig(str(out_path), bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _save_centroid_projection(img, gs, K, R, t, scale, box, out_path: Path, n_pts: int = 2000):
    """Scatter Gaussian centroids projected onto the image frame.

    Translation t is supplied by the caller (DA3-based metric depth when available,
    else the bbox-size heuristic; see main()), so the projected cloud appears at the
    estimated depth.
    """
    xyz = gs.get_xyz.detach().cpu().numpy() * scale

    if len(xyz) > n_pts:
        idx = np.random.choice(len(xyz), n_pts, replace=False)
        xyz = xyz[idx]
    xyz_cam = (R @ xyz.T + t[:, None]).T          # (N,3) in camera frame
    valid   = xyz_cam[:, 2] > 0
    proj    = (K @ xyz_cam[valid].T)              # (3,M)
    uv      = (proj[:2] / proj[2]).T              # (M,2)

    fig, ax = plt.subplots(1, 1, figsize=(img.shape[1] / 100, img.shape[0] / 100))
    ax.set_axis_off()
    ax.imshow(img)
    ax.scatter(uv[:, 0], uv[:, 1], s=1, alpha=0.5, c="cyan")
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    plt.savefig(str(out_path), bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _mask_overlay(img_rgb_u8: np.ndarray, mask_hw: np.ndarray) -> np.ndarray:
    """Return a BGR uint8 image with *mask_hw* blended as a red tint."""
    overlay = img_rgb_u8.copy()
    overlay[mask_hw.astype(bool), 0] = np.clip(
        overlay[mask_hw.astype(bool), 0].astype(np.int32) + 80, 0, 255
    ).astype(np.uint8)
    blended = cv2.addWeighted(img_rgb_u8, 0.6, overlay, 0.4, 0)
    return cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)


def _feature_mask_img(patch_mask_1d, H: int, W: int, patch_size: int = 14) -> np.ndarray:
    """Upsample a flat bool patch mask to a [H, W] uint8 grayscale image."""
    Ph, Pw = H // patch_size, W // patch_size
    grid = patch_mask_1d.reshape(Ph, Pw)
    if torch.is_tensor(grid):
        grid = grid.cpu().numpy()
    return cv2.resize(grid.astype(np.uint8) * 255, (W, H), interpolation=cv2.INTER_NEAREST)


def _dir_to_video(frames_dir: Path, out_path: Path, fps: int = 30) -> None:
    frames = sorted(frames_dir.glob("*.png"))
    if not frames:
        logger.warning(f"No frames in {frames_dir}, skipping video.")
        return
    first  = cv2.imread(str(frames[0]))
    h_v, w_v = first.shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w_v, h_v))
    for fp in frames:
        writer.write(cv2.imread(str(fp)))
    writer.release()
    logger.info(f"Video saved → {out_path}")


# ── DA3-depth translation z ──────────────────────────────────────────────────────

def _load_da3_depths(npz_path):
    """Load DA3 depth maps → {frame_stem: (H, W) float32}.

    Kept at DA3's native resolution; masks are downsized to match at sample time
    (see _da3_z), which is cheaper than upsampling every depth map.
    """
    data  = np.load(npz_path, allow_pickle=True)
    depth = data["depth"]                                   # (N, H, W)
    stems = [Path(str(f)).stem for f in data["image_files"]]
    return {stem: np.asarray(depth[i], dtype=np.float32) for i, stem in enumerate(stems)}


def _load_da3_window_id(npz_path):
    """Load the per-frame DA3 window id, or None if the npz predates windowing.

    DA3 processes long videos in overlapping windows and stitches them with one
    scalar per seam fit over the whole (background-dominated) image. That scalar
    makes the far background continuous but leaves the near foreground object
    mis-scaled, so the object depth steps ~1.3x at each seam (see _deseam_object_depth).
    The window id marks where those seams are so they can be removed on the object.
    """
    data = np.load(npz_path, allow_pickle=True)
    if "window_id" not in data.files:
        return None
    return np.asarray(data["window_id"]).astype(int)


def _deseam_object_depth(depth_series, window_id):
    """Per-frame multiplicative factors that align every DA3 window's object depth
    to the first window's scale, removing the between-window seam steps.

    depth_series : (N,) raw object median DA3 depth (NaN where the object is missing).
    window_id    : (N,) DA3 window each frame came from (monotone, steps at seams).

    Intra-window depth is left untouched; at each seam the step is estimated from
    the robust median depth of the 3 frames on each side (object motion within a
    window is smooth, so a step there is the artifact) and divided out cumulatively.
    The first window is the anchor (factor 1.0) because the metric scale `alpha`
    is calibrated on it. Returns (N,) factors (1.0 everywhere if no seams).
    """
    depth_series = np.asarray(depth_series, dtype=float)
    n = len(depth_series)
    factors = np.ones(n)
    if window_id is None:
        return factors
    cum = 1.0
    for b in range(1, n):
        if window_id[b] == window_id[b - 1]:
            continue
        before = np.nanmedian(depth_series[max(0, b - 3):b])
        after  = np.nanmedian(depth_series[b:b + 3])
        if np.isfinite(before) and np.isfinite(after) and before > 0 and after > 0:
            cum /= (after / before)
        factors[b:] = cum
    return factors


def _da3_z(depth_map, mask, alpha, erode_px: int = 6, min_px: int = 25):
    """Robust metric object depth: alpha * median of DA3 depth over the (eroded)
    object mask. Returns None if depth/mask are missing or too small to trust."""
    if depth_map is None or mask is None:
        return None
    m = np.asarray(mask).astype(bool)
    if m.sum() < min_px:
        return None
    Hd, Wd = depth_map.shape[:2]
    if m.shape != (Hd, Wd):
        m = cv2.resize(m.astype(np.uint8), (Wd, Hd), interpolation=cv2.INTER_NEAREST).astype(bool)
    m_e = cv2.erode(m.astype(np.uint8), np.ones((erode_px, erode_px), np.uint8)).astype(bool)
    if m_e.sum() >= min_px:                                 # keep raw mask if too thin to erode
        m = m_e
    vals = depth_map[m]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size < min_px:
        return None
    med  = float(np.median(vals))
    keep = np.abs(vals - med) < 1.5 * (float(np.std(vals)) + 1e-6)
    if keep.sum() >= min_px:
        med = float(np.median(vals[keep]))
    return alpha * med


def _bbox_z(gs, scale, box_xyxy, K):
    """Original bbox-size depth heuristic: z = f * (mesh diameter * scale) / bbox_px."""
    gs_xyz = gs.get_xyz.detach().cpu().numpy()
    diam   = (gs_xyz.max(0) - gs_xyz.min(0)).max() * scale
    bpx    = max(box_xyxy[2] - box_xyxy[0], box_xyxy[3] - box_xyxy[1]) + 1.0
    return float(K[0, 0] * diam / bpx)


def _override_translation(TCO, z, box_xyxy, K):
    """Return a copy of TCO with translation set from the bbox center at depth z."""
    cx = (box_xyxy[0] + box_xyxy[2]) / 2.0
    cy = (box_xyxy[1] + box_xyxy[3]) / 2.0
    TCO = TCO.copy()
    TCO[0, 3] = (cx - K[0, 2]) * z / K[0, 0]
    TCO[1, 3] = (cy - K[1, 2]) * z / K[1, 1]
    TCO[2, 3] = z
    return TCO


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    video_dir   = (Path("data") / "datasets" / "videos" / args.video).resolve()
    frame_names = sorted([p for p in video_dir.iterdir() if p.suffix.lower() in [".png"]])

    results_dir    = (Path(args.results_dir) if args.results_dir
                      else Path("data") / "results" / "mvsam3d" / args.video).resolve()
    proposals_path = results_dir / args.proposals

    pose_outputs = results_dir / args.proposals.replace(
        ".json",
        f"_dinopose_layer_{args.layer}_bbext_{args.bbox_extend}_depth_{args.depth_method}"
        f"_qimg{'m' if args.mask_query else 'u'}"
        f"_timg{'m' if args.mask_template else 'u'}"
        f"_qpatch{'fg' if args.query_fg_patches else 'all'}"
        f"_tpatch{'fg' if args.template_fg_patches else 'all'}"
        f"_n{args.num_templates}"
        f".csv",
    )

    # Debug directories
    debug_dir = results_dir / "04_coarse_poses"
    debug_gaussian_dir       = debug_dir / "gaussian"
    debug_bbox3d_dir         = debug_dir / "bbox3d"
    debug_axis_dir           = debug_dir / "axis"
    debug_bbox3d_axis_dir    = debug_dir / "bbox3d_axis"
    debug_retrieved_tmpl_dir = debug_dir / "retrieved_templates"
    # debug_segmented_dir        = debug_dir / "object_segmented"
    debug_gaussian_dir.mkdir(parents=True, exist_ok=True)
    debug_bbox3d_dir.mkdir(parents=True, exist_ok=True)
    debug_axis_dir.mkdir(parents=True, exist_ok=True)
    debug_bbox3d_axis_dir.mkdir(parents=True, exist_ok=True)
    debug_retrieved_tmpl_dir.mkdir(parents=True, exist_ok=True)
    # debug_segmented_dir.mkdir(parents=True, exist_ok=True)

    _dbg = debug_dir / "debug"
    _dbg_query_img         = _dbg / "query_img"
    _dbg_query_pmask       = _dbg / "query_pixel_mask"
    _dbg_query_overlay     = _dbg / "query_img_pixel_mask_overlay"
    _dbg_query_fmask       = _dbg / "query_feature_mask"
    _dbg_query_raw         = _dbg / "query_raw"
    _dbg_tmpl_img          = _dbg / "template_img"
    _dbg_tmpl_pmask        = _dbg / "template_pixel_mask"
    _dbg_tmpl_overlay      = _dbg / "template_img_pixel_mask_overlay"
    _dbg_tmpl_fmask        = _dbg / "template_feature_mask"
    for _d in [_dbg_query_img, _dbg_query_pmask, _dbg_query_overlay, _dbg_query_fmask,
               _dbg_query_raw, _dbg_tmpl_img, _dbg_tmpl_pmask, _dbg_tmpl_overlay, _dbg_tmpl_fmask]:
        _d.mkdir(parents=True, exist_ok=True)
    # ── Load proposals ─────────────────────────────────────────────────────────
    with open(proposals_path) as f:
        props = json.load(f)

    n_objects = len(list(takewhile(lambda x: x["image_id"] == 0, props)))
    n_frames  = len(frame_names)
    assert n_objects * n_frames == len(props), (
        f"Expected {n_objects * n_frames} proposals, got {len(props)}"
    )
    props = [props[i : i + n_objects] for i in range(0, len(props), n_objects)]

    # ── Scale estimation ───────────────────────────────────────────────────────
    # "da3": use the per-object scale precomputed in the proposals JSON (DA3 depth
    # + CLIP, from compute_scale_video.py). Asserted constant across frames.
    if args.depth_method == "da3":
        scales = [props[0][obj_idx]["scale"] for obj_idx in range(n_objects)]
        for i in range(n_objects):
            assert all(
                props[frame_idx][i]["scale"] == scales[i] for frame_idx in range(n_frames)
            ), f"Object {i} has inconsistent scales across frames."
    elif args.depth_method.startswith("const-"):
        val    = float(args.depth_method.split("-")[1])
        scales = [val] * n_objects
    else:
        raise NotImplementedError(f"Unknown depth method: {args.depth_method}")

    # ── Load Gaussian splats + build template dicts ────────────────────────────
    logger.info("Loading Gaussian splats and building template banks")
    splat_paths    = []
    gaussian_splats = []
    template_dicts  = []

    for i in range(n_objects):
        splat_path = args.mesh                    # relative path, e.g. "data/gaussian_splats/..."

        ply_abs = _FREEPOSE_ROOT / splat_path
        logger.info(f"  Object {i}: {ply_abs}")
        gs = load_gaussian(ply_abs, device=device)

        tdict = build_template_dict(
            gs,
            model_name=splat_path,
            n_views=args.num_templates,
            bbox_extend=args.bbox_extend,
            mask_template=args.mask_template,
            debug_dir=debug_dir / "render_sam3d_debug",
        )
        # Inject SAM-3D TCO_inits into model's mesh_poses placeholder (set later)
        splat_paths.append(splat_path)
        gaussian_splats.append(gs)
        template_dicts.append(tdict)

    # ── Build intrinsic from first frame ───────────────────────────────────────
    img0 = cv2.cvtColor(cv2.imread(str(frame_names[0])), cv2.COLOR_BGR2RGB).astype(np.uint8)
    h, w = img0.shape[:2]
    f_px = np.sqrt(h**2 + w**2)
    K    = np.array([[f_px, 0, w / 2.0], [0, f_px, h / 2.0], [0, 0, 1]], dtype=float)

    # ── DA3 depth for translation z (optional) ─────────────────────────────────
    # If a DA3 npz is given, object translation z is measured from DA3 depth at the
    # object mask (robust to occlusion) instead of the bbox-size heuristic. DA3-nested
    # depth may be up to a global scale, so anchor it per object with a single factor
    # alpha = z_bbox / z_da3 on that object's best-visible (largest-mask) frame, where
    # the bbox heuristic is reliable. --assume_metric skips the anchor (alpha=1).
    da3_depths = None
    alphas     = [1.0] * n_objects
    deseam     = [np.ones(n_frames) for _ in range(n_objects)]        # per-window object-depth seam correction
    if args.da3_depth:
        da3_depths = _load_da3_depths(args.da3_depth)
        window_id  = _load_da3_window_id(args.da3_depth)
        for obj_idx in range(n_objects):
            # Per-frame raw (alpha=1) object depth + track the best-visible frame.
            depth_series    = np.full(n_frames, np.nan)
            best_area, best = -1, None
            for f in range(n_frames):
                m  = rle_to_mask(props[f][obj_idx]["segmentation"])
                dz = _da3_z(da3_depths.get(frame_names[f].stem), m, alpha=1.0)
                if dz is not None:
                    depth_series[f] = dz
                a = int(m.sum())
                if a > best_area:
                    best_area, best = a, (f, m, np.array(props[f][obj_idx]["bbox"], dtype=float))
            # Divide out DA3 between-window depth seams on the object (window 0 = anchor).
            deseam[obj_idx] = _deseam_object_depth(depth_series, window_id)
            f_b, m_b, box_b = best
            box_b[2:] += box_b[:2]                                     # xywh → xyxy
            z_raw = depth_series[f_b] * deseam[obj_idx][f_b]           # calibrate on de-seamed depth
            if args.assume_metric or not np.isfinite(z_raw):
                alphas[obj_idx] = 1.0
            else:
                alphas[obj_idx] = _bbox_z(gaussian_splats[obj_idx], scales[obj_idx], box_b, K) / z_raw
            n_seams = int((np.diff(deseam[obj_idx]) != 0).sum())
            logger.info(f"[da3] obj{obj_idx}: anchor frame={f_b} mask_area={best_area} "
                        f"alpha={alphas[obj_idx]:.4f} seams_corrected={n_seams}")

    # ── Instantiate estimator ──────────────────────────────────────────────────
    SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", 0)
    cache_dir    = Path("data") / f"cache_{SLURM_JOB_ID}_{args.video}"
    model = DinoOnlinePoseEstimatorSam3d(
        n_coarse_poses=args.num_templates,
        n_fine_poses=20000,
        cache_size=args.cache_size,
        save_all=args.save_all_cache,
        cache_dir=str(cache_dir),
    ).to(device, dtype=torch.bfloat16)

    # Patch coarse estimator's mesh_poses with the correct SAM-3D TCO_inits
    # (one set per object; use object 0's renderer since all share the same Hopf-Hammersley sequence)
    model.coarse_estimator.mesh_poses = template_dicts[0]["_tcoinits"]

    # ── Per-frame inference ────────────────────────────────────────────────────
    results_dict = {
        "scene_id": [], "im_id": [], "obj_id": [], "score": [],
        "R": [], "t": [], "bbox_visib": [], "scale": [], "time": [],
        # Top-5 candidates for downstream symmetry canonicalization. Each row stores
        # 5 pipe-separated rotations / translations (9 / 3 space-separated floats per
        # candidate) and 5 space-separated DINO scores, ranked by score descending.
        "R_top5": [], "t_top5": [], "score_top5": [],
    }

    prev_poses = [None] * n_objects
    for frame_idx, frame_name in enumerate(tqdm(frame_names, ncols=100)):
        scene_proposals = props[frame_idx]
        assert all(p["image_id"] == frame_idx for p in scene_proposals)

        img   = cv2.cvtColor(cv2.imread(str(frame_name)), cv2.COLOR_BGR2RGB).astype(np.uint8)
        masks = [rle_to_mask(p["segmentation"]) for p in scene_proposals]
        boxes = [np.array(p["bbox"]) for p in scene_proposals]
        scores_prop = [p["score"] for p in scene_proposals]

        masks_t = torch.from_numpy(np.stack(masks))
        boxes_t = torch.from_numpy(np.stack(boxes))
        boxes_t[:, 2:] += boxes_t[:, :2]           # xywh → xyxy
        proposals = Proposals(img, {"boxes": boxes_t, "masks": masks_t}, 512,
                              bbox_extend=args.bbox_extend, mask_rgb=args.mask_query)
        proposals.scores = scores_prop

        for obj_idx in range(n_objects):
            prop      = proposals.proposals[obj_idx]
            prop_mask = proposals.proposals_masks[obj_idx]
            box       = boxes_t[obj_idx]
            gs        = gaussian_splats[obj_idx]
            tdict     = template_dicts[obj_idx]
            scale     = scales[obj_idx]

            # Save the segmented object image used for template comparison
            # prop_np  = (prop.permute(1, 2, 0).cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)
            # prop_bgr = cv2.cvtColor(prop_np, cv2.COLOR_RGB2BGR)
            # cv2.imwrite(str(debug_segmented_dir / f"{frame_idx:06d}_obj{obj_idx}.png"), prop_bgr)

            with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16):
                out = model.coarse_estimator.forward(
                    prop, tdict, K, box, scale,
                    layer=args.layer,
                    batch_size=args.batch_size,
                    proposal_mask=prop_mask,
                    use_query_fg_patches=args.query_fg_patches,
                    use_template_fg_patches=args.template_fg_patches,
                    top_k=args.top_n_candidates,
                )

            # Override translation z with DA3 metric depth at the object mask
            # (fallback: bbox-size heuristic). Applied to every candidate so the CSV
            # t, t_top5, and the debug overlays are all consistent.
            if da3_depths is not None:
                box_np_o = box.cpu().numpy().astype(float)
                z_da3 = _da3_z(da3_depths.get(frame_name.stem), masks[obj_idx], alphas[obj_idx])
                if z_da3 is None:
                    z_da3 = _bbox_z(gs, scale, box_np_o, K)
                else:
                    z_da3 *= deseam[obj_idx][frame_idx]               # remove DA3 window-seam step
                out["TCO"] = [_override_translation(tco, z_da3, box_np_o, K) for tco in out["TCO"]]
            prev_poses[obj_idx] = out["TCO"][0]

            # Save the top-5 retrieved template images for this frame / object
            for rank, (tmpl, tmpl_id, sim) in enumerate(
                zip(out["retrieved_proposals"], out["retrieved_template_ids"], out["scores"])
            ):
                tmpl_np  = (tmpl.permute(1, 2, 0).cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)
                tmpl_bgr = cv2.cvtColor(tmpl_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    str(debug_retrieved_tmpl_dir / f"{frame_idx:06d}_obj{obj_idx}_rank{rank}_{tmpl_id:04d}_{sim:.4f}.png"),
                    tmpl_bgr,
                )

            # ── Debug outputs 1-8 ─────────────────────────────────────────────
            try:
                _tag = f"{frame_idx:06d}_obj{obj_idx}"
                _H, _W = prop.shape[1], prop.shape[2]

                # 0. query_raw (unmasked crop)
                _raw_crop = proposals.rgb_proposal_processor(
                    proposals.image.unsqueeze(0),
                    proposals.boxes[obj_idx:obj_idx + 1],
                )[0]
                _raw_np = (_raw_crop.permute(1, 2, 0).cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)
                cv2.imwrite(str(_dbg_query_raw / f"{_tag}.png"),
                            cv2.cvtColor(_raw_np, cv2.COLOR_RGB2BGR))

                # 1. query_img
                _prop_np = (prop.permute(1, 2, 0).cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)
                cv2.imwrite(str(_dbg_query_img / f"{_tag}.png"),
                            cv2.cvtColor(_prop_np, cv2.COLOR_RGB2BGR))

                # 2. query_pixel_mask
                _pmask_np = prop_mask.cpu().numpy().astype(np.uint8) * 255
                cv2.imwrite(str(_dbg_query_pmask / f"{_tag}.png"), _pmask_np)

                # 3. query_img_pixel_mask_overlay
                _pmask_bool = prop_mask.cpu().numpy().astype(bool)
                cv2.imwrite(str(_dbg_query_overlay / f"{_tag}.png"),
                            _mask_overlay(_prop_np, _pmask_bool))

                # 4. query_feature_mask
                _q_fmask = DinoPoseEstimator._to_patch_mask(prop_mask.cpu().numpy())
                cv2.imwrite(str(_dbg_query_fmask / f"{_tag}.png"),
                            _feature_mask_img(_q_fmask, _H, _W))

                # 5-8. top-5 template outputs
                for rank, (tmpl, tmpl_id, sim) in enumerate(
                    zip(out["retrieved_proposals"], out["retrieved_template_ids"], out["scores"])
                ):
                    _tx, _ty, _tz = tdict["_xyzs"][tmpl_id]
                    _roll_deg = float(np.degrees(tdict["_rolls"][tmpl_id]))
                    _rtag = f"{_tag}_rank{rank}_{tmpl_id:04d}_{sim:.4f}_{_tx:.3f}_{_ty:.3f}_{_tz:.3f}_roll{_roll_deg:+06.1f}"
                    _tmpl_np = (tmpl.permute(1, 2, 0).cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)
                    _tH, _tW = _tmpl_np.shape[:2]

                    # 5. template_img
                    cv2.imwrite(str(_dbg_tmpl_img / f"{_rtag}.png"),
                                cv2.cvtColor(_tmpl_np, cv2.COLOR_RGB2BGR))

                    # 6. template_pixel_mask
                    _t_pmask = tdict["_masks_pixel"][tmpl_id].astype(np.uint8) * 255
                    cv2.imwrite(str(_dbg_tmpl_pmask / f"{_rtag}.png"), _t_pmask)

                    # 7. template_img_pixel_mask_overlay
                    cv2.imwrite(str(_dbg_tmpl_overlay / f"{_rtag}.png"),
                                _mask_overlay(_tmpl_np, tdict["_masks_pixel"][tmpl_id]))

                    # 8. template_feature_mask
                    cv2.imwrite(str(_dbg_tmpl_fmask / f"{_rtag}.png"),
                                _feature_mask_img(tdict["patch_masks"][tmpl_id], _tH, _tW))

            except Exception as _exc:
                logger.warning(f"Debug outputs 1-8 failed at frame {frame_idx} obj {obj_idx}: {_exc}")

            R    = out["TCO"][0][:3, :3].flatten().tolist()
            t    = out["TCO"][0][:3, 3].tolist()
            bbox = out["bbox"].cpu().numpy()
            bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

            R_top5_str = "|".join(
                " ".join(str(x) for x in tco[:3, :3].flatten()) for tco in out["TCO"]
            )
            t_top5_str = "|".join(
                " ".join(str(x) for x in tco[:3, 3]) for tco in out["TCO"]
            )
            score_top5_str = " ".join(str(s) for s in out["scores"])

            results_dict["scene_id"].append(0)
            results_dict["im_id"].append(int(frame_idx))
            results_dict["obj_id"].append(splat_paths[obj_idx])
            results_dict["score"].append(out["scores"][0])
            results_dict["R"].append(" ".join(str(x) for x in R))
            results_dict["t"].append(" ".join(str(x) for x in t))
            results_dict["bbox_visib"].append(" ".join(str(x) for x in bbox))
            results_dict["scale"].append(scale)
            results_dict["time"].append(-1)
            results_dict["R_top5"].append(R_top5_str)
            results_dict["t_top5"].append(t_top5_str)
            results_dict["score_top5"].append(score_top5_str)

        # ── Debug: save centroid projection for all frames ──────────────────────
        for obj_idx in range(n_objects):
            TCO = prev_poses[obj_idx]
            if TCO is None:
                continue
            R_np = TCO[:3, :3]
            t_np = TCO[:3, 3]
            box_np = boxes_t[obj_idx].cpu().numpy().astype(float)
            try:
                _save_centroid_projection(img, gaussian_splats[obj_idx], K, R_np, t_np, scales[obj_idx], box_np,
                                          debug_gaussian_dir / f"{frame_idx:06d}_obj{obj_idx}.png")
            except Exception as exc:
                logger.warning(f"Debug projection failed at frame {frame_idx}: {exc}")

            try:
                _save_bbox_3d(img, gaussian_splats[obj_idx], K, R_np, t_np, scales[obj_idx], box_np,
                              debug_bbox3d_dir / f"{frame_idx:06d}_obj{obj_idx}.png")
            except Exception as exc:
                logger.warning(f"Debug bbox3d failed at frame {frame_idx}: {exc}")

            try:
                _save_bbox3d_axis(img, gaussian_splats[obj_idx], K, R_np, t_np, scales[obj_idx], box_np,
                                  debug_bbox3d_axis_dir / f"{frame_idx:06d}_obj{obj_idx}.png")
            except Exception as exc:
                logger.warning(f"Debug bbox3d_axis failed at frame {frame_idx}: {exc}")

            try:
                _save_axis(img, gaussian_splats[obj_idx], K, R_np, t_np, scales[obj_idx], box_np,
                           debug_axis_dir / f"{frame_idx:06d}_obj{obj_idx}.png")
            except Exception as exc:
                logger.warning(f"Debug axis failed at frame {frame_idx}: {exc}")

            try:
                _center_cam = t_np                       # actual object translation (DA3-based when enabled)
                _cx    = (box_np[0] + box_np[2]) / 2.0
                _cy    = (box_np[1] + box_np[3]) / 2.0
                _px  = int(np.clip(round(_cx), 0, img.shape[1] - 1))
                _py  = int(np.clip(round(_cy), 0, img.shape[0] - 1))
                _rgb = img[_py, _px]
                logger.info(
                    f"[bbox3d] frame={frame_idx:06d} obj={obj_idx} "
                    f"center_cam=({_center_cam[0]:.4f}, {_center_cam[1]:.4f}, {_center_cam[2]:.4f}) "
                    f"rgb=({int(_rgb[0])}, {int(_rgb[1])}, {int(_rgb[2])})"
                )
            except Exception as exc:
                logger.warning(f"bbox3d center log failed at frame {frame_idx}: {exc}")

    df = pd.DataFrame(results_dict)
    df.to_csv(pose_outputs, index=False, header=True)
    logger.info(f"Saved poses → {pose_outputs}")

    _dir_to_video(debug_bbox3d_dir,      debug_dir / "bbox3d.mp4")
    _dir_to_video(debug_axis_dir,        debug_dir / "axis.mp4")
    _dir_to_video(debug_bbox3d_axis_dir, debug_dir / "bbox3d_axis.mp4")

    # ── Optional inline viz (--viz flag) ──────────────────────────────────────
    if args.viz:
        viz_dir = results_dir / "viz_pose"
        viz_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Saving pose visualizations → {viz_dir}")

        cmap = matplotlib.colormaps["Spectral"]
        for frame_idx, frame_name in enumerate(tqdm(frame_names, ncols=100)):
            img = cv2.cvtColor(cv2.imread(str(frame_name)), cv2.COLOR_BGR2RGB).astype(np.uint8)
            fig = plt.figure(frameon=False, figsize=(w // 100, h // 100))
            ax  = fig.add_axes([0, 0, 1, 1])
            ax.set_axis_off()
            ax.imshow(img)

            rows = df[df.im_id == frame_idx]
            for _, row in rows.iterrows():
                R_np   = np.fromstring(row.R, sep=" ").reshape(3, 3)
                t_np   = np.fromstring(row.t, sep=" ")
                obj_idx = splat_paths.index(row.obj_id)
                gs      = gaussian_splats[obj_idx]
                scale   = scales[obj_idx]

                xyz = gs.get_xyz.detach().cpu().numpy() * scale
                xyz_cam = (R_np @ xyz.T + t_np[:, None]).T
                valid   = xyz_cam[:, 2] > 0
                proj    = K @ xyz_cam[valid].T
                uv      = (proj[:2] / proj[2]).T
                colors  = cmap(np.linspace(0, 1, len(uv)))
                ax.scatter(uv[:, 0], uv[:, 1], s=1, alpha=0.4, color=colors)

            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
            plt.savefig(str(viz_dir / f"{frame_idx:06d}.png"),
                        bbox_inches="tight", pad_inches=0)
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",       type=str, required=True)
    parser.add_argument("--proposals",   type=str, required=True)
    parser.add_argument("--mesh",        type=str, required=True,
                        help="Path to the Gaussian splat / mesh (result.ply) for the tracked object, "
                             "relative to the freepose root.")
    parser.add_argument("--layer",       type=int,   default=22)
    parser.add_argument("--depth_method",type=str,   default="da3")
    parser.add_argument("--da3_depth", type=str, default=None,
                        help="Path to DA3 da3_output.npz. If given, object translation z is set "
                             "from DA3 metric depth at the object mask instead of the bbox-size heuristic.")
    parser.add_argument("--assume_metric", action="store_true",
                        help="Treat DA3 depth as already metric (skip the per-object alpha anchor).")
    parser.add_argument("--bbox_extend", type=float, default=0.05)
    parser.add_argument("--batch_size",    type=int,   default=128)
    parser.add_argument("--cache_size",    type=int,   default=21)
    parser.add_argument("--num_templates", type=int,   default=3600,
                        help="Number of Hopf-Hammersley SO(3) template views to render (default: 3600)")
    parser.add_argument("--viz",           action="store_true")
    parser.add_argument("--save_all_cache", action="store_true")
    parser.add_argument("--mask_query",    action="store_true", default=True,
                        help="Mask the query crop to the object foreground before retrieval (default: True). "
                             "Pass --no-mask_query to use the unmasked RGB crop.")
    parser.add_argument("--no-mask_query", dest="mask_query", action="store_false")
    parser.add_argument("--mask_template", action="store_true", default=False,
                        help="Zero out background pixels in rendered template images before feature extraction (default: False).")
    parser.add_argument("--no-mask_template", dest="mask_template", action="store_false")
    parser.add_argument("--query_fg_patches", action="store_true", default=True,
                        help="Average similarity only over query foreground patches (default: True). "
                             "Pass --no-query_fg_patches to use all query patches.")
    parser.add_argument("--no-query_fg_patches", dest="query_fg_patches", action="store_false")
    parser.add_argument("--template_fg_patches", action="store_true", default=False,
                        help="Average similarity only over each template's foreground patches (default: False). "
                             "Pass --template_fg_patches to enable.")
    parser.add_argument("--no-template_fg_patches", dest="template_fg_patches", action="store_false")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Directory holding the proposals JSON and receiving pose/debug output "
                             "(default: data/results/mvsam3d/<video>)")
    parser.add_argument("--top_n_candidates", type=int, default=5,
                        help="Number of top pose candidates to retrieve per frame (default: 5).")
    args = parser.parse_args()
    main(args)
