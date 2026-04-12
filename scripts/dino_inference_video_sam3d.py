"""
dino_inference_video_sam3d.py

Stage 4 of the SAM-3D pipeline:
  1. Load each object's Gaussian splat (path stored in proposals JSON 'mesh' field).
  2. Pre-render 600 Hammersley views of the splat → coarse template bank.
  3. Run DinoOnlinePoseEstimatorSam3d per frame: coarse DINOv2 matching +
     fine-pose neighbourhood refinement.
  4. Write per-frame 6D poses to CSV.

Debug outputs → data/results/sam3d/<video>/04_coarse_poses/
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

def build_template_dict(gs, model_name: str, n_poses: int = 600, bbox_extend: float = 0.05):
    """Render 600 Hammersley views of *gs* and package them as a template_dict.

    The format mirrors what WebTemplateDataset returns so DinoPoseEstimatorSam3d
    can use the same feature-extraction + caching machinery.

    Returns:
        dict with keys:
          'model_name'  str
          'templates'   Tensor [N, 3, H, W]  float, normalised 0-1
          'depths'      list of Tensor [H, W] float (metric depth)
          'intrinsic'   Tensor [3, 3] float  (K_SAM3D pixel-space)
    """
    logger.info(f"Rendering {n_poses} template views for {model_name}")
    renderer = SplatRenderer(n_poses=n_poses)
    renders  = renderer.render(gs)                                    # list of (rgb, depth, tcoinit)
    templates_cropped, tcoinits, masks_cropped = renderer.generate_proposals(renders, bbox_extend=bbox_extend)

    depths = [torch.from_numpy(renders[i][1]) for i in range(len(renders))]

    # Pre-compute patch-level foreground masks from the CROPPED template masks.
    # masks_cropped matches the spatial layout of templates_cropped (what DINOv2 sees).
    # Shape: [N, num_patches] bool — used for masked mean pooling in the estimator.
    patch_masks = DinoPoseEstimator._to_patch_mask(np.stack(masks_cropped))   # [N, num_patches]

    # Store SAM-3D TCO_init matrices in the renderer for get_z_from_pointcloud
    return {
        "model_name":   model_name,
        "templates":    templates_cropped,                             # [N,3,H,W]
        "depths":       depths,                                        # list of [H,W] Tensor
        "intrinsic":    torch.from_numpy(K_SAM3D).float(),
        "patch_masks":  patch_masks,                                   # [N, num_patches] bool
        "_tcoinits":    tcoinits,                                      # list of (4,4) np arrays
        "_renderer":    renderer,                                      # kept for fine-pose use
    }


# ── Debug visualisation ────────────────────────────────────────────────────────

def _save_bbox_3d(img, gs, K, R, t, scale, box, out_path: Path):
    """Draw the 3D bounding box of the Gaussian splat on the image.

    Edges parallel to X are drawn in Red, Y in Green, Z in Blue.
    Uses the same t re-derivation as _save_centroid_projection.
    """
    gs_xyz_world = gs.get_xyz.detach().cpu().numpy()
    xyz_min = gs_xyz_world.min(0) * scale
    xyz_max = gs_xyz_world.max(0) * scale

    # Re-derive t (same logic as _save_centroid_projection)
    gs_diameter_m = (gs_xyz_world.max(0) - gs_xyz_world.min(0)).max() * scale
    bbox_px = max(box[2] - box[0], box[3] - box[1]) + 1.0
    z_correct = K[0, 0] * gs_diameter_m / bbox_px
    bb_center = np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])
    t = np.array([
        (bb_center[0] - K[0, 2]) * z_correct / K[0, 0],
        (bb_center[1] - K[1, 2]) * z_correct / K[1, 1],
        z_correct,
    ])

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


def _save_centroid_projection(img, gs, K, R, t, scale, box, out_path: Path, n_pts: int = 2000):
    """Scatter Gaussian centroids projected onto the image frame.

    The gsplat renderer does not output depth, so get_z_from_pointcloud falls back
    to the raw SAM-3D extrinsic z (~2.0 in SAM-3D world units) rather than computing
    a real-world metric z from the video bbox.  We override t here using the estimated
    scale and the detected bbox so the projected cloud appears at the correct depth.
    """
    xyz = gs.get_xyz.detach().cpu().numpy() * scale

    # Re-derive t from scale + bbox (the TCO t[2] is unreliable; see docstring).
    gs_xyz_world = gs.get_xyz.detach().cpu().numpy()
    gs_diameter_m = (gs_xyz_world.max(0) - gs_xyz_world.min(0)).max() * scale
    bbox_px = max(box[2] - box[0], box[3] - box[1]) + 1.0
    z_correct = K[0, 0] * gs_diameter_m / bbox_px
    bb_center = np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])
    t = np.array([
        (bb_center[0] - K[0, 2]) * z_correct / K[0, 0],
        (bb_center[1] - K[1, 2]) * z_correct / K[1, 1],
        z_correct,
    ])

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


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    video_dir   = (Path("data") / "datasets" / "videos" / args.video).resolve()
    frame_names = sorted([p for p in video_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg"]])

    results_dir    = (Path("data") / "results" / "sam3d" / args.video).resolve()
    proposals_path = results_dir / args.proposals

    pose_outputs = results_dir / args.proposals.replace(
        ".json",
        f"_dinopose_layer_{args.layer}_bbext_{args.bbox_extend}_depth_{args.depth_method}.csv",
    )

    # Debug directories
    debug_dir = _FREEPOSE_ROOT / "data" / "results" / "sam3d" / args.video / "04_coarse_poses"
    debug_gaussian_dir = debug_dir / "gaussian"
    debug_bbox3d_dir   = debug_dir / "bbox3d"
    debug_gaussian_dir.mkdir(parents=True, exist_ok=True)
    debug_bbox3d_dir.mkdir(parents=True, exist_ok=True)

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
    if args.depth_method == "zoedepth":
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
        splat_path = props[0][i]["mesh"]          # relative path, e.g. "data/gaussian_splats/..."
        assert all(props[f][i]["mesh"] == splat_path for f in range(n_frames))

        ply_abs = _FREEPOSE_ROOT / splat_path
        logger.info(f"  Object {i}: {ply_abs}")
        gs = load_gaussian(ply_abs, device=device)

        tdict = build_template_dict(
            gs,
            model_name=splat_path,
            n_poses=args.num_templates,
            bbox_extend=args.bbox_extend,
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
    # (one set per object; use object 0's renderer since all share the same Hammersley sequence)
    model.coarse_estimator.mesh_poses = template_dicts[0]["_tcoinits"]

    # ── Per-frame inference ────────────────────────────────────────────────────
    results_dict = {
        "scene_id": [], "im_id": [], "obj_id": [], "score": [],
        "R": [], "t": [], "bbox_visib": [], "scale": [], "time": [],
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
        proposals = Proposals(img, {"boxes": boxes_t, "masks": masks_t}, 420,
                              bbox_extend=args.bbox_extend)
        proposals.scores = scores_prop

        for obj_idx in range(n_objects):
            prop      = proposals.proposals[obj_idx]
            prop_mask = proposals.proposals_masks[obj_idx]
            box       = boxes_t[obj_idx]
            gs        = gaussian_splats[obj_idx]
            tdict     = template_dicts[obj_idx]
            scale     = scales[obj_idx]

            with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16):
                out = model(
                    prop, prop_mask, tdict, gs, K, box, scale,
                    prev_pose=prev_poses[obj_idx],
                    neighborhood=15,
                    layer=args.layer,
                    batch_size=args.batch_size,
                )
                prev_poses[obj_idx] = out["TCO"][0]

            R    = out["TCO"][0][:3, :3].flatten().tolist()
            t    = out["TCO"][0][:3, 3].tolist()
            bbox = out["bbox"].cpu().numpy()
            bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

            results_dict["scene_id"].append(0)
            results_dict["im_id"].append(int(frame_idx))
            results_dict["obj_id"].append(splat_paths[obj_idx])
            results_dict["score"].append(out["scores"][0])
            results_dict["R"].append(" ".join(str(x) for x in R))
            results_dict["t"].append(" ".join(str(x) for x in t))
            results_dict["bbox_visib"].append(" ".join(str(x) for x in bbox))
            results_dict["scale"].append(scale)
            results_dict["time"].append(-1)

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
                                          debug_gaussian_dir / f"{frame_idx:06d}_obj{obj_idx}.jpg")
            except Exception as exc:
                logger.warning(f"Debug projection failed at frame {frame_idx}: {exc}")

            try:
                _save_bbox_3d(img, gaussian_splats[obj_idx], K, R_np, t_np, scales[obj_idx], box_np,
                              debug_bbox3d_dir / f"{frame_idx:06d}_obj{obj_idx}.jpg")
            except Exception as exc:
                logger.warning(f"Debug bbox3d failed at frame {frame_idx}: {exc}")

    df = pd.DataFrame(results_dict)
    df.to_csv(pose_outputs, index=False, header=True)
    logger.info(f"Saved poses → {pose_outputs}")

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
            plt.savefig(str(viz_dir / f"{frame_idx:06d}.jpg"),
                        bbox_inches="tight", pad_inches=0)
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",       type=str, required=True)
    parser.add_argument("--proposals",   type=str, required=True)
    parser.add_argument("--layer",       type=int,   default=22)
    parser.add_argument("--depth_method",type=str,   default="zoedepth")
    parser.add_argument("--bbox_extend", type=float, default=0.05)
    parser.add_argument("--batch_size",    type=int,   default=128)
    parser.add_argument("--cache_size",    type=int,   default=21)
    parser.add_argument("--num_templates", type=int,   default=600,
                        help="Number of Hammersley template views to render (default: 600)")
    parser.add_argument("--viz",           action="store_true")
    parser.add_argument("--save_all_cache", action="store_true")
    args = parser.parse_args()
    main(args)
