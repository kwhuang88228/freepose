#!/usr/bin/env python3
"""
dense_track_masked.py — Mask-restricted dense 3D tracking for the freepose pipeline.

Adapted from DenseTrack3Dv2/demo.py. Runs DenseTrack3DV2 over a full video, then
keeps only the dense trajectories whose query-frame pixel falls inside a per-frame
binary mask (produced by stage 1 of inference_v4.py). Depth for every frame is
computed with UniDepth (cached to <output_dir>/depth_pred.npy).

This script lives in freepose/scripts/ but depends on the DenseTrack3Dv2 package
and checkpoints, so it resolves DENSETRACK_ROOT (= freepose/DenseTrack3Dv2) itself
and adds it to sys.path. It is location/cwd-independent and must be run with the
`densetrack3d` conda env python.

Inputs
    --frames_dir   Directory of full-video RGB frames ({idx:06d}.png).
    --mask_dir     Directory of per-frame binary masks ({idx:06d}_mask.png).
    --output_dir   Where outputs + debug visualizations are written.

Outputs (in --output_dir)
    dense_3d_track_masked.pkl   Filtered 3D trajectories (points inside the mask).
    dense_3d_track_full.pkl     Unfiltered dense 3D trajectories (for comparison).
    depth_pred.npy              Cached UniDepth depth for the full video.
    masked_2d_track.mp4         Filtered tracks drawn over the video.
    full_2d_track.mp4           Unfiltered dense grid (downsampled) over the video.
    query_mask_overlay.png      Query frame + mask + kept query points (filter check).
    depth_viz.mp4               Colorized UniDepth depth (depth sanity check).
"""

import argparse
import glob
import os
import pickle
import sys
from pathlib import Path

import cv2
import matplotlib.cm as cm
import mediapy as media
import numpy as np
import torch
from einops import rearrange

# Resolve the DenseTrack3Dv2 package root (freepose/DenseTrack3Dv2) regardless of
# cwd, and put it on sys.path so `import densetrack3d` works from anywhere.
FREEPOSE_ROOT = Path(__file__).resolve().parent.parent
DENSETRACK_ROOT = FREEPOSE_ROOT / "DenseTrack3Dv2"
sys.path.insert(0, str(DENSETRACK_ROOT))

from densetrack3d.models.densetrack3d.densetrack3dv2 import DenseTrack3DV2
from densetrack3d.models.predictor.dense_predictor import DensePredictor3D
from densetrack3d.utils.visualizer import Visualizer


device = torch.device("cuda")


@torch.inference_mode()
def predict_unidepth(video, model, chunk_size=4):
    """Per-frame metric depth via UniDepth. video: (T,H,W,3) uint8 -> (T,H,W) float.

    Keeps the video on CPU and moves one chunk at a time to the GPU to bound peak
    memory (UniDepth's internal upsample is the memory hot spot on large frames).
    """
    video_torch = torch.from_numpy(video).permute(0, 3, 1, 2)

    depth_pred = []
    for chunk in torch.split(video_torch, chunk_size, dim=0):
        predictions = model.infer(chunk.to(device))
        depth_pred.append(predictions["depth"].squeeze(1).cpu().numpy())
    return np.concatenate(depth_pred, axis=0)


def load_frames(frames_dir):
    paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    if len(paths) == 0:
        raise FileNotFoundError(f"No .png frames found in {frames_dir}")
    frames = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in paths]
    return np.stack(frames), paths


def load_masks(mask_dir, frame_paths):
    """Align masks to frames by index; frames without a mask file get a zero mask."""
    H, W = cv2.imread(frame_paths[0]).shape[:2]
    masks = []
    for p in frame_paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        mpath = os.path.join(mask_dir, f"{stem}_mask.png")
        if os.path.exists(mpath):
            m = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
            masks.append((m > 127).astype(np.uint8))
        else:
            masks.append(np.zeros((H, W), dtype=np.uint8))
    return np.stack(masks)


def filter_by_mask(trajs_3d_dict, n_full, keep):
    """Index every tensor in the 3D dict along its N axis (== n_full)."""
    out = {}
    for k, v in trajs_3d_dict.items():
        axes = [i for i, s in enumerate(v.shape) if s == n_full]
        if not axes:
            out[k] = v
            continue
        idx = [slice(None)] * v.ndim
        idx[axes[0]] = keep
        out[k] = v[tuple(idx)]
    return out


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir", type=str, required=True, help="full-video RGB frames")
    parser.add_argument("--mask_dir", type=str, required=True, help="per-frame binary masks")
    parser.add_argument("--output_dir", type=str, required=True, help="output / debug dir")
    parser.add_argument("--ckpt", type=str,
                        default=str(DENSETRACK_ROOT / "checkpoints" / "densetrack3dv2.pth"))
    parser.add_argument("--query_frame", type=int, default=0, help="track from / filter against this frame")
    parser.add_argument("--downsample", type=int, default=8, help="grid downsample for 2D viz")
    parser.add_argument("--upsample_factor", type=int, default=4,
                        help="dense grid stride; larger (e.g. 8) tracks fewer points to cut GPU memory")
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--depth_chunk_size", type=int, default=4,
                        help="frames per UniDepth forward pass; smaller cuts GPU memory")
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Create DenseTrack3D model")
    model = DenseTrack3DV2(
        stride=4,
        window_len=16,
        add_space_attn=True,
        num_virtual_tracks=64,
        model_resolution=(384, 512),
        upsample_factor=args.upsample_factor,
        coarse_to_fine_dense=True,
    )

    print(f"Load checkpoint from {args.ckpt}")
    with open(args.ckpt, "rb") as f:
        state_dict = torch.load(f, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
    model.load_state_dict(state_dict, strict=True)

    predictor = DensePredictor3D(model=model, n_iters=6).eval().cuda()

    # ── Load video + masks ────────────────────────────────────────────────────
    video_np, frame_paths = load_frames(args.frames_dir)
    masks_np = load_masks(args.mask_dir, frame_paths)
    T, H, W, _ = video_np.shape
    print(f"Loaded {T} frames ({H}x{W}); masks foreground px (query frame): "
          f"{int(masks_np[args.query_frame].sum())}")

    # ── Depth: UniDepth on the full video (cached) ────────────────────────────
    depth_cache = os.path.join(args.output_dir, "depth_pred.npy")
    if os.path.exists(depth_cache):
        print(f"Load cached depth from {depth_cache}")
        videodepth_np = np.load(depth_cache)
    else:
        sys.path.append(str(DENSETRACK_ROOT / "submodules" / "UniDepth"))
        from unidepth.models import UniDepthV2

        print("Run UniDepth on full video")
        unidepth_model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14").eval().to(device)
        videodepth_np = predict_unidepth(video_np, unidepth_model, chunk_size=args.depth_chunk_size)
        np.save(depth_cache, videodepth_np)

    video = torch.from_numpy(video_np).permute(0, 3, 1, 2).cuda()[None].float()
    videodepth = torch.from_numpy(videodepth_np).unsqueeze(1).cuda()[None].float()

    # ── Dense tracking ────────────────────────────────────────────────────────
    print("Run DenseTrack3D")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.use_fp16):
        out_dict = predictor(
            video,
            videodepth,
            grid_query_frame=args.query_frame,
            use_efficient_global_attn=False,
        )

    trajs_uv = out_dict["trajs_uv"]        # (B,T,N,2) original-image coords
    trajs_vis = out_dict["vis"]            # (B,T,N)
    dense_reso = out_dict["dense_reso"]    # (h,w) of the dense grid
    n_full = trajs_uv.shape[2]

    # ── Mask filtering: keep grid points whose query-frame pixel is in the mask ─
    query_xy = trajs_uv[0, 0].round().long()          # (N,2) at query frame
    qx = query_xy[:, 0].clamp(0, W - 1)
    qy = query_xy[:, 1].clamp(0, H - 1)
    mask_q = torch.from_numpy(masks_np[args.query_frame]).to(query_xy.device)
    keep = mask_q[qy, qx] > 0                          # (N,) bool
    print(f"Kept {int(keep.sum())} / {n_full} dense points inside the mask")

    full_3d = {k: v[0].cpu().numpy() for k, v in out_dict["trajs_3d_dict"].items()}
    masked_3d_dict = filter_by_mask(out_dict["trajs_3d_dict"], n_full, keep)
    masked_3d = {k: v[0].cpu().numpy() for k, v in masked_3d_dict.items()}

    with open(os.path.join(args.output_dir, "dense_3d_track_full.pkl"), "wb") as f:
        pickle.dump(full_3d, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.output_dir, "dense_3d_track_masked.pkl"), "wb") as f:
        pickle.dump(masked_3d, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved 3D tracks → {args.output_dir}")

    # ── Debug visualizations ──────────────────────────────────────────────────
    lw = max(1, int(1 * W / 512))
    viz = Visualizer(save_dir=args.output_dir, fps=10, show_first_frame=0,
                     linewidth=lw, tracks_leave_trace=0, pad_value=0)

    # (a) full dense grid, downsampled
    full_uv = rearrange(trajs_uv, "b t (h w) c -> b t h w c", h=dense_reso[0], w=dense_reso[1])
    full_uv = rearrange(full_uv[:, :, ::args.downsample, ::args.downsample], "b t h w c -> b t (h w) c")
    full_v = rearrange(trajs_vis, "b t (h w) -> b t h w", h=dense_reso[0], w=dense_reso[1])
    full_v = rearrange(full_v[:, :, ::args.downsample, ::args.downsample], "b t h w -> b t (h w)")
    full_viz = viz.visualize(video, full_uv, full_v[..., None], filename="full", save_video=False)
    media.write_video(os.path.join(args.output_dir, "full_2d_track.mp4"),
                      full_viz[0].permute(0, 2, 3, 1).cpu().numpy(), fps=10)

    # (b) masked tracks: downsample the grid, then keep in-mask sub-grid points
    grid_uv = rearrange(trajs_uv, "b t (h w) c -> b t h w c", h=dense_reso[0], w=dense_reso[1])
    grid_v = rearrange(trajs_vis, "b t (h w) -> b t h w", h=dense_reso[0], w=dense_reso[1])
    sub_uv = rearrange(grid_uv[:, :, ::args.downsample, ::args.downsample], "b t h w c -> b t (h w) c")
    sub_v = rearrange(grid_v[:, :, ::args.downsample, ::args.downsample], "b t h w -> b t (h w)")
    sxy = sub_uv[0, 0].round().long()
    sub_keep = mask_q[sxy[:, 1].clamp(0, H - 1), sxy[:, 0].clamp(0, W - 1)] > 0
    if int(sub_keep.sum()) > 0:
        masked_viz = viz.visualize(video, sub_uv[:, :, sub_keep], sub_v[:, :, sub_keep, None],
                                   filename="masked", save_video=False)
        media.write_video(os.path.join(args.output_dir, "masked_2d_track.mp4"),
                          masked_viz[0].permute(0, 2, 3, 1).cpu().numpy(), fps=10)
    else:
        print("Warning: no sub-grid points inside the mask for viz (try smaller --downsample)")

    # (c) query frame + mask + kept query points
    overlay = video_np[args.query_frame].copy()
    colored = np.zeros_like(overlay)
    colored[masks_np[args.query_frame].astype(bool)] = (0, 255, 0)
    overlay = cv2.addWeighted(overlay, 1.0, colored, 0.4, 0)
    kept_xy = query_xy[keep].cpu().numpy()
    for x, y in kept_xy[:: max(1, len(kept_xy) // 4000)]:
        cv2.circle(overlay, (int(np.clip(x, 0, W - 1)), int(np.clip(y, 0, H - 1))), 1, (255, 0, 0), -1)
    cv2.imwrite(os.path.join(args.output_dir, "query_mask_overlay.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # (d) colorized depth video
    d = videodepth_np.astype(np.float32)
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    depth_rgb = (cm.get_cmap("turbo")(d)[..., :3] * 255).astype(np.uint8)
    media.write_video(os.path.join(args.output_dir, "depth_viz.mp4"), depth_rgb, fps=10)

    print(f"Wrote debug visualizations → {args.output_dir}")
