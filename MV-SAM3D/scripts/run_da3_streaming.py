"""
Depth Anything 3 — Streaming Runner (memory-efficient, for long videos)

Drop-in replacement for run_da3.py's DA3 stage when a clip is too long to fit
through DA3's global attention (traditional run_da3 OOMs). Instead of a single
joint pass (or CPU-heavy windowing), this drives DA3-Streaming (VGGT-Long-style
chunk streaming): frames are processed one overlapping chunk at a time, each
chunk's prediction is offloaded to disk, and consecutive chunks are stitched into
one world frame/scale by a Sim(3) alignment over their shared overlap. GPU memory
stays bounded by a single chunk regardless of clip length.

This wrapper runs DA3-Streaming (loop closure disabled) and then repackages its
globally-aligned per-frame outputs into the SAME da3_output.npz contract that
run_da3.py emits, so every downstream stage (MV-SAM3D, compute_scale_video,
dino_inference_video_mvsam3d) consumes it identically:

    depth            (N, H, W)      per-frame metric depth, globally scale-consistent
    pointmaps        (N, H, W, 3)   camera-space pointmap (for parity; unused downstream)
    pointmaps_sam3d  (N, 3, H, W)   channel-first camera-space pointmap for MV-SAM3D
    extrinsics       (N, 4, 4)      world-to-camera, globally aligned
    intrinsics       (N, 3, 3)      per-frame K (at DA3 processing resolution)
    image_files      (N,)           input image paths (order matches all arrays)
    window_id        (N,)           streaming chunk each frame's depth came from
    process_res      scalar

Usage:
    python scripts/run_da3_streaming.py --image_dir ./frames --output_dir ./out \
        --output_npz ./out/da3_output.npz --chunk_size 120 --overlap 60
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# ============================================================================
# Path setup: DA3 (and its da3_streaming subproject) is a sibling of MV-SAM3D.
# ============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # MV-SAM3D root
DA3_ROOT = PROJECT_ROOT / "Depth-Anything-3"
DA3_STREAMING = DA3_ROOT / "da3_streaming"

if not DA3_STREAMING.exists():
    raise FileNotFoundError(
        f"da3_streaming not found at {DA3_STREAMING}. Expected DA3 laid out as a "
        f"sibling of MV-SAM3D with its da3_streaming subproject."
    )

# DA3 src (depth_anything_3.*) and the da3_streaming dir (loop_utils.*, fastloop.*)
sys.path.insert(0, str(DA3_ROOT / "src"))
sys.path.insert(0, str(DA3_STREAMING))

# Reuse run_da3's exact camera-space pointmap helpers so the two runners produce
# byte-compatible pointmaps.
from run_da3 import depth_to_pointmap, pointmap_to_sam3d_format  # noqa: E402
from loop_utils.config_utils import load_config  # noqa: E402
from da3_streaming import DA3_Streaming  # noqa: E402


def _find_checkpoint_dir(model_path):
    """Directory holding both config.json and model.safetensors for DA3-Streaming's
    config-based loader. Mirrors run_da3.py's auto-detect (HF cache / local
    checkpoints) but resolves to a directory, not an HF model id."""
    if model_path is not None:
        p = Path(model_path)
        if (p / "config.json").exists() and (p / "model.safetensors").exists():
            return p
        raise FileNotFoundError(
            f"--model_path {p} must contain config.json and model.safetensors."
        )

    candidates = [
        DA3_ROOT / "checkpoints" / "DA3NESTED-GIANT-LARGE",
        DA3_ROOT / "checkpoints" / "DA3-GIANT-LARGE",
    ]
    hf_hub = Path.home() / ".cache" / "huggingface" / "hub" / "models--depth-anything--DA3NESTED-GIANT-LARGE"
    if (hf_hub / "snapshots").exists():
        candidates += sorted((hf_hub / "snapshots").iterdir(), reverse=True)

    for c in candidates:
        if c is not None and (c / "config.json").exists() and (c / "model.safetensors").exists():
            return c
    raise FileNotFoundError(
        "No DA3 checkpoint dir with config.json + model.safetensors found. "
        "Download it (e.g. da3_streaming/scripts/download_weights.sh) or pass --model_path."
    )


def _build_config(chunk_size, overlap, align_lib, ckpt_dir):
    """Streaming config: base_config.yaml with loop closure off, per-frame depth
    dumping on, and weights pointed at the resolved checkpoint dir."""
    config = load_config(str(DA3_STREAMING / "configs" / "base_config.yaml"))
    config["Weights"]["DA3"] = str(ckpt_dir / "model.safetensors")
    config["Weights"]["DA3_CONFIG"] = str(ckpt_dir / "config.json")
    config["Model"]["loop_enable"] = False           # no revisit detection (no SALAD/faiss)
    config["Model"]["save_depth_conf_result"] = True  # need per-frame aligned depth/intrinsics
    config["Model"]["save_debug_info"] = False
    config["Model"]["delete_temp_files"] = True       # frees _tmp_* but keeps results_output/
    config["Model"]["chunk_size"] = chunk_size
    config["Model"]["overlap"] = overlap
    config["Model"]["align_lib"] = align_lib
    return config


def _window_ids(chunk_indices, overlap, n):
    """Map each global frame index to the streaming chunk that owns its depth.

    Mirrors DA3_Streaming.save_depth_conf_result's tiling (overlap_s=0,
    overlap_e=overlap): chunk ci owns [start, end-overlap) except the last, which
    owns [start, end). Ranges are disjoint and tile [0, n)."""
    window_id = np.full(n, -1, dtype=np.int64)
    last = len(chunk_indices) - 1
    for ci, (s, e) in enumerate(chunk_indices):
        stop = e if ci == last else e - overlap
        window_id[s:stop] = ci
    return window_id


def run_da3_streaming_inference(
    image_dir,
    output_dir,
    output_npz=None,
    chunk_size=120,
    overlap=60,
    align_lib="torch",
    model_path=None,
    process_res=504,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stream_dir = output_dir / "streaming"   # DA3-Streaming's own working/output dir
    stream_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = _find_checkpoint_dir(model_path)
    print(f"[da3_streaming] checkpoint dir: {ckpt_dir}")
    config = _build_config(chunk_size, overlap, align_lib, ckpt_dir)

    # ── Run streaming reconstruction ──────────────────────────────────────────
    streamer = DA3_Streaming(str(image_dir), str(stream_dir), config)
    streamer.run()
    streamer.close()   # deletes _tmp_* scratch; results_output/ + camera_poses.txt kept

    img_list = list(streamer.img_list)          # sorted frame paths, index == global idx
    chunk_indices = list(streamer.chunk_indices)
    n = len(img_list)
    if len(chunk_indices) < 2:
        raise RuntimeError(
            f"[da3_streaming] only {len(chunk_indices)} chunk(s) for {n} frames "
            f"(chunk_size={chunk_size}); per-frame depth is only dumped on the "
            f"multi-chunk path. Use run_da3.py for clips this short."
        )

    # ── Globally-aligned extrinsics from camera_poses.txt (c2w) → w2c ─────────
    poses_txt = stream_dir / "camera_poses.txt"
    c2w = np.loadtxt(poses_txt).reshape(n, 4, 4)
    extrinsics = np.linalg.inv(c2w).astype(np.float64)   # (N, 4, 4) world-to-camera

    # ── Per-frame aligned depth + intrinsics from results_output/ ─────────────
    results_output = stream_dir / "results_output"
    depth, intrinsics = [], []
    for g in range(n):
        f = results_output / f"frame_{g}.npz"
        if not f.exists():
            raise FileNotFoundError(
                f"[da3_streaming] missing per-frame result {f}; streaming did not "
                f"dump depth for frame {g}."
            )
        d = np.load(f)
        depth.append(np.asarray(d["depth"], dtype=np.float32))
        intrinsics.append(np.asarray(d["intrinsics"], dtype=np.float64))
    depth = np.stack(depth, axis=0)              # (N, H, W)
    intrinsics = np.stack(intrinsics, axis=0)    # (N, 3, 3)

    # ── Camera-space pointmaps (depth + intrinsics only, no extrinsics) ───────
    pointmaps = np.stack(
        [depth_to_pointmap(depth[i], intrinsics[i]) for i in range(n)], axis=0
    )                                            # (N, H, W, 3)
    pointmaps_sam3d = np.stack(
        [pointmap_to_sam3d_format(pointmaps[i]) for i in range(n)], axis=0
    )                                            # (N, 3, H, W)

    window_id = _window_ids(chunk_indices, overlap, n)

    out = Path(output_npz) if output_npz is not None else output_dir / "da3_output.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        depth=depth,
        pointmaps=pointmaps,
        pointmaps_sam3d=pointmaps_sam3d,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        image_files=np.array([str(f) for f in img_list]),
        process_res=process_res,
        window_id=window_id,
    )
    print(f"\n[da3_streaming] Results saved to: {out}")
    print(f"  depth: {depth.shape}  range [{depth.min():.4f}, {depth.max():.4f}]")
    print(f"  pointmaps_sam3d: {pointmaps_sam3d.shape}")
    print(f"  extrinsics: {extrinsics.shape}  intrinsics: {intrinsics.shape}")
    print(f"  chunks: {len(chunk_indices)}  window_id range [{window_id.min()}, {window_id.max()}]")
    return str(out)


def main():
    parser = argparse.ArgumentParser(
        description="Run DA3-Streaming on a folder of images and emit da3_output.npz "
                    "(same format as run_da3.py) for long clips that OOM the joint pass."
    )
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Folder of input frames.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory (streaming scratch under <output_dir>/streaming).")
    parser.add_argument("--output_npz", type=str, default=None,
                        help="Path for the da3_output.npz (default: <output_dir>/da3_output.npz).")
    parser.add_argument("--chunk_size", type=int, default=120,
                        help="Frames per streaming chunk (default: 120).")
    parser.add_argument("--overlap", type=int, default=60,
                        help="Frames shared between consecutive chunks for Sim(3) "
                             "alignment (default: 60). Must be < chunk_size.")
    parser.add_argument("--align_lib", type=str, default="torch",
                        choices=["triton", "torch", "numba", "numpy"],
                        help="Chunk-alignment backend (default: torch, GPU, robust).")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Dir with config.json + model.safetensors (default: auto-detect).")
    parser.add_argument("--process_res", type=int, default=504,
                        help="Recorded in the npz for parity (default: 504).")
    args = parser.parse_args()

    run_da3_streaming_inference(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        output_npz=args.output_npz,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        align_lib=args.align_lib,
        model_path=args.model_path,
        process_res=args.process_res,
    )


if __name__ == "__main__":
    main()
