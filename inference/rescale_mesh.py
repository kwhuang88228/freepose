#!/usr/bin/env python3
"""
rescale_mesh.py — Rescale a mesh to its metric (real-world) size in metres.

The normalization step (centre + unit half-extent) is performed by
scripts.resize_meshes.rescale_mesh; the metric scale is then applied on top.

Scale source (one required):
  --scale FLOAT          Target half-extent in metres (e.g. 0.05 for a 10 cm object).
  --proposals PATH       Path to a *_gpt4_scaled.json proposals file (list format)
                         produced by scripts/compute_scale_video.py; reads the
                         pre-computed median scale for --object_idx.
  --proposals PATH       Path to data/gpt4_scales.json (dict format: name → full
    --object_name NAME   extent in metres); looks up the closest matching entry by
                         case-insensitive substring match on NAME.

Usage:
    python inference/rescale_mesh.py --mesh result.ply --scale 0.05
    python inference/rescale_mesh.py --mesh result.ply \\
        --proposals data/results/mvsam3d/clip/mvsam3d_clip_cup_gpt4_scaled.json \\
        --object_idx 0 --output result_metric.ply
    python inference/rescale_mesh.py --mesh result.ply \\
        --proposals data/gpt4_scales.json --object_name knife \\
        --output result_metric.obj
"""

import argparse
import json
import sys
from pathlib import Path

FREEPOSE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FREEPOSE_ROOT))

from scripts.resize_meshes import rescale_mesh as normalize_mesh  # centres + unit half-extent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rescale a mesh to its metric size (in metres)."
    )
    parser.add_argument("--mesh", required=True,
                        help="Path to input mesh file (PLY, OBJ, GLB, …).")
    parser.add_argument("--output", default=None,
                        help="Output path. Default: <stem>_metric.<suffix> next to the input.")

    scale_src = parser.add_mutually_exclusive_group(required=True)
    scale_src.add_argument("--scale", type=float, default=None,
                           help="Metric half-extent in metres (half the longest bounding-box side).")
    scale_src.add_argument("--proposals", default=None,
                           help="Path to a *_gpt4_scaled.json proposals file (list) or "
                                "data/gpt4_scales.json (dict). "
                                "Use --object_idx with list format, --object_name with dict format.")

    parser.add_argument("--object_idx", type=int, default=0,
                        help="[--proposals list] Zero-based index of the tracked object (default: 0).")
    parser.add_argument("--object_name", default=None,
                        help="[--proposals dict] Object name for substring lookup in gpt4_scales.json "
                             "(e.g. 'knife'). Case-insensitive.")

    args = parser.parse_args()

    mesh_path = Path(args.mesh).resolve()
    if not mesh_path.exists():
        sys.exit(f"Mesh not found: {mesh_path}")

    # ── Resolve metric scale ──────────────────────────────────────────────────
    if args.scale is not None:
        metric_scale = args.scale
    else:
        proposals_path = Path(args.proposals).resolve()
        if not proposals_path.exists():
            sys.exit(f"Proposals file not found: {proposals_path}")
        with open(proposals_path) as f:
            proposals = json.load(f)

        if isinstance(proposals, dict):
            # gpt4_scales.json format: {"a photo of knife": 0.3, ...}
            # Values are full extents; pipeline uses half-extent, so divide by 2.
            if args.object_name is None:
                sys.exit("--object_name is required when --proposals points to a dict-format scales file.")
            query = args.object_name.lower()
            matches = {k: v for k, v in proposals.items() if query in k.lower()}
            if not matches:
                sys.exit(f"No entry matching '{query}' in {proposals_path.name}. "
                         f"Sample keys: {list(proposals)[:5]}")
            key, full_extent = min(matches.items(), key=lambda kv: len(kv[0]))
            metric_scale = full_extent / 2.0
            print(f"Matched '{key}' → full extent {full_extent} m → half-extent {metric_scale:.4f} m")
        else:
            # Pipeline proposals list: [{image_id, scale, ...}, ...]
            # scale field is already the half-extent in metres.
            object_proposals = [p for p in proposals
                                if p.get("object_id", p.get("track_id", args.object_idx)) == args.object_idx]
            if not object_proposals:
                from itertools import takewhile
                n_objects = len(list(takewhile(lambda x: x["image_id"] == 0, proposals)))
                object_proposals = proposals[args.object_idx::n_objects]
            if not object_proposals:
                sys.exit(f"No proposals found for object_idx={args.object_idx} in {proposals_path}")
            metric_scale = object_proposals[0]["scale"]
            print(f"Read scale={metric_scale:.4f} m from {proposals_path.name} (object {args.object_idx})")

    # ── Output path ───────────────────────────────────────────────────────────
    if args.output is not None:
        out_path = Path(args.output)
    else:
        out_path = mesh_path.with_name(mesh_path.stem + "_metric" + mesh_path.suffix)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Rescale ───────────────────────────────────────────────────────────────
    print(f"Loading mesh: {mesh_path}")
    mesh = normalize_mesh(mesh_path)           # centred, half-extent = 1.0

    mesh.apply_scale(metric_scale)             # half-extent → metric_scale metres

    print(f"Exporting rescaled mesh ({metric_scale:.4f} m half-extent) → {out_path}")
    mesh.export(str(out_path))
    print("Done.")


if __name__ == "__main__":
    main()
