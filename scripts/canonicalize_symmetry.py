#!/usr/bin/env python3
"""
canonicalize_symmetry.py

Per-frame, select the top-N candidate rotation closest (SO(3) geodesic) to the
previous frame's selected rotation. Addresses symmetry / front-back-flip
ambiguities that produce multiple near-equivalent pose solutions per frame and
manifest as wild rotation jumps in the top-1 trajectory.

Greedy forward pass:
    Frame 0: keep top-1 (highest DINO score).
    Frame t>0: argmin_k geodesic(R_chosen[t-1], R_topn[t, k]).

Cross-reference each chosen rank against the logged top-N template images at
    data/results/<backend>/<video>/04_coarse_poses/retrieved_templates/
    <frame:06d>_obj<obj>_rank{rank}_<tmpl_id>_<score>.png

Usage:
    python -m scripts.canonicalize_symmetry \
        --video <video> --poses <stage7_csv> --backend mvsam3d
"""

import base64
import itertools
import re
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_topn_R(s: str) -> np.ndarray:
    return np.stack([
        np.array([float(x) for x in part.split()]).reshape(3, 3)
        for part in s.split("|")
    ])  # (K, 3, 3)


def _parse_topn_t(s: str) -> np.ndarray:
    return np.stack([
        np.array([float(x) for x in part.split()])
        for part in s.split("|")
    ])  # (K, 3)


def _parse_topn_score(s: str) -> np.ndarray:
    return np.array([float(x) for x in s.split()])  # (K,)


def _geodesic(R_prev: np.ndarray, R_candidates: np.ndarray) -> np.ndarray:
    """Geodesic distance (rad) from R_prev (3,3) to each of R_candidates (K,3,3)."""
    diffs = R_candidates @ R_prev.T                                   # (K, 3, 3)
    cos_theta = (np.trace(diffs, axis1=1, axis2=2) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)


def canonicalize_track(R_topn: np.ndarray, t_topn: np.ndarray, score_topn: np.ndarray):
    """Greedy forward selection. Returns (R, t, score, rank) per frame."""
    N = R_topn.shape[0]
    chosen_R = np.empty((N, 3, 3))
    chosen_t = np.empty((N, 3))
    chosen_s = np.empty(N)
    ranks    = np.zeros(N, dtype=int)

    chosen_R[0] = R_topn[0, 0]
    chosen_t[0] = t_topn[0, 0]
    chosen_s[0] = score_topn[0, 0]

    for f in range(1, N):
        dists = _geodesic(chosen_R[f - 1], R_topn[f])                 # (K,)
        best = int(np.argmin(dists))
        chosen_R[f] = R_topn[f, best]
        chosen_t[f] = t_topn[f, best]
        chosen_s[f] = score_topn[f, best]
        ranks[f] = best

    return chosen_R, chosen_t, chosen_s, ranks


_HTML_STYLE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Canonical selection</title>
<style>
  body { background:#111; color:#ddd; font-family:monospace; margin:0; padding:12px; }
  table { border-collapse:collapse; }
  th, td { padding:3px 6px; border:1px solid #333; vertical-align:top; font-size:11px; }
  th { background:#222; position:sticky; top:0; }
  img { width:120px; height:auto; display:block; }
  .chosen { outline:3px solid #00ff66; outline-offset:-3px; }
  .miss { color:#888; font-style:italic; }
</style></head><body>
<table>
"""


def _html_head(top_n: int) -> str:
    rank_cols = "".join(f"<th>rank {r}</th>" for r in range(top_n))
    return _HTML_STYLE + f"<tr><th>frame / obj</th><th>query</th>{rank_cols}</tr>\n"

_HTML_TAIL = "</table></body></html>\n"


def _img_data_uri(path: Path) -> str:
    """Return a base64 data: URI for a PNG so the HTML is self-contained."""
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _generate_selection_html(df_canonical: pd.DataFrame, results_dir: Path,
                             top_n: int = 5) -> None:
    """Per-row grid: query crop + top_n retrieved templates, with the chosen one outlined.

    All images are base64-embedded so the resulting HTML is portable.
    Reads images written by stage 7 (dino_inference_video_mvsam3d):
      - query crops: 04_coarse_poses/debug/query_img/<frame>_obj<obj>.png
      - templates:   04_coarse_poses/retrieved_templates/<frame>_obj<obj>_rank<r>_*.png
    """
    debug_dir = results_dir / "04_coarse_poses"
    query_dir = debug_dir / "debug" / "query_img"
    tmpl_dir  = debug_dir / "retrieved_templates"
    out_html  = debug_dir / "canonical_selection.html"

    if not tmpl_dir.exists():
        print(f"[warn] {tmpl_dir} not found, skipping HTML generation.")
        return

    # Index template files: (frame_idx, obj_idx) -> {rank: (absolute_path, tmpl_id)}
    pat = re.compile(r"^(\d{6})_obj(\d+)_rank(\d+)_(\d+)_.*\.png$")
    tmpl_lookup: dict = defaultdict(dict)
    for p in tmpl_dir.iterdir():
        m = pat.match(p.name)
        if m is None:
            continue
        frame, obj, rank, tmpl_id = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        tmpl_lookup[(frame, obj)][rank] = (p, tmpl_id)

    df_sorted = df_canonical.sort_values(["im_id", "obj_id"], kind="stable").reset_index(drop=True)

    rows_html = []
    for i, row in df_sorted.iterrows():
        frame   = int(row["im_id"])
        obj_id  = str(row["obj_id"])
        # Reconstruct obj_idx as stage 7 used it: position within this frame's group.
        obj_idx = int(i - df_sorted[df_sorted["im_id"] == frame].index.min())
        chosen  = int(row.get("canonical_rank", 0))

        q_abs  = query_dir / f"{frame:06d}_obj{obj_idx}.png"
        q_cell = (f'<img src="{_img_data_uri(q_abs)}">'
                  if q_abs.exists() else '<span class="miss">missing</span>')

        tmpl_cells = []
        for r in range(top_n):
            entry = tmpl_lookup.get((frame, obj_idx), {}).get(r)
            if entry is None:
                tmpl_cells.append('<td><span class="miss">missing</span></td>')
                continue
            tp, tmpl_id = entry
            cls = ' class="chosen"' if r == chosen else ""
            tmpl_cells.append(f'<td><img src="{_img_data_uri(tp)}"{cls}><br>tmpl {tmpl_id}</td>')

        obj_tag = Path(obj_id).stem
        rows_html.append(
            f"<tr><td>{frame:06d} / {obj_tag}<br>rank {chosen}</td>"
            f"<td>{q_cell}</td>"
            f"{''.join(tmpl_cells)}</tr>\n"
        )

    out_html.write_text(_html_head(top_n) + "".join(rows_html) + _HTML_TAIL)
    size_mb = out_html.stat().st_size / (1024 * 1024)
    print(f"Selection HTML → {out_html} ({size_mb:.1f} MB)")


def main(args):
    data_dir    = Path("data")
    results_dir = data_dir / "results" / args.backend / args.video 
    csv_path    = results_dir / args.poses

    df = pd.read_csv(csv_path)
    required = {"R_top5", "t_top5", "score_top5"}
    missing  = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"Input CSV {csv_path.name} is missing columns {missing}. "
            f"Delete it and re-run stage 7 (scripts.dino_inference_video_mvsam3d) "
            f"so top-N candidates are written."
        )

    n_objects = len(list(itertools.takewhile(
        lambda x: x == df.iloc[0]["im_id"], df["im_id"]
    )))

    out_df = df.copy()
    # Preserve the original top-1 selection alongside the canonicalized one for diagnostics.
    out_df["R_top1"]         = df["R"]
    out_df["t_top1"]         = df["t"]
    out_df["score_top1"]     = df["score"]
    out_df["canonical_rank"] = 0

    for obj_idx in range(n_objects):
        sub_idx = list(range(obj_idx, len(df), n_objects))
        sub     = df.iloc[sub_idx]

        top_n = args.top_n
        R_topn = np.stack([_parse_topn_R(s)     for s in sub["R_top5"]])[:, :top_n]      # (N, top_n, 3, 3)
        t_topn = np.stack([_parse_topn_t(s)     for s in sub["t_top5"]])[:, :top_n]      # (N, top_n, 3)
        s_topn = np.stack([_parse_topn_score(s) for s in sub["score_top5"]])[:, :top_n]  # (N, top_n)

        R_can, t_can, s_can, ranks = canonicalize_track(R_topn, t_topn, s_topn)

        for k, i in enumerate(sub_idx):
            out_df.at[i, "R"]              = " ".join(str(x) for x in R_can[k].flatten())
            out_df.at[i, "t"]              = " ".join(str(x) for x in t_can[k])
            out_df.at[i, "score"]          = float(s_can[k])
            out_df.at[i, "canonical_rank"] = int(ranks[k])

    out_csv = results_dir / (csv_path.stem + "_canonical.csv")
    out_df.to_csv(out_csv, index=False)

    n_nontop = int((out_df["canonical_rank"] != 0).sum())
    rank_hist = out_df["canonical_rank"].value_counts().sort_index().to_dict()
    print(f"Canonicalized poses → {out_csv}")
    print(f"  Frames where chosen != top-1: {n_nontop} / {len(out_df)} "
          f"({100 * n_nontop / max(len(out_df), 1):.1f}%)")
    print(f"  Rank histogram: {rank_hist}")

    _generate_selection_html(out_df, results_dir, top_n=args.top_n)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video",   type=str, required=True)
    parser.add_argument("--poses",   type=str, required=True,
                        help="CSV filename inside results/<backend>/<video>/ "
                             "with R_top5 / t_top5 / score_top5 candidate columns.")
    parser.add_argument("--backend", type=str, choices=["sam3d", "mvsam3d"], required=True)
    parser.add_argument("--top_n", type=int, default=5,
                        help="Number of top candidates to consider (default: 5).")
    args = parser.parse_args()
    main(args)
