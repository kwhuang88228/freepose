"""
Generate a self-contained HTML file showing, for each frame:
  col 0 : frame with bounding box  (01_detection_tracking/tracking/boxes/XXXXXX.png)
  col 1 : query image               (04_coarse_poses/debug/query_img/...)
  col 2-6: retrieved templates   rank0 … rank4  (04_coarse_poses/debug/template_img/...)
All images are base64-embedded so the file is fully self-contained.
"""

import base64
import os
import re
from tqdm import tqdm


# ── paths ──────────────────────────────────────────────────────────────────────
VIDEO_NAMES = ["P01-20240202-110250_3_knife", "P01-20240202-161948_19_spatula", "P01-20240202-171220_0_ladle", "P01-20240202-195538_6_spoon"]
GROUP_NAME = "3d_templates"
for VIDEO_NAME in tqdm(VIDEO_NAMES):
    BASE = f"/share/hariharan/kh775/code/freepose/data/results/mvsam3d/{GROUP_NAME}/{VIDEO_NAME}"
    TRACKING_DIR = os.path.join(BASE, "01_detection_tracking", "tracking", "boxes")
    QUERY_DIR = os.path.join(BASE, "04_coarse_poses", "debug", "query_img")
    TEMPLATES_DIR = os.path.join(BASE, "04_coarse_poses", "debug", "template_img")
    OUT_PATH = os.path.join(os.path.dirname(__file__), f"retrieval_results_mvsam3d_{GROUP_NAME}_{VIDEO_NAME}.html")
    def encode_image(path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


    def img_tag(b64: str, label: str = "") -> str:
        title = f' title="{label}"' if label else ""
        return (
            f'<img src="data:image/png;base64,{b64}"{title} '
            f'style="max-width:160px; max-height:160px; display:block; margin:auto;">'
        )


    # ── collect frames ─────────────────────────────────────────────────────────────
    query_files = sorted(
        f for f in os.listdir(QUERY_DIR) if f.endswith(".png")
    )

    # build a lookup: frame_id -> {rank: (filename, tmpl_id, sim, tx, ty, tz)}
    # filename format: {_tag}_rank{rank}_{tmpl_id:04d}_{sim:.4f}_{tx:.3f}_{ty:.3f}_{tz:.3f}.png
    template_lookup = {}
    rank_re = re.compile(r"^(\d+)_obj\d+_rank(\d+)_(\d+)_([\d.]+)_([-\d.]+)_([-\d.]+)_([-\d.]+)(?:_roll[+\-\d.]+)?\.png$")
    for fname in os.listdir(TEMPLATES_DIR):
        m = rank_re.match(fname)
        if m:
            frame_id, rank = m.group(1), int(m.group(2))
            tmpl_id = m.group(3)
            sim = m.group(4)
            tx, ty, tz = m.group(5), m.group(6), m.group(7)
            template_lookup.setdefault(frame_id, {})[rank] = (fname, tmpl_id, sim, tx, ty, tz)

    # ── build rows ─────────────────────────────────────────────────────────────────
    rows_html = []
    for query_fname in tqdm(query_files):
        # e.g. "000042_obj0.png"  →  frame_id = "000042"
        frame_id = query_fname.split("_")[0]

        # column 0 – frame with bounding box
        box_path = os.path.join(TRACKING_DIR, f"{frame_id}.png")
        if os.path.exists(box_path):
            box_cell = img_tag(encode_image(box_path), f"{frame_id}.png")
        else:
            box_cell = "<span style='color:#aaa'>—</span>"
        cells = [
            f'<td style="padding:6px; text-align:center; vertical-align:middle;">'
            f'{box_cell}</td>'
        ]

        # column 1 – query image
        query_path = os.path.join(QUERY_DIR, query_fname)
        cells.append(
            f'<td style="padding:6px; text-align:center; vertical-align:middle;">'
            f'{img_tag(encode_image(query_path), query_fname)}</td>'
        )

        # columns 2-6 – rank0 … rank4
        frame_templates = template_lookup.get(frame_id, {})
        for rank in range(5):
            entry = frame_templates.get(rank)
            if entry:
                fname, tmpl_id, sim, tx, ty, tz = entry
                tpl_path = os.path.join(TEMPLATES_DIR, fname)
                caption_parts = [
                    f"<b>#{tmpl_id}</b>",
                    f'<span style="color:#555;">sim: {sim}</span>',
                    f'<span style="color:#555;">tx: {tx} ty: {ty} tz: {tz}</span>',
                ]
                caption = "<br>".join(caption_parts)
                cell_content = (
                    img_tag(encode_image(tpl_path), fname)
                    + f'<div style="font-size:11px; margin-top:4px;">{caption}</div>'
                )
            else:
                cell_content = "<span style='color:#aaa'>—</span>"
            cells.append(
                f'<td style="padding:6px; text-align:center; vertical-align:middle;">'
                f'{cell_content}</td>'
            )

        rows_html.append(
            f'<tr>'
            f'<td style="padding:6px; font-weight:bold; text-align:center; '
            f'vertical-align:middle; font-size:12px;">{frame_id}</td>'
            + "".join(cells)
            + "</tr>"
        )

    # ── assemble HTML ──────────────────────────────────────────────────────────────
    header_cells = "".join(
        f'<th style="padding:8px; background:#333; color:#fff;">{h}</th>'
        for h in ["Frame", "Bounding Box", "Query Image", "Rank 0", "Rank 1", "Rank 2", "Rank 3", "Rank 4"]
    )

    html = f"""<!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <title>FreePose – Retrieval Results</title>
    <style>
        body {{ font-family: sans-serif; padding: 20px; background: #fafafa; }}
        h2   {{ color: #222; }}
        table {{ border-collapse: collapse; }}
        tr:nth-child(even) {{ background: #f0f0f0; }}
        tr:nth-child(odd)  {{ background: #ffffff; }}
        td, th {{ border: 1px solid #ccc; }}
        tr:hover {{ background: #e8f0fe; }}
    </style>
    </head>
    <body>
    <h2>FreePose – Template Retrieval Results<br>
    </h2>
    <table>
        <thead><tr>{header_cells}</tr></thead>
        <tbody>
        {"".join(rows_html)}
        </tbody>
    </table>
    </body>
    </html>"""

    with open(OUT_PATH, "w") as f:
        f.write(html)

    print(f"\nSaved → {OUT_PATH}")
