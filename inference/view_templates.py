#!/usr/bin/env python3
"""Generate a self-contained HTML grid of rendered template views."""

import argparse
import base64
import sys
from pathlib import Path

COLS = 4
ROWS = 600


def img_to_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def make_html(img_dir: Path, out_path: Path) -> None:
    imgs = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    imgs = sorted(set(imgs))[:COLS * ROWS]

    if not imgs:
        print(f"No images found in {img_dir}", file=sys.stderr)
        sys.exit(1)

    cells = []
    for i, p in enumerate(imgs):
        b64 = img_to_b64(p)
        ext = p.suffix.lstrip(".")
        mime = "jpeg" if ext in ("jpg", "jpeg") else "png"
        cells.append(
            f'<td style="padding:2px;text-align:center;vertical-align:top">'
            f'<img src="data:image/{mime};base64,{b64}" style="width:100%;display:block"/>'
            f'<div style="font-size:10px;color:#888">{i:04d}</div>'
            f'</td>'
        )

    rows_html = []
    for r in range(0, len(cells), COLS):
        row_cells = cells[r:r + COLS]
        rows_html.append("<tr>" + "".join(row_cells) + "</tr>")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>Template Views — {img_dir.name}</title>
<style>
  body {{ font-family: monospace; background: #111; color: #eee; margin: 8px; }}
  h2 {{ margin-bottom: 8px; }}
  table {{ border-collapse: collapse; table-layout: fixed; width: 100%; }}
  td {{ width: {100 // COLS}%; }}
</style>
</head>
<body>
<h2>{img_dir}</h2>
<p>{len(cells)} images &nbsp;|&nbsp; {COLS} columns &times; {ROWS} rows</p>
<table>
{"".join(rows_html)}
</table>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}  ({len(cells)} images)")


def main():
    parser = argparse.ArgumentParser(description="Render template views as a self-contained HTML grid.")
    parser.add_argument("img_dir", type=Path, help="Directory containing template images")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output HTML path (default: <img_dir>/templates_grid.html)")
    args = parser.parse_args()

    img_dir = args.img_dir.resolve()
    if not img_dir.is_dir():
        print(f"Not a directory: {img_dir}", file=sys.stderr)
        sys.exit(1)

    out = args.output or img_dir / "templates_grid.html"
    make_html(img_dir, out)


if __name__ == "__main__":
    main()
