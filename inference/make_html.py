import base64
import subprocess
import tempfile
from pathlib import Path


def encode_video(path: Path) -> str:
    """Re-encode to H.264 (browser-compatible) and return a base64 data URI."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(path),
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            "-acodec", "aac", "-movflags", "+faststart",
            str(tmp_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    data = base64.b64encode(tmp_path.read_bytes()).decode("utf-8")
    tmp_path.unlink()
    return f"data:video/mp4;base64,{data}"


def make_html(videos_dir: Path, output_path: Path) -> None:
    video_dirs = sorted(videos_dir.iterdir())

    rows = []
    for video_dir in video_dirs:
        if not video_dir.is_dir():
            continue
        name = video_dir.name
        bbox_mp4 = video_dir / f"{name}_bbox.mp4"
        orig_mp4 = video_dir / f"{name}.mp4"
        bbox_2d_mp4 = video_dir / f"{name}_2d_bbox.mp4"
        mask_mp4 = video_dir / f"{name}_mask.mp4"
        if not bbox_mp4.exists() or not orig_mp4.exists():
            print(f"Skipping {name}: missing video files")
            continue

        print(f"Encoding {name}...")
        orig_src = encode_video(orig_mp4)
        bbox_src = encode_video(bbox_mp4)
        bbox_2d_src = encode_video(bbox_2d_mp4) if bbox_2d_mp4.exists() else None
        mask_src = encode_video(mask_mp4) if mask_mp4.exists() else None

        extra_tds = ""
        if bbox_2d_src:
            extra_tds += f"""
      <td>
        <video width="640" controls>
          <source src="{bbox_2d_src}" type="video/mp4">
        </video>
      </td>"""
        if mask_src:
            extra_tds += f"""
      <td>
        <video width="640" controls>
          <source src="{mask_src}" type="video/mp4">
        </video>
      </td>"""

        ncols = 2 + (1 if bbox_2d_src else 0) + (1 if mask_src else 0)
        rows.append(f"""
    <tr>
      <td colspan="{ncols}"><b>{name}</b></td>
    </tr>
    <tr>
      <td>
        <video width="640" controls>
          <source src="{orig_src}" type="video/mp4">
        </video>
      </td>
      <td>
        <video width="640" controls>
          <source src="{bbox_src}" type="video/mp4">
        </video>
      </td>{extra_tds}
    </tr>""")

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>FreePose Results</title>
  <style>
    body {{ font-family: sans-serif; padding: 20px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    td {{ padding: 10px; vertical-align: top; }}
    tr:nth-child(odd) {{ background: #f5f5f5; }}
    video {{ display: block; }}
    p {{ margin: 0 0 4px; font-weight: bold; color: #555; }}
  </style>
</head>
<body>
  <h1>FreePose Results v2</h1>
  <table>
{"".join(rows)}
  </table>
</body>
</html>
"""
    output_path.write_text(html)
    print(f"HTML saved to {output_path}")


if __name__ == "__main__":
    videos_dir = Path("/share/hariharan/kh775/code/freepose/data/results/hd_epic_clips/v2")
    output_path = videos_dir / "index_v2.html"
    make_html(videos_dir, output_path)
