import base64
import os
import subprocess
import tempfile

videos = [
    "P01-20240202-110250_3_knife",
    "P01-20240202-161948_19_spatula",
    "P01-20240202-171220_0_ladle",
    "P01-20240202-195538_6_spoon",
]

suffixes = ["coarse_poses_bbox3d", "coarse_poses_gaussian", "tracked_bbox3d"]
base_dir = "/share/hariharan/kh775/code/freepose/data/results/sam3d/num_template=4800"


def encode_video(path):
    """Re-encode to H.264 with faststart, then base64-encode for browser embedding."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", path,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-movflags", "+faststart",
                "-an",  # no audio
                tmp_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with open(tmp_path, "rb") as f:
            data = f.read()
    finally:
        os.unlink(tmp_path)
    return base64.b64encode(data).decode("utf-8")


rows_html = []
for video_name in videos:
    cells = []
    for suffix in suffixes:
        mp4_path = os.path.join(base_dir, video_name, f"{video_name}_{suffix}.mp4")
        b64 = encode_video(mp4_path)
        cell = f"""
        <td style="padding:8px; text-align:center;">
          <video controls width="480">
            <source src="data:video/mp4;base64,{b64}" type="video/mp4">
          </video>
        </td>"""
        cells.append(cell)
    row = f"""
    <tr>
      <td style="padding:8px; font-weight:bold; white-space:nowrap; vertical-align:middle;">{video_name}</td>
      {"".join(cells)}
    </tr>"""
    rows_html.append(row)
    print(f"Encoded {video_name}")

html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>FreePose Results</title>
  <style>
    body {{ font-family: sans-serif; padding: 20px; }}
    table {{ border-collapse: collapse; }}
    tr:nth-child(even) {{ background: #f5f5f5; }}
    td {{ border: 1px solid #ddd; vertical-align: top; }}
  </style>
</head>
<body>
  <h2>FreePose Results — num_template=4800</h2>
  <table>
    <thead>
      <tr>
        <th style="padding:8px;">Video</th>
        <th style="padding:8px;">coarse_poses_bbox3d</th>
        <th style="padding:8px;">coarse_poses_gaussian</th>
        <th style="padding:8px;">tracked_bbox3d</th>
      </tr>
    </thead>
    <tbody>
      {"".join(rows_html)}
    </tbody>
  </table>
</body>
</html>"""

out_path = os.path.join(os.path.dirname(__file__), "freepose_sam3d_4800_views.html")
with open(out_path, "w") as f:
    f.write(html)

print(f"Saved to {out_path}")
