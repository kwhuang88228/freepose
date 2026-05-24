import argparse, glob, os, sys, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from PIL import Image
from depth_anything_3.api import DepthAnything3
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Run Depth Anything 3 inference and save depth maps as 16-bit PNGs in millimeters.")
    parser.add_argument("frames_dir", help="Directory containing input PNG frames.")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: <frames_dir>/depth_mm).")
    parser.add_argument("--model", default="depth-anything/DA3NESTED-GIANT-LARGE", help="HuggingFace model repo or local path.")
    parser.add_argument("--device", default="cuda", help="Torch device (default: cuda).")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    model = DepthAnything3.from_pretrained(args.model)
    model = model.to(device=device)

    images = sorted(glob.glob(os.path.join(args.frames_dir, "*.png")))
    if not images:
        raise FileNotFoundError(f"No PNG files found in {args.frames_dir}")

    out_dir = args.out_dir or os.path.join(args.frames_dir, "depth_mm")
    os.makedirs(out_dir, exist_ok=True)

    batch_size = 200
    for batch_start in range(0, len(images), batch_size):
        batch = images[batch_start:batch_start + batch_size]

        prediction = model.inference(batch)

        # prediction.depth is [N, H, W] float32 in meters; convert to millimeters
        depth_mm = prediction.depth * 1000.0

        for i, img_path in enumerate(tqdm(batch, desc=f"Saving batch {batch_start // batch_size + 1}")):
            stem = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(out_dir, f"{stem}.png")
            Image.fromarray(depth_mm[i].astype(np.uint16)).save(out_path)
            print(f"Saved {out_path}  min={depth_mm[i].min():.1f}mm  max={depth_mm[i].max():.1f}mm")


if __name__ == "__main__":
    main()
