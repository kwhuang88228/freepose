import argparse
import json
from collections import defaultdict
from itertools import takewhile
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from loguru import logger
from sam2.utils.amg import rle_to_mask

from src.pipeline.estimators.scale_estimators import GPT4ScaleEstimator
from src.pipeline.retrieval.clip import CLIPFeatureExtractor
from src.pipeline.utils import Proposals


def load_da3_depths(npz_path):
    """DA3 depth maps → {frame_stem: (H, W) float32}, at DA3's native resolution."""
    data = np.load(npz_path, allow_pickle=True)
    depth = data["depth"]
    stems = [Path(str(f)).stem for f in data["image_files"]]
    return {stem: np.asarray(depth[i], dtype=np.float32) for i, stem in enumerate(stems)}


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--video", type=str)
    args.add_argument("--proposals", type=str)
    args.add_argument("--backend", type=str, default="sam3d",
                      help="Pipeline backend: 'sam3d' or 'mvsam3d' (sets results subdirectory)")
    args.add_argument("--da3_depth", type=str, required=True,
                      help="Path to Depth-Anything-3 da3_output.npz; supplies the depth used for scale estimation.")
    args.add_argument("--results_dir", type=str, default=None,
                      help="Directory holding the proposals JSON and receiving the scaled output "
                           "(default: data/results/<backend>/<video>)")
    args = args.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    video_dir = Path("data") / "datasets" / "videos" / args.video
    frame_paths = sorted([p for p in video_dir.iterdir() if p.suffix.lower() in [".png"]])

    results_dir = (Path(args.results_dir) if args.results_dir
                   else Path("data") / "results" / args.backend / args.video).resolve()
    input_path = results_dir / args.proposals
    output_path = results_dir / args.proposals.replace(".json", "_gpt4_scaled.json")

    # Log scale-estimation internals (with source file:line of each log call) to a txt.
    log_path = results_dir / args.proposals.replace(".json", "_scale_log.txt")
    logger.add(str(log_path), format="{time:YYYY-MM-DD HH:mm:ss} | {file}:{line} | {message}",
               level="INFO", mode="w")

    with open(input_path, "r") as f:
        proposals_all = json.load(f)

    N_objects = len(list(takewhile(lambda x: x['image_id']==0, proposals_all)))
    

    clip = CLIPFeatureExtractor().to(device, dtype=torch.bfloat16)
    scale_estimator = GPT4ScaleEstimator(clip, scale_file="data/gpt4_scales.json")
    da3_depths = load_da3_depths(args.da3_depth)

    image_path = frame_paths[0]
    image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB).astype(
        np.uint8
    )
    image_h, image_w, _ = image.shape
    cx = image_w / 2.0
    cy = image_h / 2.0
    f = np.sqrt(image_h**2 + image_w**2)
    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1]]
    ).astype(float)

    props = defaultdict(list)
    for p in proposals_all:
        props[p["image_id"]].append(p)

    for frame_idx, frame_path in tqdm(enumerate(frame_paths), total=len(frame_paths)):
        if frame_path.stem not in da3_depths:
            continue
        image = cv2.cvtColor(
            cv2.imread(str(frame_paths[frame_idx])), cv2.COLOR_BGR2RGB
        ).astype(np.uint8)

        frame_props = props[frame_idx]
        masks = [rle_to_mask(p["segmentation"]) for p in frame_props]
        boxes = [np.array(p["bbox"]) for p in frame_props]
        scores = [p["score"] for p in frame_props]

        masks = torch.from_numpy(np.stack(masks))
        boxes = torch.from_numpy(np.stack(boxes))
        # convert bbox from xywh to xyxy
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]

        out = {"boxes": boxes, "masks": masks}
        proposals = Proposals(image, out, 224, bbox_extend=0.05)

        # DA3 metric depth for this frame, resized to full-frame resolution so it
        # aligns with the full-res masks and K used by the scale estimator.
        depth_pred = da3_depths[frame_path.stem]
        if depth_pred.shape != (image_h, image_w):
            depth_pred = cv2.resize(
                depth_pred, (image_w, image_h), interpolation=cv2.INTER_NEAREST
            )
        with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
            # Object's real-world size, used to resize the canonical mesh
            img_pred_scales = scale_estimator.estimate(proposals, depth_pred, K)

        logger.info(
            f"img_pred_scales (real-world size to resize canonical mesh) "
            f"frame {frame_idx} ({frame_path.stem}): {[float(s) for s in img_pred_scales]}"
        )

        for i, scale in enumerate(img_pred_scales):
            frame_props[i]["scale"] = float(scale.item())

    # replace scales with medians for each tracked object
    for object_idx in range(N_objects):
        object_proposals = proposals_all[object_idx::N_objects]
        scales = [x["scale"] for x in object_proposals]
        scale = np.median(scales)
        for p in object_proposals:
            p["scale"] = scale
    with open(str(output_path), "w") as f:
        json.dump(proposals_all, f)
