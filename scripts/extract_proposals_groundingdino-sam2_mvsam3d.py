"""
extract_proposals_ground_video_mvsam3d.py

Stage 1 of the MV-SAM3D pipeline:
  1. Detect objects in frame 0 with Grounding DINO.
  2. Track masks across all frames with SAM2.
  3. Prepare MV-SAM3D input structure (images + masks).
  4. Save proposals JSON (mesh paths filled in after MV-SAM3D inference).

Debug outputs → data/results/mvsam3d/<video>/
  01_detection_tracking/detection/   — Grounding DINO detection visualisation
  01_detection_tracking/tracking/    — SAM2 box overlays (all frames)

MV-SAM3D input → data/results/mvsam3d/<video>/mvsam3d_input/
  images/           — N evenly-spaced video frames
  {prompt_slug}/    — corresponding binary masks ({name}_mask.png)
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
import random

import cv2
import numpy as np
import torch
from loguru import logger
from sam2.build_sam import build_sam2_video_predictor
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from src.pipeline.utils import Proposals, mask_to_bbox

_FREEPOSE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_FREEPOSE_ROOT / "inference"))
from frames_to_video import frames_to_video

random.seed(10)


def get_init_bboxes(image, text_prompt, box_thresh, text_thresh, device="cuda"):
    assert isinstance(image, np.ndarray)
    assert len(image.shape) == 3 and image.shape[2] == 3

    logger.info("Loading Grounding DINO model")
    model_id  = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model     = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_thresh,
        text_threshold=text_thresh,
        target_sizes=[image.shape[:2]],
    )[0]

    bboxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = results["labels"]
    idxs   = np.where(np.array(labels) != '')[0]
    bboxes = [bboxes[i] for i in idxs]
    scores = [scores[i] for i in idxs]
    labels = [labels[i] for i in idxs]

    if len(scores) > 0:
        best   = int(np.argmax(scores))
        bboxes = [bboxes[best]]
        scores = [scores[best]]
        labels = [labels[best]]

    return bboxes, scores, labels


def track_with_sam2(video_dir, bboxes, scores, frame_paths, reverse=False, device="cuda"):
    logger.info("Loading SAM2 model")
    checkpoint = "./data/checkpoints/sam2_hiera_large.pt"
    model_cfg  = "sam2_hiera_l.yaml"

    predictor       = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
    inference_state = predictor.init_state(video_path=str(video_dir))

    logger.info("Tracking masks with SAM2")
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        for object_id, (bbox, score) in enumerate(zip(bboxes, scores)):
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=object_id,
                box=bboxes[object_id],
            )

        ignore_objects = set()
        tracking_output = {}

        start_frame = len(frame_paths) - 1 if reverse else 0
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(
            inference_state, reverse=reverse, start_frame_idx=start_frame
        ):
            scores_frame = [1.0] * len(obj_ids)
            masks  = [(mask_logits[i] > 0.0)[0] for i in range(len(obj_ids))]

            boxes  = []
            for i, mask in enumerate(masks):
                if mask.sum() < 100:
                    ignore_objects.add(i)
                    boxes.append(None)
                    continue
                bbox = mask_to_bbox(mask.cpu().numpy())
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                if w < 10 or h < 10:
                    ignore_objects.add(i)
                    boxes.append(None)
                    continue
                boxes.append(bbox)

            tracking_output[frame_idx] = {
                "boxes":  boxes,
                "masks":  masks,
                "scores": scores_frame,
            }

    if ignore_objects:
        logger.info(f"Ignoring objects: {ignore_objects}")

    obj_idxs = sorted(list(ignore_objects))[::-1]
    for output in tracking_output.values():
        for idx in obj_idxs:
            output["boxes"].pop(idx)
            output["masks"].pop(idx)
            output["scores"].pop(idx)
        valid_boxes = [b if b is not None else np.zeros(4) for b in output["boxes"]]
        output["boxes"] = torch.tensor(np.array(valid_boxes)).to(device)
        output["masks"] = torch.stack(output["masks"]).to(device)

    return tracking_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",       type=str, required=True)
    parser.add_argument("--box_thresh",  type=float, default=0.2)
    parser.add_argument("--text_thresh", type=float, default=0.2)
    parser.add_argument("--reverse",     action="store_true")
    parser.add_argument("--prompt",      type=str, default="objects.")
    parser.add_argument("--num_views",   type=int, default=6,
                        help="Number of evenly-spaced frames to select for MV-SAM3D (default: 6)")
    args = parser.parse_args()

    device      = "cuda" if torch.cuda.is_available() else "cpu"
    video_dir   = Path("data/datasets/videos") / args.video
    frame_paths = sorted([p for p in video_dir.iterdir() if p.suffix.lower() == ".png"])

    results_dir = _FREEPOSE_ROOT / "data" / "results" / "mvsam3d" / args.video
    results_dir.mkdir(parents=True, exist_ok=True)
    prompt_slug = args.prompt.rstrip(".").strip().replace(" ", "_").replace("/", "_") or "objects"
    output_file = results_dir / f"mvsam3d_{args.video}_{prompt_slug}.json"

    debug_detection = results_dir / "01_detection_tracking" / "detection"
    debug_tracking  = results_dir / "01_detection_tracking" / "tracking"
    for d in [debug_detection, debug_tracking]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Stage 1a: Grounding DINO detection on frame 0 ─────────────────────────
    image_path = frame_paths[-1 if args.reverse else 0]
    image      = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB).astype(np.uint8)
    bboxes, scores, labels = get_init_bboxes(
        image, args.prompt, args.box_thresh, args.text_thresh, device=device
    )

    from scripts.vis_detections_video import vis_detections
    viz_path = debug_detection / f"detections_{args.video}.png"
    vis_detections(image, bboxes, viz_path, xywh=False, labels=labels, scores=scores)
    logger.info(f"Saved detection viz → {viz_path}")

    # ── Stage 1b: SAM2 tracking ───────────────────────────────────────────────
    tracking_output = track_with_sam2(
        video_dir, bboxes, scores, frame_paths, reverse=args.reverse, device=device
    )

    sam2_boxes_dir         = debug_tracking / "boxes"
    sam2_masks_overlay_dir = debug_tracking / "masks_overlay"
    sam2_binary_masks_dir  = debug_tracking / "binary_masks"
    for d in [sam2_boxes_dir, sam2_masks_overlay_dir, sam2_binary_masks_dir]:
        d.mkdir(exist_ok=True)

    _colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    for frame_idx, output in tracking_output.items():
        frame_img    = cv2.imread(str(frame_paths[frame_idx]))
        boxes_np     = output["boxes"].cpu().numpy()
        masks_np     = output["masks"].cpu().numpy()
        box_overlay  = frame_img.copy()
        mask_overlay = frame_img.copy()
        for obj_idx, (mask, box) in enumerate(zip(masks_np, boxes_np)):
            color = _colors[obj_idx % len(_colors)]
            colored_mask = np.zeros_like(frame_img)
            colored_mask[mask.astype(bool)] = color
            mask_overlay = cv2.addWeighted(mask_overlay, 1.0, colored_mask, 0.4, 0)
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(box_overlay, (x1, y1), (x2, y2), color, 2)
        cv2.imwrite(str(sam2_boxes_dir / f"{frame_idx:06d}.png"), box_overlay)
        cv2.imwrite(str(sam2_masks_overlay_dir / f"{frame_idx:06d}.png"), mask_overlay)

        for obj_idx, mask in enumerate(masks_np):
            cv2.imwrite(
                str(sam2_binary_masks_dir / f"{frame_idx:06d}_mask.png"),
                (mask.astype(np.uint8)) * 255,
            )

    logger.info(f"SAM2 boxes        → {sam2_boxes_dir}")
    logger.info(f"SAM2 masks        → {sam2_masks_overlay_dir}")
    logger.info(f"SAM2 binary masks → {sam2_binary_masks_dir}")

    # Convert selected frames to video
    frames_to_video(sam2_boxes_dir, debug_tracking / "boxes.mp4", fps=30)
    frames_to_video(sam2_masks_overlay_dir, debug_tracking / "masks_overlay.mp4", fps=30)
    frames_to_video(sam2_binary_masks_dir, debug_tracking / "binary_masks.mp4", fps=30)
    logger.info(f"Saved selected frames video → {debug_tracking}")

    # ── Stage 1c: Prepare MV-SAM3D input structure ────────────────────────────
    valid_frame_idxs = sorted(tracking_output.keys())
    num_views = min(args.num_views, len(valid_frame_idxs))
    step = len(valid_frame_idxs) / num_views
    # selected_idxs = [valid_frame_idxs[int(i * step)] for i in range(num_views)]
    selected_idxs = random.sample(range(len(valid_frame_idxs)), num_views)

    mvsam3d_input_dir = results_dir / "mvsam3d_input"
    images_dir        = mvsam3d_input_dir / "images"
    masks_dir         = mvsam3d_input_dir / prompt_slug
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    selected_names = []
    for frame_idx in selected_idxs:
        frame_name = f"{frame_idx:06d}"
        src_img  = frame_paths[frame_idx]
        dst_img  = images_dir / f"{frame_name}.png"
        src_mask = sam2_binary_masks_dir / f"{frame_name}_mask.png"
        dst_mask = masks_dir / f"{frame_name}_mask.png"

        if not dst_img.exists():
            shutil.copy2(str(src_img), str(dst_img))
        if src_mask.exists() and not dst_mask.exists():
            shutil.copy2(str(src_mask), str(dst_mask))

        selected_names.append(frame_name)

    logger.info(f"MV-SAM3D input prepared → {mvsam3d_input_dir}")
    logger.info(f"Selected {num_views} frames: {selected_names}")

    # Save selected frame names so downstream stages can read them
    names_file = results_dir / f"mvsam3d_{args.video}_{prompt_slug}_selected_frames.txt"
    names_file.write_text("\n".join(selected_names))
    logger.info(f"Saved selected frame names → {names_file}")

    # ── Stage 1d: Build proposals JSON ────────────────────────────────────────
    all_proposals = {}
    for frame_idx, output in tracking_output.items():
        frame_img = cv2.cvtColor(
            cv2.imread(str(frame_paths[frame_idx])), cv2.COLOR_BGR2RGB
        ).astype(np.uint8)
        proposals = Proposals(
            frame_img,
            output,
            target_size=512,
            scene_id=0,
            frame_id=frame_idx,
            bbox_extend=0.2,
            mask_rgb=True,
        )
        proposals.meshes  = [None]   # filled in after MV-SAM3D inference
        proposals.scores  = [1.0]
        del proposals.features
        del proposals.proposals
        proposals.features  = None
        proposals.proposals = None
        all_proposals[frame_idx] = proposals

    out_data = []
    for frame_idx, proposals in all_proposals.items():
        out_data.extend(proposals.to_bop_dict())

    with open(output_file, "w") as f:
        json.dump(out_data, f)
    logger.info(f"Saved proposals → {output_file}")
