"""
extract_proposals_sam3_mvsam3d.py

Stage 1 (SAM3.1 variant) of the MV-SAM3D pipeline:
  1. Detect & track one object across all frames with SAM3.1 (text prompt → argmax confidence).
  2. Prepare MV-SAM3D input structure (images + masks).
  3. Save proposals JSON.

Debug outputs → data/results/mvsam3d/<video>/
  01_detection_tracking/tracking/   — SAM3.1 box/mask overlays (all frames)

MV-SAM3D input → data/results/mvsam3d/<video>/mvsam3d_input/
  images/           — N evenly-spaced video frames
  {prompt_slug}/    — corresponding binary masks ({name}_mask.png)
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger
from sam3.model_builder import build_sam3_multiplex_video_predictor
from sam3.train.masks_ops import rle_encode

_FREEPOSE_ROOT = Path(__file__).resolve().parent.parent

random.seed(10)


def mask_to_bbox(mask: np.ndarray) -> np.ndarray:
    """Binary mask (H, W) → [x1, y1, x2, y2] pixel bbox."""
    y_indices, x_indices = np.nonzero(mask)
    return np.array([x_indices.min(), y_indices.min(), x_indices.max(), y_indices.max()])


def build_proposals_json(tracking_output: dict) -> list:
    out_data = []
    for frame_idx, output in tracking_output.items():
        mask  = output["mask"]   # (H, W) numpy bool
        bbox  = output["box"]    # [x1, y1, x2, y2] pixels
        score = output["score"]

        x1, y1, x2, y2 = [int(v) for v in bbox]
        bop_bbox = [x1, y1, x2 - x1, y2 - y1]

        mask_tensor  = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)  # (1, H, W)
        segmentation = rle_encode(mask_tensor)[0]

        out_data.append({
            "bbox":        bop_bbox,
            "segmentation": segmentation,
            "mesh":        None,
            "score":       float(score),
            "scene_id":    0,
            "image_id":    int(frame_idx),
            "time":        0.01,
        })
    return out_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",     type=str, required=True)
    parser.add_argument("--prompt",    type=str, default="objects.")
    parser.add_argument("--num_views", type=int, default=6,
                        help="Number of evenly-spaced frames to select for MV-SAM3D (default: 6)")
    args = parser.parse_args()

    video_dir   = Path("data/datasets/videos") / args.video
    frame_paths = sorted([p for p in video_dir.iterdir() if p.suffix.lower() == ".png"])

    if not frame_paths:
        sys.exit(f"No PNG frames found in {video_dir}")

    results_dir = _FREEPOSE_ROOT / "data" / "results" / "mvsam3d" / args.video
    results_dir.mkdir(parents=True, exist_ok=True)
    prompt_slug = args.prompt.rstrip(".").strip().replace(" ", "_").replace("/", "_") or "objects"
    output_file = results_dir / f"mvsam3d_{args.video}_{prompt_slug}.json"

    debug_tracking    = results_dir / "01_detection_tracking" / "tracking"
    boxes_dir         = debug_tracking / "boxes"
    masks_overlay_dir = debug_tracking / "masks_overlay"
    binary_masks_dir  = debug_tracking / "binary_masks"
    for d in [boxes_dir, masks_overlay_dir, binary_masks_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── SAM3.1: multi-anchor tracking to reduce drift ────────────────────────
    # Run one independent session per anchor frame so the tracker never needs
    # to propagate more than N/num_anchors frames from a known-good detection.
    # For each video frame we use the result from the session whose anchor is
    # the nearest preceding keyframe.
    NUM_TRACKING_ANCHORS = 4

    logger.info("Building SAM3.1 predictor")
    predictor = build_sam3_multiplex_video_predictor()

    text_prompt = args.prompt.rstrip(".")
    N = len(frame_paths)
    anchor_frames = [i * N // NUM_TRACKING_ANCHORS for i in range(NUM_TRACKING_ANCHORS)]
    logger.info(f"Multi-anchor tracking: {NUM_TRACKING_ANCHORS} anchors at frames {anchor_frames}")

    # anchor_frame → {frame_idx → {"mask", "box", "score"}}
    anchor_results: dict[int, dict] = {}

    for anchor_frame in anchor_frames:
        logger.info(f"Starting SAM3.1 session anchored at frame {anchor_frame}")
        response   = predictor.handle_request(
            request=dict(type="start_session", resource_path=str(video_dir))
        )
        session_id = response["session_id"]

        logger.info(f"Adding text prompt '{text_prompt}' on frame {anchor_frame}")
        response = predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=anchor_frame,
                text=text_prompt,
            )
        )
        init_out = response["outputs"]
        obj_ids  = init_out["out_obj_ids"]
        probs    = init_out["out_probs"]

        if len(obj_ids) == 0:
            logger.warning(
                f"SAM3.1 found no objects matching '{text_prompt}' at frame {anchor_frame}; "
                f"skipping this anchor."
            )
            predictor.handle_request(request=dict(type="close_session", session_id=session_id))
            anchor_results[anchor_frame] = {}
            continue

        best_idx    = int(np.argmax(probs))
        best_obj_id = int(obj_ids[best_idx])
        logger.info(
            f"Anchor {anchor_frame}: keeping obj_id={best_obj_id} "
            f"(score={probs[best_idx]:.3f}) from {len(obj_ids)} detection(s)"
        )

        for oid in obj_ids:
            if int(oid) != best_obj_id:
                predictor.handle_request(
                    request=dict(type="remove_object", session_id=session_id, obj_id=int(oid))
                )

        logger.info(f"Propagating forward from anchor frame {anchor_frame}")
        session_output: dict[int, dict] = {}
        for resp in predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
                propagation_direction="forward",
            )
        ):
            fidx = resp["frame_index"]
            out  = resp["outputs"]
            bin_masks = out["out_binary_masks"]
            if bin_masks.shape[0] == 0 or not bin_masks[0].any():
                continue
            mask = bin_masks[0]
            if mask.sum() < 100:
                continue
            bbox = mask_to_bbox(mask)
            if (bbox[2] - bbox[0]) < 10 or (bbox[3] - bbox[1]) < 10:
                continue
            out_probs = out["out_probs"]
            score = float(out_probs[0]) if out_probs.shape[0] > 0 else 1.0
            session_output[fidx] = {"mask": mask, "box": bbox, "score": score}

        predictor.handle_request(request=dict(type="close_session", session_id=session_id))
        anchor_results[anchor_frame] = session_output
        logger.info(f"Anchor {anchor_frame}: {len(session_output)} frames propagated")

    # ── Merge: each frame uses the result from its nearest preceding anchor ───
    tracking_output = {}
    for frame_idx in range(N):
        # Nearest anchor frame that is ≤ frame_idx
        candidates = [a for a in anchor_frames if a <= frame_idx]
        if not candidates:
            continue
        nearest = max(candidates)
        if frame_idx in anchor_results.get(nearest, {}):
            tracking_output[frame_idx] = anchor_results[nearest][frame_idx]

    logger.info(f"Valid tracking frames: {len(tracking_output)}")
    if not tracking_output:
        sys.exit("No valid tracking frames — check the prompt or video.")

    # ── Debug visualizations ──────────────────────────────────────────────────
    _color = (255, 0, 0)
    for frame_idx, output in tracking_output.items():
        frame_img    = cv2.imread(str(frame_paths[frame_idx]))
        mask         = output["mask"]
        bbox         = output["box"].astype(int)
        colored_mask = np.zeros_like(frame_img)
        colored_mask[mask] = _color
        mask_overlay = cv2.addWeighted(frame_img.copy(), 1.0, colored_mask, 0.4, 0)
        box_overlay  = frame_img.copy()
        x1, y1, x2, y2 = bbox
        cv2.rectangle(box_overlay, (x1, y1), (x2, y2), _color, 2)
        cv2.imwrite(str(boxes_dir        / f"{frame_idx:06d}.png"),      box_overlay)
        cv2.imwrite(str(masks_overlay_dir / f"{frame_idx:06d}.png"),      mask_overlay)
        cv2.imwrite(str(binary_masks_dir  / f"{frame_idx:06d}_mask.png"), mask.astype(np.uint8) * 255)

    logger.info(f"SAM3.1 debug visualizations → {debug_tracking}")

    # ── Prepare MV-SAM3D input structure ─────────────────────────────────────
    valid_idxs    = sorted(tracking_output.keys())
    num_views     = min(args.num_views, len(valid_idxs))
    selected_idxs = random.sample(valid_idxs, num_views)

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
        src_mask = binary_masks_dir / f"{frame_name}_mask.png"
        dst_mask = masks_dir / f"{frame_name}_mask.png"
        if not dst_img.exists():
            shutil.copy2(str(src_img), str(dst_img))
        if src_mask.exists() and not dst_mask.exists():
            shutil.copy2(str(src_mask), str(dst_mask))
        selected_names.append(frame_name)

    logger.info(f"MV-SAM3D input prepared → {mvsam3d_input_dir}")
    logger.info(f"Selected {num_views} frames: {selected_names}")

    names_file = results_dir / f"mvsam3d_{args.video}_{prompt_slug}_selected_frames.txt"
    names_file.write_text("\n".join(selected_names))
    logger.info(f"Saved selected frame names → {names_file}")

    # ── Build proposals JSON ──────────────────────────────────────────────────
    out_data = build_proposals_json(tracking_output)
    with open(output_file, "w") as f:
        json.dump(out_data, f)
    logger.info(f"Saved proposals → {output_file}")
