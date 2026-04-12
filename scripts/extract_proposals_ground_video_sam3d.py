"""
extract_proposals_ground_video_sam3d.py

Stage 1 of the SAM-3D pipeline:
  1. Detect objects in frame 0 with Grounding DINO.
  2. Track masks across all frames with SAM2.
  3. Generate a Gaussian splat for each tracked object using SAM-3D-Objects
     (from frame 0 image + frame-0 SAM2 mask).
  4. Save proposals JSON with splat paths in the 'mesh' field (instead of mesh IDs).

Debug outputs → data/results/sam3d/<video>/
  01_detection/   — Grounding DINO detection visualisation
  02_tracking/    — SAM2 box overlays (all frames)
  03_splats/      — multi-view renders of each generated Gaussian splat
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from sam2.build_sam import build_sam2_video_predictor
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from src.pipeline.utils import Proposals, mask_to_bbox

# ── SAM-3D imports ─────────────────────────────────────────────────────────────
_FREEPOSE_ROOT = Path(__file__).resolve().parent.parent
_SAM3D_ROOT    = _FREEPOSE_ROOT / "sam-3d-objects"
sys.path.insert(0, str(_SAM3D_ROOT / "notebook"))
sys.path.insert(0, str(_SAM3D_ROOT))

from inference import Inference  # SAM-3D notebook wrapper
from sam3d_objects.model.backbone.tdfy_dit.utils.render_utils import (
    render_frames,
    yaw_pitch_r_fov_to_extrinsics_intrinsics,
)
from sam3d_objects.model.backbone.tdfy_dit.utils.random_utils import sphere_hammersley_sequence

SAM3D_CONFIG_PATH = str(_SAM3D_ROOT / "checkpoints" / "hf" / "pipeline.yaml")


# ── Detection ──────────────────────────────────────────────────────────────────

def get_init_bboxes(image, text_prompt, box_thresh, text_thresh, device="cuda"):
    assert isinstance(image, np.ndarray)
    assert len(image.shape) == 3 and image.shape[2] == 3

    logger.info("Loading Grounding DINO model")

    model_id  = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model     = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    inputs  = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    
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


# ── SAM2 tracking ──────────────────────────────────────────────────────────────

def track_with_sam2(video_dir, bboxes, scores, frame_paths, reverse=False, device="cuda"):
    logger.info("Loading SAM2 model")
    checkpoint = "./data/checkpoints/sam2_hiera_large.pt"
    model_cfg  = "sam2_hiera_l.yaml"
    
    predictor  = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
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
            scores = [1.0] * len(obj_ids)
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
                "scores": scores,
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
        output["boxes"] = torch.tensor(np.array(valid_boxes)).cuda()
        output["masks"] = torch.stack(output["masks"]).cuda()

    return tracking_output


# ── SAM-3D splat generation ────────────────────────────────────────────────────

def generate_splats(image_frame0, tracking_output, video, device="cuda"):
    """Generate one Gaussian splat per tracked object using SAM-3D-Objects.

    Uses frame-0 image + frame-0 SAM2 mask for each object.
    Splats are saved to data/gaussian_splats/<video>/obj_<i>.ply.

    Returns:
        splat_paths: list of Path objects (relative to FREEPOSE_ROOT), one per object.
    """
    logger.info("Loading SAM-3D model")
    inference = Inference(SAM3D_CONFIG_PATH, compile=False)

    n_objects   = tracking_output[0]["masks"].shape[0]
    splat_dir   = _FREEPOSE_ROOT / "data" / "gaussian_splats" / video
    splat_dir.mkdir(parents=True, exist_ok=True)
    splat_paths = []

    for obj_idx in range(n_objects):
        ply_path = splat_dir / f"obj_{obj_idx}.ply"
        if ply_path.exists():
            logger.info(f"Splat for obj {obj_idx} already exists at {ply_path}, skipping.")
        else:
            logger.info(f"Generating SAM-3D splat for object {obj_idx}")
            mask_np = tracking_output[0]["masks"][obj_idx].cpu().numpy().astype(np.uint8)
            output  = inference(image_frame0.copy(), mask_np, seed=42)
            output["gs"].save_ply(str(ply_path))
            logger.info(f"Saved splat → {ply_path}")

        # Store path relative to FREEPOSE_ROOT so downstream scripts can load it
        splat_paths.append(str(ply_path.relative_to(_FREEPOSE_ROOT)))

    return splat_paths


# ── Debug visualisation helpers ────────────────────────────────────────────────

def _render_splat_views(gs, output_path: Path, n_views: int = 6):
    """Render a multi-view strip of the Gaussian splat for debugging."""
    n_total   = max(n_views, 30)
    step      = n_total // n_views
    indices   = list(range(0, n_total, step))[:n_views]
    cams      = [sphere_hammersley_sequence(i, n_total) for i in indices]
    yaws      = [c[0] for c in cams]
    pitchs    = [c[1] for c in cams]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, rs=2, fovs=40)
    result    = render_frames(
        gs, extrinsics, intrinsics,
        options={"resolution": 420, "near": 0.8, "far": 1.6, "bg_color": (0, 0, 0), "backend": "gsplat"},
        verbose=False,
    )
    strip = np.concatenate(result["color"], axis=1)
    cv2.imwrite(str(output_path), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",       type=str)
    parser.add_argument("--box_thresh",  type=float, default=0.2)
    parser.add_argument("--text_thresh", type=float, default=0.2)
    parser.add_argument("--reverse",     action="store_true")
    parser.add_argument("--prompt",      type=str, default="objects.")
    args = parser.parse_args()

    device     = "cuda" if torch.cuda.is_available() else "cpu"
    video_dir  = Path("data/datasets/videos") / args.video
    frame_paths = sorted([p for p in video_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg"]])

    results_dir = (Path("data/results/sam3d") / args.video).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    output_file = results_dir / f"sam3d_{args.video}.json"

    # Debug directories
    debug_root      = _FREEPOSE_ROOT / "data" / "results" / "sam3d" / args.video
    debug_detection = debug_root / "01_detection"
    debug_tracking  = debug_root / "02_tracking"
    debug_splats    = debug_root / "03_splats"
    for d in [debug_detection, debug_tracking, debug_splats]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Stage 1a: Grounding DINO detection on frame 0 ─────────────────────────
    image_path = frame_paths[-1 if args.reverse else 0]
    image      = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB).astype(np.uint8)
    bboxes, scores, labels = get_init_bboxes(
        image, args.prompt, args.box_thresh, args.text_thresh, device=device
    )

    # Save detection visualisation
    from scripts.vis_detections_video import vis_detections
    viz_path = debug_detection / f"detections_{args.video}.png"
    vis_detections(image, bboxes, viz_path, xywh=False, labels=labels, scores=scores)
    logger.info(f"Saved detection viz → {viz_path}")

    # ── Stage 1b: SAM2 tracking ───────────────────────────────────────────────
    tracking_output = track_with_sam2(
        video_dir, bboxes, scores, frame_paths, reverse=args.reverse, device=device
    )

    # Save SAM2 box/mask overlays to results_dir (pipeline convention)
    sam2_boxes_dir = results_dir / "sam2_boxes"
    sam2_masks_dir = results_dir / "sam2_masks"
    masks_dir      = results_dir / "masks"
    for d in [sam2_boxes_dir, sam2_masks_dir, masks_dir]:
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
        cv2.imwrite(str(sam2_boxes_dir / f"{frame_idx:06d}.jpg"), box_overlay)
        cv2.imwrite(str(sam2_masks_dir / f"{frame_idx:06d}.jpg"), mask_overlay)

        # Raw binary masks
        for obj_idx, mask in enumerate(masks_np):
            cv2.imwrite(
                str(masks_dir / f"{frame_idx:06d}_mask.png"),
                (mask.astype(np.uint8)) * 255,
            )

        # Debug: save all frames to 02_tracking/
        cv2.imwrite(str(debug_tracking / f"{frame_idx:06d}_boxes.jpg"), box_overlay)

    logger.info(f"SAM2 boxes   → {sam2_boxes_dir}")
    logger.info(f"SAM2 masks   → {sam2_masks_dir}")
    logger.info(f"Binary masks → {masks_dir}")

    # ── Stage 1c: SAM-3D Gaussian splat generation ────────────────────────────
    splat_paths = generate_splats(image, tracking_output, args.video, device=device)

    # Load splats for visualisation and render multi-view strips
    from src.pipeline.retrieval.renderer_sam3d import load_gaussian
    for obj_idx, splat_path in enumerate(splat_paths):
        ply_abs  = _FREEPOSE_ROOT / splat_path
        views_path = debug_splats / f"obj_{obj_idx}_views.jpg"
        try:
            gs = load_gaussian(ply_abs, device=device)
            _render_splat_views(gs, views_path)
            logger.info(f"Saved splat views → {views_path}")
        except Exception as exc:
            logger.warning(f"Could not render splat views for obj {obj_idx}: {exc}")

    # ── Stage 1d: Build proposals JSON ────────────────────────────────────────
    # For each frame, set proposals.meshes to the splat path (same for all frames).
    all_proposals = {}
    for frame_idx, output in tracking_output.items():
        frame_img = cv2.cvtColor(cv2.imread(str(frame_paths[frame_idx])), cv2.COLOR_BGR2RGB).astype(np.uint8)
        proposals = Proposals(frame_img, output, 420, 0, frame_idx, bbox_extend=0.1, mask_rgb=True)
        proposals.meshes = list(splat_paths)       # one splat path per object
        proposals.scores = [1.0] * len(splat_paths)
        del proposals.features
        del proposals.proposals
        proposals.features  = None
        proposals.proposals = None
        all_proposals[frame_idx] = proposals

    out_file = []
    for frame_idx, proposals in all_proposals.items():
        out_file.extend(proposals.to_bop_dict())

    with open(output_file, "w") as f:
        json.dump(out_file, f)
    logger.info(f"Saved proposals → {output_file}")
