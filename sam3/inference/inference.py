"""
SAM 3.1 video inference: track objects via text prompt across all frames.

Usage:
    python inference.py --video_path <path> --text_prompt <object> [--output_dir <dir>]

Returns per-frame dicts with:
    binary_masks  : {obj_id: np.ndarray (H, W, bool)}
    bboxes        : {obj_id: [x1, y1, x2, y2]}  pixel-coord XYXY
    scores        : {obj_id: float}

When --output_dir is given, saves three subdirectories of PNG files:
    binary_mask/     frame{N:05d}.png  — all detection masks (grayscale 0/255), no boxes
    boxes/           frame{N:05d}.png  — all detection bounding boxes on the frame, no masks
    masks_overlay/   frame{N:05d}.png  — colored semi-transparent masks + bounding boxes + scores
"""

import argparse
import glob
import os
import subprocess
from tqdm import tqdm

import cv2
import numpy as np
from scipy import ndimage
from sam3.model_builder import build_sam3_multiplex_video_predictor


def _load_video_frames(video_path):
    if video_path.endswith(".mp4"):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames
    else:
        paths = glob.glob(os.path.join(video_path, "*.png"))
        try:
            paths.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
        except ValueError:
            paths.sort()
        return paths


def _mask_to_bbox(mask):
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    return [int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])]


def run_inference(video_path, text_prompt, score_threshold=None):
    """
    Run SAM 3.1 on a video with a text prompt.

    Args:
        video_path: path to an .mp4 file or a folder of .png frames
        text_prompt: natural language description of the object(s) to track
        score_threshold: optional override for the detector's
            `score_threshold_detection` (default 0.4 from the builder)

    Returns:
        results: dict {frame_idx -> {"binary_masks", "bboxes", "masks_overlay"}}
            binary_masks  : {obj_id: np.ndarray (H, W) bool}
            bboxes        : {obj_id: [x1, y1, x2, y2] in pixel coords}
            masks_overlay : np.ndarray (H, W, 3) uint8 RGB
        video_frames: list of frames (np.ndarray or str paths) used for visualization
    """
    predictor = build_sam3_multiplex_video_predictor()
    if score_threshold is not None:
        predictor.model.score_threshold_detection = score_threshold
    # Disable buffering so partial results survive a mid-video crash.
    # Default hotstart_delay=15 + postprocess_batch_size=16 means the first
    # ~15 frames are buffered before any yield — useless if the model crashes early.
    predictor.model.hotstart_delay = 0
    predictor.model.postprocess_batch_size = 1
    video_frames = _load_video_frames(video_path)

    response = predictor.handle_request(
        request=dict(type="start_session", resource_path=video_path)
    )
    session_id = response["session_id"]

    predictor.handle_request(
        request=dict(type="add_prompt", session_id=session_id, frame_index=0, text=text_prompt)
    )

    raw_per_frame = {}
    try:
        for response in predictor.handle_stream_request(
            request=dict(type="propagate_in_video", session_id=session_id)
        ):
            raw_per_frame[response["frame_index"]] = response["outputs"]
    except RuntimeError as e:
        # Propagation crashed mid-video (known SAM3 multiplex orphan-state bug).
        # Keep whatever frames completed so we can still visualize them.
        print(f"[WARN] propagation crashed at frame ~{max(raw_per_frame) + 1 if raw_per_frame else 0}: {e}")
        print(f"[WARN] proceeding with {len(raw_per_frame)} successfully propagated frames")

    try:
        predictor.handle_request(request=dict(type="close_session", session_id=session_id))
    except Exception as e:
        print(f"[WARN] close_session failed: {e}")

    results = {}
    for frame_idx in tqdm(sorted(raw_per_frame), desc="Running inference on frames"):
        out = raw_per_frame[frame_idx]
        obj_ids = out["out_obj_ids"].tolist()
        binary_masks_tensor = out["out_binary_masks"]

        frame_data = video_frames[frame_idx]
        if isinstance(frame_data, str):
            img = cv2.cvtColor(cv2.imread(frame_data), cv2.COLOR_BGR2RGB)
        else:
            img = frame_data
        H, W = img.shape[:2]

        probs = out["out_probs"]
        probs_arr = probs if isinstance(probs, np.ndarray) else probs.cpu().numpy()
        probs_flat = probs_arr.ravel()

        binary_masks = {}
        bboxes = {}
        scores = {}
        kept_indices = []
        kept_masks = []
        for i, obj_id in enumerate(obj_ids):
            raw_mask = binary_masks_tensor[i]
            mask = raw_mask.astype(bool) if isinstance(raw_mask, np.ndarray) else raw_mask.cpu().numpy().astype(bool)
            # Drop connected components smaller than 5 pixels (noise)
            labeled, n = ndimage.label(mask)
            if n > 0:
                sizes = ndimage.sum(mask, labeled, range(1, n + 1))
                small = np.where(sizes < 5)[0] + 1
                if len(small) > 0:
                    mask = mask & ~np.isin(labeled, small)
            if mask.any():
                binary_masks[obj_id] = mask
                bboxes[obj_id] = _mask_to_bbox(mask)
                scores[obj_id] = float(probs_flat[i])
                kept_indices.append(i)
                kept_masks.append(mask)

        results[frame_idx] = {
            "binary_masks": binary_masks,
            "bboxes": bboxes,
            "scores": scores,
        }

    return results, video_frames


_PALETTE_BGR = [
    (0, 255, 0), (0, 0, 255), (255, 0, 0),
    (0, 255, 255), (255, 0, 255), (255, 255, 0),
    (0, 165, 255), (128, 0, 128),
]


def save_results(results, output_dir, video_frames):
    masks_dir = os.path.join(output_dir, "binary_mask")
    boxes_dir = os.path.join(output_dir, "boxes")
    overlay_dir = os.path.join(output_dir, "masks_overlay")
    for d in (masks_dir, boxes_dir, overlay_dir):
        os.makedirs(d, exist_ok=True)

    for frame_idx, frame_data in tqdm(sorted(results.items()), desc="Saving results"):
        scores = frame_data.get("scores", {})
        binary_masks = frame_data["binary_masks"]
        bboxes = frame_data["bboxes"]

        # Load original frame in BGR
        frame_data_raw = video_frames[frame_idx]
        if isinstance(frame_data_raw, str):
            img_bgr = cv2.imread(frame_data_raw)
        else:
            img_bgr = cv2.cvtColor(frame_data_raw, cv2.COLOR_RGB2BGR)
        H, W = img_bgr.shape[:2]

        # binary_mask: grayscale (0/255) union of all detection masks, no boxes, no text
        masks_gray = np.zeros((H, W), dtype=np.uint8)
        for mask in binary_masks.values():
            masks_gray[mask] = 255
        cv2.imwrite(os.path.join(masks_dir, f"frame{frame_idx:05d}.png"), masks_gray)

        # boxes: all detection bounding boxes on the frame, no masks
        boxes_bgr = img_bgr.copy()
        for obj_id, bbox in bboxes.items():
            if bbox is None:
                continue
            color = _PALETTE_BGR[obj_id % len(_PALETTE_BGR)]
            x1, y1, x2, y2 = bbox
            cv2.rectangle(boxes_bgr, (x1, y1), (x2, y2), color, 2)
            if obj_id in scores:
                cv2.putText(boxes_bgr, f"{scores[obj_id]:.2f}", (x1, max(y1 - 4, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imwrite(os.path.join(boxes_dir, f"frame{frame_idx:05d}.png"), boxes_bgr)

        # masks_overlay: colored semi-transparent masks + bounding boxes + scores
        overlay_bgr = img_bgr.copy()
        for obj_id, mask in binary_masks.items():
            color = _PALETTE_BGR[obj_id % len(_PALETTE_BGR)]
            colored = np.zeros_like(overlay_bgr)
            colored[mask] = color
            overlay_bgr = cv2.addWeighted(overlay_bgr, 1.0, colored, 0.5, 0)
        for obj_id, bbox in bboxes.items():
            if bbox is None:
                continue
            color = _PALETTE_BGR[obj_id % len(_PALETTE_BGR)]
            x1, y1, x2, y2 = bbox
            cv2.rectangle(overlay_bgr, (x1, y1), (x2, y2), color, 2)
            if obj_id in scores:
                cv2.putText(overlay_bgr, f"{scores[obj_id]:.2f}", (x1, max(y1 - 4, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imwrite(os.path.join(overlay_dir, f"frame{frame_idx:05d}.png"), overlay_bgr)

    _save_videos(output_dir, fps=16)
    print(f"Saved results to {output_dir}")


def _save_videos(output_dir, fps=16):
    subdirs = ["binary_mask", "boxes", "masks_overlay"]
    for subdir in subdirs:
        frame_dir = os.path.join(output_dir, subdir)
        frames = sorted(glob.glob(os.path.join(frame_dir, "frame*.png")))
        if not frames:
            continue
        out_path = os.path.join(output_dir, f"{subdir}.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-framerate", str(fps),
             "-i", os.path.join(frame_dir, "frame%05d.png"),
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
             out_path],
            check=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, help="Path to .mp4 or PNG folder")
    parser.add_argument("--text_prompt", required=True, help="Object(s) to track (e.g. 'person')")
    parser.add_argument("--output_dir", default=None, help="Directory to save results")
    parser.add_argument("--score_threshold", type=float, default=None,
                        help="Override detector score threshold (default: 0.4)")
    args = parser.parse_args()

    results, video_frames = run_inference(
        args.video_path, args.text_prompt, score_threshold=args.score_threshold
    )

    print(f"Processed {len(results)} frames")
    if results:
        all_obj_ids = sorted({oid for r in results.values() for oid in r["binary_masks"].keys()})
        print(f"Objects detected (union over frames): {all_obj_ids}")

        # Per-object score summary across frames
        print(f"\nPer-object confidence summary:")
        print(f"  {'obj_id':>6}  {'n_frames':>9}  {'mean':>6}  {'min':>6}  {'max':>6}")
        for oid in all_obj_ids:
            scores = [r["scores"][oid] for r in results.values() if oid in r.get("scores", {})]
            if scores:
                print(f"  {oid:>6}  {len(scores):>9}  {sum(scores)/len(scores):>6.3f}  "
                      f"{min(scores):>6.3f}  {max(scores):>6.3f}")

        # Per-frame scores (so you can spot which obj_id is the "real" spoon)
        print(f"\nPer-frame scores:")
        for fidx in sorted(results.keys()):
            r = results[fidx]
            score_str = ", ".join(f"obj{oid}={r['scores'][oid]:.3f}"
                                  for oid in sorted(r["scores"].keys()))
            print(f"  frame {fidx:5d}: {score_str if score_str else '(no detections)'}")

    if args.output_dir:
        save_results(results, args.output_dir, video_frames)
