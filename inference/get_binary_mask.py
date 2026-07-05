"""
get_binary_mask.py — single-image text-prompted binary segmentation.

Distilled from Stage 1 of the MV-SAM3D pipeline (extract_proposals_groundingdino-sam2_mvsam3d.py):
Grounding DINO turns a text prompt into the best-scoring box, then SAM2 turns that
box into a mask. Here we use SAM2's *image* predictor (the pipeline uses the video
predictor for tracking, which is overkill for a single frame).

Run from the freepose root so the SAM2 checkpoint/config paths resolve, e.g.
    python inference/get_binary_mask.py /path/to/image.png "cup"
"""

from pathlib import Path

import cv2
import numpy as np
import torch


def get_binary_mask(image, text, box_thresh=0.2, text_thresh=0.2, device=None):
    """Generate a binary mask for the object described by `text` in `image`.

    Args:
        image: RGB uint8 array (H, W, 3), or a path to an image file.
        text: free-form prompt, e.g. "cup" or "the red mug".
        box_thresh, text_thresh: Grounding DINO detection thresholds.
        device: torch device string; defaults to cuda if available.

    Returns:
        Boolean mask of shape (H, W). All-False if nothing is detected.
    """
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(image, (str, Path)):
        image = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2RGB)
    image = np.asarray(image).astype(np.uint8)
    assert image.ndim == 3 and image.shape[2] == 3, "image must be RGB (H, W, 3)"

    # Grounding DINO wants a lowercase prompt terminated by a period.
    prompt = text.strip().lower()
    prompt = prompt if prompt.endswith(".") else prompt + "."

    dino_id   = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(dino_id)
    dino      = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(device)

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = dino(**inputs)
    result = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        box_threshold=box_thresh, text_threshold=text_thresh,
        target_sizes=[image.shape[:2]],
    )[0]

    boxes  = result["boxes"].cpu().numpy()
    scores = result["scores"].cpu().numpy()
    if len(scores) == 0:
        return np.zeros(image.shape[:2], dtype=bool)
    box = boxes[int(np.argmax(scores))]  # keep only the highest-scoring detection

    sam2 = build_sam2("sam2_hiera_l.yaml", "./data/checkpoints/sam2_hiera_large.pt", device=device)
    predictor = SAM2ImagePredictor(sam2)
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        predictor.set_image(image)
        masks, _, _ = predictor.predict(box=box[None, :], multimask_output=False)

    return masks[0].astype(bool)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Text-prompted binary mask for a single image.")
    ap.add_argument("image", help="Path to input image.")
    ap.add_argument("text", help="Object prompt, e.g. 'cup'.")
    ap.add_argument("--output", default="mask.png", help="Where to save the binary mask PNG.")
    args = ap.parse_args()

    mask = get_binary_mask(args.image, args.text)
    cv2.imwrite(args.output, mask.astype(np.uint8) * 255)
    print(f"Saved mask ({int(mask.sum())} px) → {args.output}")
