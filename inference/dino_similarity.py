"""
dino_similarity.py

Compute the DINOv2-based similarity score between a single query image and a
single template image, using the exact same feature-extraction and masking
logic as dino_inference_video_sam3d.py / DinoPoseEstimatorSam3d.forward().

Typical validation workflow
---------------------------
Compare against the scores already embedded in the retrieved-template filenames
produced by Stage 4:

  data/results/sam3d/<video>/04_coarse_poses/debug/query_raw/<frame>_obj0.jpg
  data/results/sam3d/<video>/04_coarse_poses/retrieved_templates/
      <frame>_obj0_rank0_<tmpl_id>_<score>.jpg

Example:
  python inference/dino_similarity.py \\
      --query  data/results/sam3d/vid/04_coarse_poses/debug/query_raw/000000_obj0.jpg \\
      --template data/results/sam3d/vid/04_coarse_poses/retrieved_templates/000000_obj0_rank0_0042_0.8123.jpg

Notes on masking flags
----------------------
mask_query / mask_template
  Zero out background pixels in the image BEFORE feature extraction.
  Requires --query_mask / --template_mask to be provided as well.
  This mirrors the Proposals(mask_rgb=True) path and build_template_dict(mask_template=True).

query_fg_patches / template_fg_patches
  When averaging per-patch cosine similarities, restrict the average to
  foreground (object) patches only.
  Requires the corresponding pixel mask to derive the patch mask.
  This mirrors the use_query_fg_patches / use_template_fg_patches flags in
  DinoPoseEstimatorSam3d.forward().
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import einsum
from loguru import logger

_FREEPOSE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_FREEPOSE_ROOT))

from src.pipeline.estimators.pose_estimator import DinoPoseEstimator
from src.pipeline.retrieval.dino import DINOv2FeatureExtractor

device = "cuda" if torch.cuda.is_available() else "cpu"


# ── Image helpers ──────────────────────────────────────────────────────────────

def load_image(path: Path) -> torch.Tensor:
    """Load an image as a float32 [3, H, W] tensor normalised to [0, 1]."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1)  # [3, H, W]


def load_mask(path: Path) -> torch.Tensor:
    """Load a grayscale mask as a bool [H, W] tensor (non-zero → True)."""
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    return torch.from_numpy(gray > 0)  # bool [H, W]


# ── Core similarity computation ────────────────────────────────────────────────

def compute_similarity(
    query_img: torch.Tensor,        # [3, H, W] float 0-1
    template_img: torch.Tensor,     # [3, H, W] float 0-1
    feature_extractor: DINOv2FeatureExtractor,
    query_mask: torch.Tensor = None,       # bool [H, W], optional
    template_mask: torch.Tensor = None,    # bool [H, W], optional
    mask_query: bool = True,
    mask_template: bool = False,
    use_query_fg_patches: bool = True,
    use_template_fg_patches: bool = False,
    layer: int = 22,
) -> float:
    """Return the scalar similarity score between query and template.

    Replicates the per-patch cosine-similarity + masked-mean logic from
    DinoPoseEstimatorSam3d.forward() exactly.
    """
    # ── Apply pixel-level masking to images ───────────────────────────────────
    q_img = query_img.clone()
    if mask_query and query_mask is not None:
        q_img = q_img * query_mask.float().unsqueeze(0)  # zero out background
    elif mask_query and query_mask is None:
        logger.warning("--mask_query requested but no --query_mask provided; skipping pixel masking")

    t_img = template_img.clone()
    if mask_template and template_mask is not None:
        t_img = t_img * template_mask.float().unsqueeze(0)
    elif mask_template and template_mask is None:
        logger.warning("--mask_template requested but no --template_mask provided; skipping pixel masking")

    # ── Build patch masks (pure boolean ops, dtype-insensitive) ──────────────────
    effective_mask = None

    if use_query_fg_patches and query_mask is not None:
        q_patch_mask = DinoPoseEstimator._to_patch_mask(query_mask)         # [num_patches]
        effective_mask = q_patch_mask.unsqueeze(0).to(device)               # [1, num_patches]
    elif use_query_fg_patches and query_mask is None:
        logger.warning("--query_fg_patches requested but no --query_mask provided; using all patches")

    if use_template_fg_patches and template_mask is not None:
        t_patch_mask = DinoPoseEstimator._to_patch_mask(template_mask).unsqueeze(0).to(device)
        effective_mask = (effective_mask & t_patch_mask) if effective_mask is not None else t_patch_mask
    elif use_template_fg_patches and template_mask is None:
        logger.warning("--template_fg_patches requested but no --template_mask provided; using all patches")

    # ── Feature extraction + similarity, mirroring the pipeline exactly ───────
    # DinoPoseEstimatorSam3d.forward() is called inside torch.autocast(bfloat16).
    # _get_template_features always returns features as bfloat16 (via
    # .to('cuda', dtype=torch.bfloat16)), while query features come out of the
    # feature extractor as float32 (LayerNorm output).  The einsum therefore runs
    # on (bfloat16, float32) inside autocast, which downcasts to bfloat16.
    # We replicate that here: t_feat → bfloat16, einsum inside autocast.
    with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16):
        q_feat = feature_extractor(
            q_img.unsqueeze(0).to(device, dtype=torch.bfloat16),
            layer=layer, feature_type="patch",
        )  # [1, num_patches, D]  float32

        t_feat = feature_extractor(
            t_img.unsqueeze(0).to(device, dtype=torch.bfloat16),
            layer=layer, feature_type="patch",
        ).to(dtype=torch.bfloat16)  # cast to bfloat16, matching _get_template_features

        # (bfloat16, float32) einsum inside autocast → bfloat16, same as pipeline
        per_patch_sim = einsum(
            F.normalize(t_feat, dim=-1),
            F.normalize(q_feat, dim=-1),
            "b n d, b n d -> b n",
        )  # [1, num_patches]

        if effective_mask is not None:
            count = effective_mask.float().sum(dim=-1).clamp(min=1)         # [1]
            score = (per_patch_sim * effective_mask.float()).sum(dim=-1) / count
        else:
            score = per_patch_sim.mean(dim=-1)                              # [1]

    return float(score[0].float().cpu())


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    query_path    = Path(args.query)
    template_path = Path(args.template)

    query_img    = load_image(query_path)
    template_img = load_image(template_path)

    query_mask    = load_mask(Path(args.query_mask))    if args.query_mask    else None
    template_mask = load_mask(Path(args.template_mask)) if args.template_mask else None

    logger.info(f"Query:    {query_path}  ({query_img.shape[1]}×{query_img.shape[2]})")
    logger.info(f"Template: {template_path}  ({template_img.shape[1]}×{template_img.shape[2]})")
    if query_mask is not None:
        logger.info(f"Query mask:    {args.query_mask}")
    if template_mask is not None:
        logger.info(f"Template mask: {args.template_mask}")
    logger.info(
        f"Flags: mask_query={args.mask_query}  mask_template={args.mask_template}  "
        f"query_fg_patches={args.query_fg_patches}  template_fg_patches={args.template_fg_patches}  "
        f"layer={args.layer}"
    )

    feature_extractor = DINOv2FeatureExtractor().to(device, dtype=torch.bfloat16)

    score = compute_similarity(
        query_img, template_img, feature_extractor,
        query_mask=query_mask,
        template_mask=template_mask,
        mask_query=args.mask_query,
        mask_template=args.mask_template,
        use_query_fg_patches=args.query_fg_patches,
        use_template_fg_patches=args.template_fg_patches,
        layer=args.layer,
    )

    # Try to parse the reference score from the template filename, e.g.
    # 000000_obj0_rank0_0042_0.8123.jpg  →  0.8123
    ref_score = None
    stem_parts = template_path.stem.split("_")
    try:
        ref_score = float(stem_parts[-1])
    except ValueError:
        pass

    print(f"\nSimilarity score: {score:.6f}")
    if ref_score is not None:
        diff = abs(score - ref_score)
        print(f"Reference score (from filename): {ref_score:.6f}")
        print(f"Absolute difference: {diff:.6f}")
            # Both images are JPEG saves from the pipeline (cv2.imwrite .jpg), so pixel
        # values differ from the original in-memory tensors by ±1-5/255 per channel.
        # This propagates to a score error of ~3e-4 to 5e-4. The filename has 4
        # decimal places (precision 1e-4), so the combined tolerance is ~5e-4.
        if diff < 5e-4:
            print("✓ Scores match (within JPEG compression + 4-decimal filename precision)")
        else:
            print("✗ Scores differ — check masking flags match those used during Stage 4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute DINOv2 similarity between a query crop and a rendered template."
    )
    parser.add_argument("--query",         type=str, required=True,
                        help="Path to the query image (e.g. debug/query_raw/<frame>_obj0.jpg)")
    parser.add_argument("--template",      type=str, required=True,
                        help="Path to the template image (e.g. retrieved_templates/<frame>_obj0_rank0_*.jpg)")
    parser.add_argument("--query_mask",    type=str, default=None,
                        help="Path to the query binary mask (grayscale image, non-zero = foreground)")
    parser.add_argument("--template_mask", type=str, default=None,
                        help="Path to the template binary mask")
    parser.add_argument("--layer",         type=int, default=22,
                        help="DINOv2 layer to extract features from (default: 22)")

    # ── Masking flags — mirrors dino_inference_video_sam3d.py ─────────────────
    parser.add_argument("--mask_query",    action="store_true", default=True,
                        help="Zero out query background pixels before feature extraction (default: True). "
                             "Pass --no-mask_query to use the unmasked RGB crop.")
    parser.add_argument("--no-mask_query", dest="mask_query", action="store_false")

    parser.add_argument("--mask_template", action="store_true", default=False,
                        help="Zero out template background pixels before feature extraction (default: False).")
    parser.add_argument("--no-mask_template", dest="mask_template", action="store_false")

    parser.add_argument("--query_fg_patches", action="store_true", default=True,
                        help="Average similarity only over query foreground patches (default: True). "
                             "Pass --no-query_fg_patches to use all patches.")
    parser.add_argument("--no-query_fg_patches", dest="query_fg_patches", action="store_false")

    parser.add_argument("--template_fg_patches", action="store_true", default=False,
                        help="Average similarity only over template foreground patches (default: False). "
                             "Pass --template_fg_patches to enable.")
    parser.add_argument("--no-template_fg_patches", dest="template_fg_patches", action="store_false")

    args = parser.parse_args()
    main(args)