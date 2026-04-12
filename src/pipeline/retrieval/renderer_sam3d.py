"""
renderer_sam3d.py — Gaussian-splat-based template renderer (replaces MeshRenderer).

Uses SAM-3D-Objects' GaussianRenderer to pre-render a Gaussian splat at N
Hammersley-sampled viewpoints, producing (RGB, depth, extrinsic) tuples in the
same format expected by downstream pose estimators.
"""

import math
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from src.utils.bbox_utils import CropResizePad

# ── SAM-3D paths ──────────────────────────────────────────────────────────────
_FREEPOSE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_SAM3D_ROOT = _FREEPOSE_ROOT / "sam-3d-objects"
sys.path.insert(0, str(_SAM3D_ROOT / "notebook"))
sys.path.insert(0, str(_SAM3D_ROOT))

from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian.gaussian_model import Gaussian
from sam3d_objects.model.backbone.tdfy_dit.utils.render_utils import (
    render_frames,
    yaw_pitch_r_fov_to_extrinsics_intrinsics,
)
from sam3d_objects.model.backbone.tdfy_dit.utils.random_utils import sphere_hammersley_sequence

# ── SAM-3D rendering constants (must match the training camera convention) ────
SAM3D_RESOLUTION = 420
SAM3D_FOV_DEG    = 40
SAM3D_R          = 2.0
SAM3D_NEAR       = 0.8
SAM3D_FAR        = 1.6

# Pixel-space intrinsic matrix for SAM-3D rendering camera at 420×420, fov=40°
_fov_rad = math.radians(SAM3D_FOV_DEG)
_f = SAM3D_RESOLUTION / (2.0 * math.tan(_fov_rad / 2.0))
K_SAM3D = np.array(
    [[_f, 0.0, SAM3D_RESOLUTION / 2.0],
     [0.0, _f, SAM3D_RESOLUTION / 2.0],
     [0.0, 0.0, 1.0]],
    dtype=np.float64,
)

# Fixed aabb used by SAM-3D's GS decoder
_SAM3D_AABB = [-0.5, -0.5, -0.5, 1.0, 1.0, 1.0]


# ── Gaussian helper functions ─────────────────────────────────────────────────

def load_gaussian(ply_path: str | Path, device: str = "cuda") -> Gaussian:
    """Load a saved SAM-3D Gaussian splat from a .ply file."""
    gs = Gaussian(aabb=_SAM3D_AABB, sh_degree=0, device=device)
    gs.load_ply(str(ply_path))
    return gs


def scale_gaussian(gs: Gaussian, scale: float) -> Gaussian:
    """Return a new Gaussian splat uniformly scaled by *scale* (positions + kernel sizes)."""
    gs_copy = deepcopy(gs)
    gs_copy.from_xyz(gs_copy.get_xyz * scale)
    gs_copy.from_scaling(gs_copy.get_scaling * scale)
    return gs_copy


def percent_depth_to_metric(percent_depth: np.ndarray) -> np.ndarray:
    """Convert SAM-3D percent-depth [0,1] to metric depth in camera-frame units."""
    return SAM3D_NEAR + percent_depth * (SAM3D_FAR - SAM3D_NEAR)


def extrinsic_to_tcoinit(extrinsic: torch.Tensor) -> np.ndarray:
    """Convert a SAM-3D world-to-camera extrinsic tensor to a 4×4 numpy TCO_init matrix.

    The object is at the world origin, so the extrinsic directly gives TCO:
        p_cam = R @ p_world + t  →  object (at 0,0,0) is at t in camera frame.
    """
    E = extrinsic.cpu().numpy().astype(np.float64)
    TCO = np.eye(4, dtype=np.float64)
    TCO[:3, :3] = E[:3, :3]
    TCO[:3, 3]  = E[:3, 3]
    return TCO


# ── SplatRenderer ─────────────────────────────────────────────────────────────

class SplatRenderer:
    """Renders a SAM-3D Gaussian splat at Hammersley-sampled viewpoints.

    Mirrors the interface of MeshRenderer so downstream estimators can swap it in
    with minimal changes.

    Args:
        n_poses:    Number of Hammersley viewpoints to pre-compute.
        resolution: Rendered image resolution (square).
    """

    def __init__(self, n_poses: int = 600, resolution: int = SAM3D_RESOLUTION):
        self.n_poses    = n_poses
        self.resolution = resolution

        # Pre-compute Hammersley camera parameters (yaw, pitch)
        cams = [sphere_hammersley_sequence(i, n_poses) for i in range(n_poses)]
        self._yaws   = [c[0] for c in cams]
        self._pitchs = [c[1] for c in cams]

        # Pre-compute extrinsics (world-to-camera) and store as numpy 4×4 arrays
        extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(
            self._yaws, self._pitchs, rs=SAM3D_R, fovs=SAM3D_FOV_DEG
        )
        self._extrinsics = extrinsics   # list of (4,4) tensors
        self._intrinsics = intrinsics   # list of (3,3) tensors (normalised)

        # Rotation matrices (for geodesic distance computation)
        self.rotations = np.stack([
            e.cpu().numpy()[:3, :3] for e in self._extrinsics
        ])  # (N, 3, 3)

        # 4×4 TCO_init matrices used by get_z_from_pointcloud
        self.tcoinits = np.stack([
            extrinsic_to_tcoinit(e) for e in self._extrinsics
        ])  # (N, 4, 4)

    # ------------------------------------------------------------------
    def _render_at_indices(self, gs: Gaussian, indices: list[int]) -> list[tuple]:
        """Render the splat at the given pose indices.

        Returns:
            List of (rgb: ndarray HxWx3 uint8, depth_metric: ndarray HxW float32,
                     extrinsic_4x4: ndarray 4×4 float64).
        """
        extrinsics = [self._extrinsics[i] for i in indices]
        intrinsics = [self._intrinsics[i] for i in indices]

        res = render_frames(
            gs,
            extrinsics,
            intrinsics,
            options={
                "resolution": self.resolution,
                "near":       SAM3D_NEAR,
                "far":        SAM3D_FAR,
                "bg_color":   (0, 0, 0),
                "backend":    "gsplat",
            },
            verbose=False,
        )

        results = []
        for k, i in enumerate(indices):
            rgb   = res["color"][k]   # HxWx3 uint8
            pdep  = res["depth"][k]   # HxW float32 — camera-space z in SAM3D world units,
                                      # or None if the backend returned no depth
            if pdep is not None:
                # gsplat returns camera-space z directly (not percent depth);
                # use as-is so depthmap_to_pointcloud backprojects correctly.
                depth_metric = pdep.astype(np.float32)
            else:
                depth_metric = np.zeros((self.resolution, self.resolution), dtype=np.float32)
            results.append((rgb, depth_metric, self.tcoinits[i]))
        return results

    # ------------------------------------------------------------------
    def render(self, gs: Gaussian) -> list[tuple]:
        """Render all n_poses views of *gs*.

        Returns:
            List of (rgb, depth_metric, tcoinit_4x4) for each pose.
        """
        return self._render_at_indices(gs, list(range(self.n_poses)))

    def render_from_poses(self, gs: Gaussian, pose_indices: list[int]) -> list[tuple]:
        """Render *gs* at the specified pose indices.

        Args:
            gs:           Gaussian splat to render.
            pose_indices: Indices into self._extrinsics to render at.

        Returns:
            List of (rgb, depth_metric, tcoinit_4x4).
        """
        return self._render_at_indices(gs, list(pose_indices))

    # ------------------------------------------------------------------
    @staticmethod
    def mask_to_bbox(mask: np.ndarray) -> np.ndarray:
        y_indices, x_indices = np.nonzero(mask)
        return np.array([x_indices.min(), y_indices.min(), x_indices.max(), y_indices.max()])

    @staticmethod
    def generate_proposals(
        res: list[tuple],
        resolution: int = SAM3D_RESOLUTION,
        bbox_extend: float = 0,
    ) -> tuple:
        """Convert render results to cropped template tensors.

        Mirrors MeshRenderer.generate_proposals().

        Returns:
            (templates_cropped: Tensor [N, 3, H, W],
             tcoinits: list of ndarray 4×4,
             masks: list of ndarray bool HxW)
        """
        templates, boxes, tcoinits, masks = [], [], [], []
        rgb_proposal_processor = CropResizePad(resolution, (resolution, resolution), bbox_extend=bbox_extend)

        for rgb, depth, tcoinit in res:
            mask = depth > 0

            if mask.sum() < 100:
                mask[105:315, 105:315] = True

            bbox = SplatRenderer.mask_to_bbox(mask)

            image = torch.from_numpy(rgb / 255.0).float()
            templates.append(image)
            boxes.append(bbox)
            tcoinits.append(tcoinit)
            masks.append(mask)

        templates_t = torch.stack(templates).permute(0, 3, 1, 2)   # (N, 3, H, W)
        boxes_t     = torch.tensor(np.array(boxes))
        templates_cropped = rgb_proposal_processor(templates_t, boxes_t)

        # Crop masks with the same transform so their spatial layout matches
        # templates_cropped (i.e. what DINOv2 will actually see).
        masks_t = torch.from_numpy(np.stack(masks).astype(np.float32)).unsqueeze(1)  # (N, 1, H, W)
        cropped_masks_t = rgb_proposal_processor(masks_t, boxes_t)                   # (N, 1, H, W)
        cropped_masks = [(cropped_masks_t[i, 0].numpy() > 0.5) for i in range(len(masks))]

        return templates_cropped, tcoinits, cropped_masks
