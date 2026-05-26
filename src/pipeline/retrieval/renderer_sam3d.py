"""
renderer_sam3d.py — Gaussian-splat-based template renderer (replaces MeshRenderer).

Uses SAM-3D-Objects' GaussianRenderer to pre-render a Gaussian splat at N
Hopf-Hammersley-sampled SO(3) poses (yaw, pitch, roll drawn jointly from one
low-discrepancy sequence), producing (RGB, depth, extrinsic) tuples in the
same format expected by downstream pose estimators.
"""

import math
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

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
from sam3d_objects.model.backbone.tdfy_dit.utils.random_utils import radical_inverse

# ── SAM-3D rendering constants (must match the training camera convention) ────
SAM3D_RESOLUTION = 512
SAM3D_FOV_DEG    = 40
SAM3D_R          = 2.0
SAM3D_NEAR       = 0.8
SAM3D_FAR        = 1.6

# Pixel-space intrinsic matrix for SAM-3D rendering camera at 512×512, fov=40°
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


# ── Hopf-Hammersley SO(3) sampler ─────────────────────────────────────────────

def hopf_hammersley_sequence(n: int, num_samples: int) -> tuple[float, float, float]:
    """Joint Hopf-Hammersley sample on SO(3): returns (yaw, pitch, roll).

    Coords come from one low-discrepancy sequence (n/N, Φ_2(n), Φ_3(n)) routed
    through the Hopf parametrization, so the (yaw, pitch, roll) triples do NOT
    factor as a product of an S² sequence and an S¹ sequence. On the first two
    coords this matches sphere_hammersley_sequence's (yaw, pitch); the third
    coord is the Hopf fiber angle (in-plane camera rotation).
    """
    u = n / num_samples
    v = radical_inverse(2, n)
    w = radical_inverse(3, n)
    pitch = math.acos(1.0 - 2.0 * u) - math.pi / 2.0   # elevation, [-π/2, π/2]
    yaw   = v * 2.0 * math.pi                          # azimuth, [0, 2π)
    roll  = w * 2.0 * math.pi                          # fiber, [0, 2π)
    return yaw, pitch, roll


# ── SplatRenderer ─────────────────────────────────────────────────────────────

class SplatRenderer:
    """Renders a SAM-3D Gaussian splat at Hopf-Hammersley-sampled SO(3) poses.

    Mirrors the interface of MeshRenderer so downstream estimators can swap it in
    with minimal changes.

    Args:
        n_views:    Number of Hopf-Hammersley SO(3) samples to pre-compute.
        resolution: Rendered image resolution (square).
    """

    def __init__(self, n_views: int = 600, resolution: int = SAM3D_RESOLUTION):
        self.n_views    = n_views
        self.resolution = resolution

        # Joint Hopf-Hammersley (yaw, pitch, roll) — one triple per view, not a
        # product of an S² sequence and an S¹ sequence.
        self._yaws, self._pitchs, self._rolls = [], [], []
        for i in range(self.n_views):
            yaw, pitch, roll = hopf_hammersley_sequence(i, self.n_views)
            self._yaws.append(yaw)
            self._pitchs.append(pitch)
            self._rolls.append(roll)

        base_extrinsics, base_intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(
            self._yaws, self._pitchs, rs=SAM3D_R, fovs=SAM3D_FOV_DEG
        )

        # Apply the Hopf fiber angle as an in-plane rotation about the camera
        # optical axis. E_new = Rz(ψ) @ E_base rotates the camera frame about its
        # own +Z; since principal point is centered and fx=fy, this is equivalent
        # to rotating the rendered image about its center by ψ.
        self._extrinsics, self._intrinsics = [], []
        for i in range(self.n_views):
            E_i  = base_extrinsics[i]
            K_i  = base_intrinsics[i]
            psi  = self._rolls[i]
            c, s = math.cos(psi), math.sin(psi)
            R_roll = torch.eye(4, dtype=E_i.dtype, device=E_i.device)
            R_roll[0, 0] = c;  R_roll[0, 1] = -s
            R_roll[1, 0] = s;  R_roll[1, 1] =  c
            self._extrinsics.append(R_roll @ E_i)
            self._intrinsics.append(K_i)   # K unchanged by roll about optical axis

        # Rotation matrices (for geodesic distance computation)
        self.rotations = np.stack([
            e.cpu().numpy()[:3, :3] for e in self._extrinsics
        ])  # (n_views, 3, 3)

        # 4×4 TCO_init matrices used by get_z_from_pointcloud
        self.tcoinits = np.stack([
            extrinsic_to_tcoinit(e) for e in self._extrinsics
        ])  # (n_views, 4, 4)

        # Unit-sphere viewing direction per sample (Hopf base point on S²).
        self._xyz = np.array([
            [math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)]
            for y, p in zip(self._yaws, self._pitchs)
        ])  # (n_views, 3)

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
            results.append((rgb, depth_metric, self.tcoinits[i], self._xyz[i]))
        return results

    # ------------------------------------------------------------------
    def render(self, gs: Gaussian) -> list[tuple]:
        """Render all n_views of *gs*.

        Returns:
            List of (rgb, depth_metric, tcoinit_4x4) for each pose.
        """
        return self._render_at_indices(gs, list(range(self.n_views)))

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
    def _save_debug_images(tensor: torch.Tensor, subdir: Path, is_mask: bool = False) -> None:
        """Save each frame of a (N, C, H, W) tensor as a JPEG in *subdir*."""
        subdir.mkdir(parents=True, exist_ok=True)
        arr = tensor.detach().cpu().numpy()          # (N, C, H, W), float 0-1
        for i, frame in enumerate(arr):
            if is_mask:
                # (1, H, W) → (H, W), scale to [0, 255]
                img_np = (frame[0] * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(img_np, mode="L").save(subdir / f"{i:04d}.jpg")
            else:
                # (3, H, W) → (H, W, 3), scale to [0, 255]
                img_np = (frame.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(img_np, mode="RGB").save(subdir / f"{i:04d}.jpg")

    @staticmethod
    def generate_proposals(
        res: list[tuple],
        resolution: int = SAM3D_RESOLUTION,
        bbox_extend: float = 0,
        debug_dir: "str | Path | None" = None,
    ) -> tuple:
        """Convert render results to cropped template tensors.

        Mirrors MeshRenderer.generate_proposals().

        Args:
            debug_dir: If given, each intermediate variable from the crop/mask
                       pipeline is saved as JPEG images under a named subdirectory
                       of *debug_dir* (e.g. debug_dir/templates_t/0000.jpg).
                       boxes_t is saved as boxes_t/boxes_t.txt.

        Returns:
            (templates_cropped: Tensor [N, 3, H, W],
             tcoinits: list of ndarray 4×4,
             masks: list of ndarray bool HxW)
        """
        templates, boxes, tcoinits, masks, xyzs = [], [], [], [], []
        rgb_proposal_processor = CropResizePad(resolution, (resolution, resolution), bbox_extend=bbox_extend)

        for rgb, depth, tcoinit, xyz in res:
            mask = depth > 0

            if mask.sum() < 100:
                mask[128:384, 128:384] = True

            bbox = SplatRenderer.mask_to_bbox(mask)

            image = torch.from_numpy(rgb / 255.0).float()
            templates.append(image)
            boxes.append(bbox)
            tcoinits.append(tcoinit)
            masks.append(mask)
            xyzs.append(xyz)

        templates_t = torch.stack(templates).permute(0, 3, 1, 2)   # (N, 3, H, W)
        boxes_t     = torch.tensor(np.array(boxes))
        templates_cropped = rgb_proposal_processor(templates_t, boxes_t)   # this maximizes the object's pixel area in the cropped template, which is what DINOv2 will see.

        # Crop masks with the same transform so their spatial layout matches
        # templates_cropped (i.e. what DINOv2 will actually see).
        masks_t = torch.from_numpy(np.stack(masks).astype(np.float32)).unsqueeze(1)  # (N, 1, H, W)
        masks_cropped_t = rgb_proposal_processor(masks_t, boxes_t)                   # (N, 1, H, W)
        masks_cropped = [(masks_cropped_t[i, 0].numpy() > 0.5) for i in range(len(masks))]

        # ── Debug dumps ───────────────────────────────────────────────────────
        if debug_dir is not None:
            _dbg = Path(debug_dir)
            # RGB tensors with bounding boxes overlaid → JPEG per frame
            _tmpl_dir = _dbg / "templates_t"
            _tmpl_dir.mkdir(parents=True, exist_ok=True)
            _tmpl_arr = templates_t.detach().cpu().numpy()   # (N, 3, H, W), float 0-1
            _boxes_np = boxes_t.numpy()                      # (N, 4): x0 y0 x1 y1
            for _i, _frame in enumerate(_tmpl_arr):
                _img = Image.fromarray(
                    (_frame.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8), mode="RGB"
                )
                _draw = ImageDraw.Draw(_img)
                x0, y0, x1, y1 = _boxes_np[_i]
                _draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
                _hx, _hy, _hz = xyzs[_i]
                _img.save(_tmpl_dir / f"{_i:04d}_{_hx:.3f}_{_hy:.3f}_{_hz:.3f}.jpg")
            SplatRenderer._save_debug_images(templates_cropped, _dbg / "templates_cropped")
            # Mask tensors with bounding boxes overlaid → grayscale JPEG per frame
            _mask_dir = _dbg / "masks_t"
            _mask_dir.mkdir(parents=True, exist_ok=True)
            _masks_arr = masks_t.detach().cpu().numpy()      # (N, 1, H, W), float 0-1
            for _i, _mframe in enumerate(_masks_arr):
                _mimg = Image.fromarray(
                    (_mframe[0] * 255).clip(0, 255).astype(np.uint8), mode="L"
                ).convert("RGB")
                _draw = ImageDraw.Draw(_mimg)
                x0, y0, x1, y1 = _boxes_np[_i]
                _draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
                _mimg.save(_mask_dir / f"{_i:04d}.jpg")
            SplatRenderer._save_debug_images(masks_cropped_t,  _dbg / "masks_cropped_t", is_mask=True)
            # # Bounding boxes → plain text (one row per frame: x0 y0 x1 y1)
            # _boxes_dir = _dbg / "boxes_t"
            # _boxes_dir.mkdir(parents=True, exist_ok=True)
            # np.savetxt(_boxes_dir / "boxes_t.txt", boxes_t.numpy(), fmt="%.2f",
            #            header="x0 y0 x1 y1")
            # masks_cropped (bool list) → grayscale JPEG per frame
            _cm_dir = _dbg / "masks_cropped"
            _cm_dir.mkdir(parents=True, exist_ok=True)
            for i, cm in enumerate(masks_cropped):
                Image.fromarray((cm * 255).astype(np.uint8), mode="L").save(_cm_dir / f"{i:04d}.jpg")

        return templates_cropped, tcoinits, masks_cropped
