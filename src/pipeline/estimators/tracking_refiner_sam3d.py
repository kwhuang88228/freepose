"""
tracking_refiner_sam3d.py — SAM-3D-aware tracking refiner.

Replaces trimesh/pyrender mesh operations with Gaussian splat centroid projection:
  - _render:              projects Gaussian centroids → sparse depth mask
  - _crop_image:          uses centroid projections for bbox
  - _compute_3d_points:   uses Gaussian centroids instead of mesh.sample()
  - compute_2d3d_correspondences: uses a scaled copy of the splat for the interior mask
"""

import math
from collections import defaultdict
from copy import deepcopy

import cv2
import numpy as np
import torch
import tqdm
from loguru import logger
from PIL import Image

from src.pipeline import refiner_utils
from src.pipeline.retrieval.renderer_sam3d import load_gaussian, scale_gaussian


class TrackingRefinerSam3d:
    """Tracking refiner operating on Gaussian splat centroids instead of mesh geometry.

    All rendering-dependent steps are replaced by centroid projection, which
    avoids the need for a mesh renderer and OpenCV/OpenGL convention conversions.
    """

    def __init__(self, dino_model="dinov2_vitb14_reg", dino_device="cpu", cotracker_device="cpu"):
        self.dino_device       = dino_device
        self.cotracker_device  = cotracker_device
        self.dinov2    = torch.hub.load('facebookresearch/dinov2', dino_model).to(dino_device).eval()
        self.cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker2").to(cotracker_device).eval()

        self.image_size = int(math.sqrt(1370 - 1) * self.dinov2.patch_embed.proj.kernel_size[0])  # 518
        self.patch_size = self.dinov2.patch_embed.proj.kernel_size[0]   # 14
        self.feats_size = self.image_size // self.patch_size             # 37

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _project_centroids(self, gs, K: np.ndarray, transform: np.ndarray):
        """Project Gaussian centroids into the camera frame.

        Args:
            gs:        Gaussian splat.
            K:         3×3 camera intrinsics (pixel-space).
            transform: 4×4 TCO (object-to-camera, OpenCV).

        Returns:
            xyz_cam:  (N, 3) centroids in camera space.
            uv:       (N, 2) pixel coordinates (may be outside image bounds).
        """
        xyz = gs.get_xyz.detach().cpu().numpy()                       # (N, 3)
        xyz_h = np.concatenate([xyz, np.ones((len(xyz), 1))], axis=1).T  # (4, N)
        xyz_cam = (transform @ xyz_h)[:3].T                           # (N, 3)
        valid   = xyz_cam[:, 2] > 0
        proj    = (K @ xyz_cam[valid].T)                              # (3, M)
        uv_valid = (proj[:2] / proj[2:3]).T.astype(np.float32)        # (M, 2)

        # Full arrays (invalid points get uv=-1)
        uv = np.full((len(xyz), 2), -1.0, dtype=np.float32)
        uv[valid] = uv_valid
        return xyz_cam, uv, valid

    def _make_depth_mask(self, gs, width: int, height: int,
                         K: np.ndarray, transform: np.ndarray) -> tuple:
        """Create a pseudo-depth map by projecting Gaussian centroids.

        Returns:
            rgb:   (H, W, 3) uint8 — silhouette visualisation.
            depth: (H, W) float32 — z-depth at each pixel (0 where no centroid).
        """
        xyz_cam, uv, valid = self._project_centroids(gs, K, transform)
        rgb   = np.zeros((height, width, 3), dtype=np.uint8)
        depth = np.zeros((height, width), dtype=np.float32)

        uv_int   = uv[valid].astype(int)
        z_values = xyz_cam[valid, 2]
        in_bounds = (
            (uv_int[:, 0] >= 0) & (uv_int[:, 0] < width) &
            (uv_int[:, 1] >= 0) & (uv_int[:, 1] < height)
        )
        uv_ib = uv_int[in_bounds]
        z_ib  = z_values[in_bounds]

        depth[uv_ib[:, 1], uv_ib[:, 0]] = z_ib
        rgb[uv_ib[:, 1], uv_ib[:, 0]]   = [200, 120, 50]

        # Dilate sparse projection to get a filled silhouette
        kernel = np.ones((9, 9), np.uint8)
        depth  = cv2.dilate(depth,  kernel, iterations=2)
        rgb    = cv2.dilate(rgb,    kernel, iterations=2)

        return rgb, depth

    # Public interface (mirrors TrackingRefiner) ---------------------------

    def _render(self, gs, width: int, height: int, K: np.ndarray, transform: np.ndarray):
        """Return (rgb, depth) for the splat at the given pose."""
        return self._make_depth_mask(gs, width, height, K, transform)

    def _crop_image(self, gs, image, K: np.ndarray, transform: np.ndarray):
        """Crop image around the projected centroid bounding box."""
        # crop_image expects [N, 4] object-space homogeneous points; it projects
        # them internally via K @ T.  Use the raw 3-D centroids, not UV.
        xyz = gs.get_xyz.detach().cpu().numpy()          # (N, 3) object-space
        _, _, valid = self._project_centroids(gs, K, transform)
        xyz_valid = xyz[valid]

        # Subsample to keep memory / speed reasonable
        if len(xyz_valid) > 1000:
            idx = np.random.choice(len(xyz_valid), 1000, replace=False)
            xyz_valid = xyz_valid[idx]

        # Homogeneous 3-D object-space points [X, Y, Z, 1]
        pts = np.concatenate([xyz_valid, np.ones((len(xyz_valid), 1), dtype=np.float32)], axis=1)
        vertices = torch.from_numpy(pts).float()

        image_t = refiner_utils.MaybeToTensor()(image)
        K_t     = torch.from_numpy(K).view(3, 3).float()
        T_t     = torch.from_numpy(transform).view(1, 4, 4).float()

        cropped_images, recomputed_bboxes = refiner_utils.crop_image(
            image_t, T_t, vertices, K_t, 518, 518
        )
        new_Ks = refiner_utils.update_K_with_crop(K_t, recomputed_bboxes, 518, 518)
        return cropped_images[0], recomputed_bboxes[0], new_Ks[0]

    # ------------------------------------------------------------------
    def _get_threshold_for_confidence(self, similarity_matrices, top_quantile=0.2):
        counts, values = np.histogram(similarity_matrices[similarity_matrices > 0], bins=50)
        cutoff_value = counts.sum() * top_quantile
        cum_ = 0
        for c, v in zip(counts[::-1], values[:-1][::-1]):
            cum_ += c
            if cum_ > cutoff_value:
                break
        return v

    def pose_confidence(self, gs, photo, K, transform):
        cropped_photo, new_bbox, new_K = self._crop_image(gs, photo, K, transform)

        rendered_detection, rendered_depth = self._render(gs, 518, 518, new_K.numpy(), transform)
        rendered_detection     = Image.fromarray(rendered_detection)
        render_valid_37x37_mask = cv2.resize(
            (rendered_depth > 0).astype(np.float32), (37, 37), interpolation=cv2.INTER_CUBIC
        ) > 0.5

        with torch.no_grad():
            input_ = refiner_utils.pil2torch(cropped_photo).unsqueeze(0).to(self.dino_device)
            photo_feats = self.dinov2.forward_features(input_)["x_norm_patchtokens"].cpu()
            photo_feats = photo_feats.squeeze(0).view(37, 37, -1)
            photo_feats = photo_feats / torch.linalg.norm(photo_feats, dim=-1, keepdim=True)

            input_ = refiner_utils.pil2torch(rendered_detection).unsqueeze(0).to(self.dino_device)
            render_feats = self.dinov2.forward_features(input_)["x_norm_patchtokens"].cpu()
            render_feats = render_feats.squeeze(0).view(37, 37, -1)
            render_feats = render_feats / torch.linalg.norm(render_feats, dim=-1, keepdim=True)

        cosine_sim = (photo_feats * render_feats).sum(-1)
        cosine_sim = cosine_sim * torch.from_numpy(render_valid_37x37_mask).float()
        return cosine_sim.numpy()

    def n_inliers_per_pose(self, gs, frames, K, transforms):
        confidences = []
        for frame, transform in tqdm.tqdm(zip(frames, transforms)):
            confidences.append(
                self.pose_confidence(gs, Image.fromarray(frame), K, transform)
            )
        confidences = np.stack(confidences)
        thr = self._get_threshold_for_confidence(confidences)
        return (confidences > thr).sum(-1).sum(-1), thr

    # ------------------------------------------------------------------
    def _compute_3d_points(self, gs, render_valid_coords, K, transform):
        """Map image patch coordinates to 3D Gaussian centroids.

        Replaces mesh.sample(10000) with Gaussian centroid positions.
        """
        xyz = gs.get_xyz.detach().cpu().numpy()  # (N, 3)
        if len(xyz) > 10000:
            idx      = np.random.choice(len(xyz), 10000, replace=False)
            xyz = xyz[idx]

        # Project centroids to camera frame + image coords
        xyz_h   = np.pad(xyz, ((0, 0), (0, 1)), constant_values=1.0)  # (M, 4)
        xyz_cam = (xyz_h @ transform.T)[:, :3]                          # (M, 3)
        proj    = xyz_cam @ K.T                                         # (M, 3)
        proj_2d = proj[:, :2] / proj[:, 2:]                            # (M, 2)

        coords2indices = defaultdict(list)
        for i, p in enumerate(np.floor(proj_2d / self.patch_size).astype(np.int32)):
            coords2indices[tuple(p)].append(i)
        coords2indices = dict(coords2indices)

        render_real_coords = []
        for p in render_valid_coords:
            if tuple(p) not in coords2indices:
                logger.error(f"ERROR: patch {p} not in projected point cloud!")
                render_real_coords.append(np.array([0.0, 0.0, 0.0]))
            else:
                indices = np.array(coords2indices[tuple(p)])
                local_proj = proj_2d[indices] / self.patch_size
                closest = np.argsort(
                    np.square(local_proj - np.floor(local_proj) - 0.5).sum(1)
                )[:max(1, int(math.ceil(len(local_proj) * 0.25)))]
                min_z_idx = np.argmin(xyz_cam[indices[closest], 2])
                render_real_coords.append(xyz[indices[closest[min_z_idx]]])

        return np.stack(render_real_coords)

    def compute_2d3d_correspondences(self, gs, photo, K, transform, mask=None):
        cropped_photo, new_bbox, new_K = self._crop_image(gs, photo, K, transform)
        if mask is not None:
            cropped_mask, _, _ = self._crop_image(
                gs, mask[:, :, None].astype(np.float32), K, transform
            )
            cropped_mask = cv2.resize(
                cropped_mask[0].numpy(), (37, 37), interpolation=cv2.INTER_CUBIC
            ) > 0.5

        # Slightly shrunk splat for interior-only valid-mask
        gs_smaller = deepcopy(gs)
        gs_smaller.from_xyz(gs_smaller.get_xyz * 0.8)
        gs_smaller.from_scaling(gs_smaller.get_scaling * 0.8)

        _, rendered_depth = self._render(gs_smaller, 518, 518, new_K.numpy(), transform)
        render_valid_37x37_mask = cv2.resize(
            (rendered_depth > 0).astype(np.float32), (37, 37), interpolation=cv2.INTER_CUBIC
        ) > 0.5

        if mask is None:
            image_valid_coords = np.stack(np.where(render_valid_37x37_mask)[::-1], 1)
        else:
            image_valid_coords = np.stack(
                np.where(render_valid_37x37_mask & cropped_mask)[::-1], 1
            )
            if len(image_valid_coords) < 4:
                logger.warning("Not enough valid points in mask, using unmasked points.")
                image_valid_coords = np.stack(np.where(render_valid_37x37_mask)[::-1], 1)

        render_real_coords = self._compute_3d_points(
            gs, image_valid_coords, new_K.numpy(), transform
        )

        # Project the selected 3D centroids back to the ORIGINAL image frame to
        # get their exact 2D positions as CoTracker starting queries.
        #
        # The alternative — mapping patch centres through the crop bbox — places
        # the query up to half a patch width (≈7 px in the 518-px crop) away from
        # where the centroid actually projects.  That systematic offset biases the
        # PnP solution for every frame because CoTracker tracks the PATCH-CENTRE
        # feature, not the centroid, so the 2D–3D pairing is misaligned from the
        # start.  Using the exact projection removes this bias.
        K_np = K.numpy().astype(np.float64) if hasattr(K, "numpy") else np.asarray(K, dtype=np.float64)
        T_np = np.asarray(transform, dtype=np.float64)
        xyz_h = np.concatenate(
            [render_real_coords.astype(np.float64),
             np.ones((len(render_real_coords), 1), dtype=np.float64)], axis=1
        ).T  # (4, N)
        xyz_cam = (T_np @ xyz_h)[:3].T  # (N, 3) in camera space
        in_front = xyz_cam[:, 2] > 0
        if in_front.sum() >= 4:
            proj = (K_np @ xyz_cam[in_front].T)          # (3, M)
            tracking_query_points = (proj[:2] / proj[2]).T.astype(np.float32)  # (M, 2)
            render_real_coords     = render_real_coords[in_front]
        else:
            # Degenerate case: fall back to patch-centre mapping.
            logger.warning(
                f"Only {in_front.sum()} correspondences project in front of the camera "
                "— falling back to patch-centre query positions."
            )
            x1, y1, x2, y2 = new_bbox.numpy()
            tracking_query_points = (
                np.float32(image_valid_coords) * self.patch_size + self.patch_size * 0.5
            )
            tracking_query_points = (
                tracking_query_points / self.image_size * np.array([[x2 - x1, y2 - y1]])
                + np.array([[x1, y1]])
            )

        return tracking_query_points, render_real_coords

    def _track_frames(self, frames, query_points):
        video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(self.cotracker_device)
        with torch.no_grad():
            pred_tracks, pred_visibility = self.cotracker(
                video,
                queries=torch.from_numpy(query_points)[None].to(self.cotracker_device),
                backward_tracking=True,
            )
        return pred_tracks.squeeze(0).cpu().numpy(), pred_visibility.squeeze(0).cpu().numpy()
