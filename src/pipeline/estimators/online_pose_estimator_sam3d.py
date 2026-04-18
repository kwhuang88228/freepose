"""
online_pose_estimator_sam3d.py — SAM-3D-aware 6D pose estimators.

Replaces MeshRenderer / trimesh calls with SplatRenderer / Gaussian splat.
Key differences from online_pose_estimator.py:
  - No mesh.apply_scale(): splat is already unit-normalised by SAM-3D.
  - Depth from render is SAM-3D percent-depth → metric; no /0.25 rescale.
  - TCO_init comes from SAM-3D extrinsics (world-to-camera 4×4).
  - template_dict must contain 'intrinsic' = K_SAM3D (pixel-space).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum
from scipy.spatial.transform import Rotation as Rot

from src.pipeline.estimators.pose_estimator import DinoPoseEstimator
from src.pipeline.retrieval.dino import DINOv2FeatureExtractor
from src.pipeline.retrieval.renderer_sam3d import SplatRenderer, K_SAM3D, load_gaussian
from src.pipeline.utils import depthmap_to_pointcloud, get_z_from_pointcloud

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DinoPoseEstimatorSam3d(DinoPoseEstimator):
    """Coarse pose estimator using SAM-3D rendered templates.

    Differences from DinoPoseEstimator:
      - self.mesh_poses is replaced by SAM-3D TCO_init matrices.
      - No /0.25 rescale of point clouds (splat is unit-normalised).
      - depthmap_to_pointcloud uses K_SAM3D (rendering camera K).
    """

    def __init__(self, n_poses: int = 600, sam3d_tcoinits: list = None, **kwargs):
        super().__init__(n_poses=n_poses, **kwargs)
        if sam3d_tcoinits is not None:
            self.mesh_poses = sam3d_tcoinits  # override with SAM-3D extrinsics

    def forward(self, proposal, template_dict, K, bbox, est_scale,
                layer=22, batch_size=128, return_query_feat=False, proposal_mask=None,
                use_query_fg_patches=True, use_template_fg_patches=False):
        """Same flow as DinoPoseEstimator.forward but adapted for SAM-3D depth."""
        if self.cache_size > 0:
            feats_template = self._get_template_features(template_dict, layer=layer, batch_size=batch_size)
        else:
            feats_template = self._extract_features(template_dict['templates'], layer=layer, batch_size=batch_size)

        query_feat = self.feature_extractor(
            proposal[None].to('cuda', dtype=torch.bfloat16), layer=layer, feature_type='patch'
        )
        signature = 'b n d, b n d -> b n'

        # Per-patch cosine similarities: [N_templates, num_patches]
        per_patch_sim = einsum(
            F.normalize(feats_template, dim=-1),
            F.normalize(query_feat, dim=-1),
            signature,
        )

        # Build effective patch mask from enabled options (AND if both active).
        N = per_patch_sim.shape[0]
        effective_mask = None

        if use_query_fg_patches and proposal_mask is not None:
            q_patch_mask = self._to_patch_mask(proposal_mask)                      # [num_patches]
            q_patch_mask = q_patch_mask.to(per_patch_sim.device)
            effective_mask = q_patch_mask.unsqueeze(0).expand(N, -1).clone()       # [N, num_patches]

        if use_template_fg_patches and 'patch_masks' in template_dict:
            tmpl_masks = template_dict['patch_masks'].to(per_patch_sim.device)     # [N, num_patches]
            effective_mask = (effective_mask & tmpl_masks) if effective_mask is not None else tmpl_masks

        if effective_mask is not None:
            count = effective_mask.float().sum(dim=-1).clamp(min=1)                # [N]
            scores = (per_patch_sim * effective_mask.float()).sum(dim=-1) / count
        else:
            scores = per_patch_sim.mean(dim=-1)

        top_scores, top_indices = torch.topk(scores, 5)
        top_scores  = top_scores.float().cpu().numpy()
        top_indices = top_indices.cpu().numpy()

        out_dict = {
            'TCO': [],
            'scores': top_scores,
            'proposal': proposal,
            'K': K,
            'bbox': bbox,
            'retrieved_proposals':    [template_dict['templates'][idx] for idx in top_indices],
            'retrieved_template_ids': top_indices,  # (5,) int, indices 0..599 matching templates_cropped/
        }

        K_render = template_dict['intrinsic'].numpy()  # pixel-space K of SAM-3D render
        for idx in top_indices:
            depth = template_dict['depths'][idx].numpy()  # already metric (converted by SplatRenderer)
            point_cloud = depthmap_to_pointcloud(depth, K_render)
            # No /0.25 rescale — splat is unit-normalised, not mesh-scaled.
            point_cloud *= est_scale
            TCO = get_z_from_pointcloud(bbox, point_cloud, K, self.mesh_poses[idx])
            out_dict['TCO'].append(TCO)

        if return_query_feat:
            out_dict['query_feat'] = query_feat

        return out_dict


class DinoOnlinePoseEstimatorSam3d(nn.Module):
    """Fine-pose estimator using SAM-3D Gaussian splat rendering.

    Mirrors DinoOnlinePoseEstimator but:
      - Uses SplatRenderer (20 000 fine Hammersley poses, SAM-3D camera).
      - No mesh.apply_scale(); passes Gaussian splat directly.
      - Depth handling adjusted for SAM-3D percent-depth → metric.
    """

    def __init__(self, n_coarse_poses=600, n_fine_poses=20000,
                 cache_size=50, save_all=False, cache_dir='./data/cache'):
        super().__init__()
        self.coarse_estimator = DinoPoseEstimatorSam3d(
            n_poses=n_coarse_poses, cache_size=cache_size,
            save_all=save_all, cache_dir=cache_dir,
        )
        self.feature_extractor = DINOv2FeatureExtractor().to('cuda', dtype=torch.bfloat16)
        # Fine renderer: 20 000 Hammersley poses; extrinsics pre-computed but renders on demand.
        self.fine_renderer = SplatRenderer(n_poses=n_fine_poses)
        # Copy SAM-3D TCO_inits into coarse estimator so coarse forward uses them.
        self.coarse_estimator.mesh_poses = list(self.fine_renderer.tcoinits[:n_coarse_poses])

    # ------------------------------------------------------------------
    @staticmethod
    def geodesic_distance(render_rots: np.ndarray, query_pose: np.ndarray, degrees=True):
        """Angular distance between each rotation in render_rots and query_pose."""
        query_rot = query_pose[:3, :3]
        diffs = render_rots @ query_rot.T
        dists = np.linalg.norm(Rot.from_matrix(diffs).as_rotvec(), axis=1)
        if degrees:
            dists = np.rad2deg(dists)
        return dists

    # ------------------------------------------------------------------
    def forward(self, proposal, proposal_mask, template_dict, gs, K, bbox,
                est_scale, prev_pose=None, neighborhood=15, layer=22, batch_size=128):
        if prev_pose is None:
            coarse_results = self.coarse_estimator.forward(
                proposal, template_dict, K, bbox, est_scale, layer, batch_size,
                return_query_feat=True, proposal_mask=proposal_mask,
            )
            query_feat = coarse_results['query_feat']
            prev_pose  = coarse_results['TCO'][0]
        else:
            query_feat = None

        return self.forward_fine(
            proposal, proposal_mask, template_dict, gs, K, bbox,
            est_scale, prev_pose, neighborhood, layer, query_feat,
        )

    def forward_fine(self, proposal, proposal_mask, template_dict, gs, K, bbox,
                     est_scale, prev_pose, neighborhood=15, layer=22, query_feat=None):
        if query_feat is None:
            query_feat = self.feature_extractor(
                proposal[None].cuda().half(), layer=layer, feature_type='patch'
            )
            query_feat = F.normalize(query_feat, dim=-1)

        # Find fine poses close to the current estimate
        dists      = self.geodesic_distance(self.fine_renderer.rotations, prev_pose)
        close_idxs = np.where(dists < neighborhood)[0]

        # Render fine poses on demand — no scale applied (splat is unit-normalised)
        renders = self.fine_renderer.render_from_poses(gs, close_idxs.tolist())
        ren_props, tcoinits_fine, masks_fine = self.fine_renderer.generate_proposals(renders)

        feats_fine = self.feature_extractor(
            ren_props.cuda().half(), layer=layer, feature_type='patch'
        )

        signature = 'b n d, b n d -> b n'
        per_patch_sim = einsum(query_feat, F.normalize(feats_fine, dim=-1), signature)

        # Masked mean over query foreground patches only (constant denominator across templates).
        q_patch_mask = DinoPoseEstimator._to_patch_mask(proposal_mask)              # [num_patches]
        q_patch_mask = q_patch_mask.to(per_patch_sim.device).unsqueeze(0)           # [1, num_patches]
        count = q_patch_mask.float().sum().clamp(min=1)
        scores = (per_patch_sim * q_patch_mask.float()).sum(dim=-1) / count

        k = min(5, scores.shape[0])
        top_scores, top_indices = torch.topk(scores, k)
        top_index = top_indices[0].item()
        top_score = top_scores[0]

        _, depth_metric, tcoinit = renders[top_index]
        K_render = template_dict['intrinsic'].numpy()
        point_cloud = depthmap_to_pointcloud(depth_metric, K_render)
        # No /0.25 rescale — splat is already unit-normalised.
        point_cloud *= est_scale
        TCO = get_z_from_pointcloud(bbox, point_cloud, K, tcoinit)

        return {
            'TCO':                [TCO],
            'scores':             [top_score.float().cpu().numpy()],
            'proposal':           proposal,
            'K':                  K,
            'bbox':               bbox,
            'retrieved_templates': ren_props[top_indices.cpu()],  # [5, 3, H, W] float 0-1
        }
