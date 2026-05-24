#!/usr/bin/env python3
"""
Render 5 random views of a GLB file into a single PNG.
Uses PyTorch3D for offscreen rendering (no display required).

Usage:
    python view_glb.py result.glb
    python view_glb.py result.glb --output views.png --resolution 512
"""

import argparse
import sys
import numpy as np
import torch
import trimesh
from PIL import Image
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    look_at_view_transform,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardFlatShader,
    TexturesVertex,
    AmbientLights,
)


def load_glb_as_pytorch3d(glb_path: str, device: torch.device) -> Meshes:
    scene = trimesh.load(glb_path)
    if isinstance(scene, trimesh.Scene):
        mesh = scene.to_geometry()
    else:
        mesh = scene

    verts = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int64)
    colors_rgba = np.array(mesh.visual.vertex_colors, dtype=np.float32) / 255.0
    colors_rgb = colors_rgba[:, :3]

    # Center and normalise to unit sphere
    centroid = verts.mean(axis=0)
    verts -= centroid
    scale = np.linalg.norm(verts, axis=1).max()
    verts /= scale

    # OpenGL → PyTorch3D: flip X and Z
    verts[:, 0] *= -1
    verts[:, 2] *= -1

    verts_t = torch.from_numpy(verts).to(device)
    faces_t = torch.from_numpy(faces).to(device)
    colors_t = torch.from_numpy(colors_rgb).to(device)

    textures = TexturesVertex(verts_features=colors_t[None])
    return Meshes(verts=[verts_t], faces=[faces_t], textures=textures)


def make_renderer(device: torch.device, R: torch.Tensor, T: torch.Tensor, resolution: int) -> MeshRenderer:
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=45)
    raster_settings = RasterizationSettings(
        image_size=resolution,
        blur_radius=0.0,
        faces_per_pixel=1,
        cull_backfaces=False,
    )
    lights = AmbientLights(device=device)
    return MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=HardFlatShader(device=device, cameras=cameras, lights=lights),
    )


def random_views(n: int, dist: float = 2.5, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    azim = rng.uniform(0, 360, n)
    elev = rng.uniform(-30, 60, n)
    return elev, azim


def render_views(mesh: Meshes, elev: np.ndarray, azim: np.ndarray, resolution: int, dist: float = 2.5) -> list[np.ndarray]:
    device = mesh.device
    images = []
    for e, a in zip(elev, azim):
        R, T = look_at_view_transform(dist=dist, elev=float(e), azim=float(a), device=device)
        renderer = make_renderer(device, R, T, resolution)
        with torch.no_grad():
            img = renderer(mesh)[0, ..., :3].cpu().numpy()  # (H, W, 3) float [0,1]
        images.append((np.clip(img, 0, 1) * 255).astype(np.uint8))
    return images


def tile_images(images: list[np.ndarray], cols: int = 5) -> np.ndarray:
    rows = (len(images) + cols - 1) // cols
    h, w = images[0].shape[:2]
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = img
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Render 5 random views of a GLB file.")
    parser.add_argument("glb", help="Path to the .glb file")
    parser.add_argument("--output", default="views.png", help="Output PNG path (default: views.png)")
    parser.add_argument("--resolution", type=int, default=512, help="Render resolution per view (default: 512)")
    parser.add_argument("--n", type=int, default=5, help="Number of views (default: 5)")
    parser.add_argument("--dist", type=float, default=2.5, help="Camera distance (default: 2.5)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.glb} ...")
    mesh = load_glb_as_pytorch3d(args.glb, device)
    print(f"  {mesh.verts_packed().shape[0]} verts, {mesh.faces_packed().shape[0]} faces  device={device}")

    elev, azim = random_views(args.n, dist=args.dist, seed=args.seed)
    print(f"Rendering {args.n} views ...")
    for i, (e, a) in enumerate(zip(elev, azim)):
        print(f"  view {i+1}: elev={e:.1f}°  azim={a:.1f}°")

    images = render_views(mesh, elev, azim, args.resolution, args.dist)
    canvas = tile_images(images, cols=args.n)
    Image.fromarray(canvas).save(args.output)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
