import os
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

os.environ['PYOPENGL_PLATFORM'] = 'egl'
import pyrender

# ── 3-D bounding-box helpers ──────────────────────────────────────────────────

# BGR colors for each axis
_AXIS_COLORS = {
    'x': (0,   0, 255),   # Red
    'y': (0, 255,   0),   # Green
    'z': (255,  0,   0),  # Blue
}

# 12 edges of a box; each entry: (corner_a, corner_b, axis_color_key)
# Corner order from _get_bbox_corners():
#   0:(xmin,ymin,zmin)  1:(xmax,ymin,zmin)  2:(xmax,ymax,zmin)  3:(xmin,ymax,zmin)
#   4:(xmin,ymin,zmax)  5:(xmax,ymin,zmax)  6:(xmax,ymax,zmax)  7:(xmin,ymax,zmax)
_BBOX_EDGES = [
    (0, 1, 'x'), (3, 2, 'x'), (4, 5, 'x'), (7, 6, 'x'),  # x-direction (red)
    (0, 3, 'y'), (1, 2, 'y'), (4, 7, 'y'), (5, 6, 'y'),  # y-direction (green)
    (0, 4, 'z'), (1, 5, 'z'), (2, 6, 'z'), (3, 7, 'z'),  # z-direction (blue)
]


def _get_bbox_corners(mesh):
    """Return the 8 AABB corners of *mesh* in object space."""
    (xmin, ymin, zmin), (xmax, ymax, zmax) = mesh.bounds
    return np.array([
        [xmin, ymin, zmin], [xmax, ymin, zmin],
        [xmax, ymax, zmin], [xmin, ymax, zmin],
        [xmin, ymin, zmax], [xmax, ymin, zmax],
        [xmax, ymax, zmax], [xmin, ymax, zmax],
    ])


def _project(pts_3d, K, R, t):
    """Project Nx3 object-space points to Nx2 pixel coords."""
    pts_cam = (R @ pts_3d.T).T + t
    pts_h   = (K @ pts_cam.T).T
    return pts_h[:, :2] / pts_h[:, 2:3]


def _draw_bbox_3d(img, corners_2d, thickness=2):
    for i, j, axis in _BBOX_EDGES:
        p1 = tuple(corners_2d[i].astype(int))
        p2 = tuple(corners_2d[j].astype(int))
        cv2.line(img, p1, p2, _AXIS_COLORS[axis], thickness, cv2.LINE_AA)


def _draw_axes(img, K, R, t, length, thickness=3):
    center = _project(np.zeros((1, 3)), K, R, t)[0]
    for vec, key in zip(np.eye(3), ['x', 'y', 'z']):
        tip = _project((vec * length)[np.newaxis], K, R, t)[0]
        cv2.arrowedLine(img, tuple(center.astype(int)), tuple(tip.astype(int)),
                        _AXIS_COLORS[key], thickness, cv2.LINE_AA, tipLength=0.25)


def _draw_pose_text(img, R, t, origin_2d, obj_idx):
    rx, ry, rz = Rotation.from_matrix(R).as_euler('xyz', degrees=True)
    tx, ty, tz = t
    lines = [
        f"Obj {obj_idx}",
        f"t  ({tx:+.0f}, {ty:+.0f}, {tz:+.0f}) mm",
        f"R  ({rx:+.1f}, {ry:+.1f}, {rz:+.1f}) deg",
    ]
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
    x0, y0 = int(origin_2d[0]), int(origin_2d[1])
    for k, line in enumerate(lines):
        (tw, th), _ = cv2.getTextSize(line, font, scale, thick)
        yy = y0 + k * (th + 4)
        cv2.rectangle(img, (x0 - 2, yy - th - 2), (x0 + tw + 2, yy + 2), (0, 0, 0), -1)
        cv2.putText(img, line, (x0, yy), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

DATA_DIR = Path('data')

def render(mesh, width, height, K, transform):
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height, point_size=1.0)
    camera = pyrender.IntrinsicsCamera(K[0, 0], K[1, 1], K[0, 2], K[1, 2], znear=0.0001, zfar=9999)
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0], ambient_light=[5.0, 5.0, 5.0])

    opencv2opengl = np.eye(4)
    opencv2opengl[1, 1] = -1
    opencv2opengl[2, 2] = -1

    scene.add_node(pyrender.Node(mesh=pyrender.Mesh.from_trimesh(mesh), matrix=transform))
    scene.add_node(pyrender.Node(camera=camera, matrix=opencv2opengl))

    color_buff, depth_buff = renderer.render(scene)
    return color_buff, depth_buff

def fix_mesh_texture(mesh):
    has_visual = hasattr(mesh, 'visual')
    has_material = has_visual and hasattr(mesh.visual, 'material')
    has_image = has_material and hasattr(mesh.visual.material, 'image')

    if has_image:
        if mesh.visual.material.image is not None:
            if mesh.visual.material.image.mode == 'LA':
                mesh.visual.material.image = mesh.visual.material.image.convert('RGBA')
            elif mesh.visual.material.image.mode == '1':
                mesh.visual.material.image = mesh.visual.material.image.convert('RGB')

    return mesh

def load_mesh(path, scale=1.0):
    mesh = trimesh.load_mesh(path)
    mesh = fix_mesh_texture(mesh)
    mesh.apply_scale(scale)
    return mesh


def create_outline(mask, color=[0.21,0.49,0.74]):
    kernel = np.ones((3, 3), np.uint8)
    mask_rgb = np.stack([np.uint8(mask)] * 3, 2) * 255
    mask_rgb = cv2.dilate(mask_rgb, kernel, iterations=2)

    canny = cv2.Canny(mask_rgb, threshold1=30, threshold2=100)
    canny = cv2.dilate(canny, kernel, iterations=2)
    outline = np.clip(np.stack([canny] * 3, 2).astype(np.float32) * np.array([[color]]), 0, 255.).astype(np.uint8)
    outline = np.concatenate([outline, canny[:, :, np.newaxis]], 2)
    return outline


def main(args):
    pred_path = DATA_DIR / 'results' / 'videos' / args.video / args.predictions
    pred = pd.read_csv(pred_path)
    viz_dir = pred_path.parent / f'viz_{pred_path.stem}'
    viz_dir.mkdir(exist_ok=True, parents=True)

    N = (pred['im_id'] == 0).sum().item()
    objects_pred = [pred.iloc[i::N] for i in range(N)]
    mesh_ids = [x.iloc[0]['obj_id'] for x in objects_pred]
    scales = [x.iloc[0]['scale'].item() for x in objects_pred]
    meshes, all_bbox_corners = [], []
    for mesh_id, scale in zip(mesh_ids, scales):
        mesh = load_mesh(DATA_DIR / 'mesh_cache' / mesh_id / f'{mesh_id}.glb.obj', scale)
        meshes.append(mesh)
        all_bbox_corners.append(_get_bbox_corners(mesh))

    video_path = DATA_DIR / 'datasets' / 'videos' / args.video
    frame_paths = sorted(video_path.iterdir())

    h,w,_ = np.array(Image.open(frame_paths[0])).shape

    f = np.sqrt(h**2 + w**2)
    cx = w/2
    cy = h/2
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])

    # Optional: second output directory for 3-D bounding-box visualization
    if args.bbox:
        bbox_dir = pred_path.parent / f'viz_bbox_{pred_path.stem}'
        bbox_dir.mkdir(exist_ok=True, parents=True)

    for frame_idx, frame_path in tqdm(enumerate(frame_paths), ncols=100, total=len(frame_paths)):
        frame = Image.open(frame_path)

        # Extract poses
        Ts = []
        for object_idx, mesh in enumerate(meshes):
            pred = objects_pred[object_idx].iloc[frame_idx]

            R = np.array([float(x) for x in pred['R'].split(' ')]).reshape(3,3)
            t = np.array([float(x) for x in pred['t'].split(' ')])
            T = np.eye(4)
            T[:3,:3] = R
            T[:3,3] = t
            Ts.append(T)

        # Sort object by distance to camera - render further objects first
        distances = [np.linalg.norm(T[:3,3]) for T in Ts]
        order = np.argsort(distances)[::-1]

        # Render and overlay (original mesh visualization)
        for i in order:
            T = Ts[i]
            mesh = meshes[i]

            color, depth = render(mesh, w, h, K, T)
            mask = (depth>0)
            outline = Image.fromarray(create_outline(mask))
            ren_img = Image.fromarray(np.concatenate([color, np.uint8(mask)[:, :, np.newaxis] * 255], 2))

            frame.paste(ren_img, (0, 0), ren_img)
            frame.paste(outline, (0, 0), outline)

        frame.save(viz_dir / f'{frame_idx:06d}.png')

        # ── 3-D bounding-box visualization (--bbox) ───────────────────────────
        if args.bbox:
            img = cv2.cvtColor(np.array(Image.open(frame_path)), cv2.COLOR_RGB2BGR)
            for obj_idx, (mesh, corners) in enumerate(zip(meshes, all_bbox_corners)):
                row = objects_pred[obj_idx].iloc[frame_idx]
                R = np.array([float(x) for x in row['R'].split(' ')]).reshape(3, 3)
                t = np.array([float(x) for x in row['t'].split(' ')])

                corners_2d = _project(corners, K, R, t)
                _draw_bbox_3d(img, corners_2d)

                # axis_len = np.max(mesh.bounds[1] - mesh.bounds[0]) * 0.4
                # _draw_axes(img, K, R, t, axis_len)

                # center_2d = _project(np.zeros((1, 3)), K, R, t)[0]
                # _draw_pose_text(img, R, t, center_2d + np.array([6, 6]), obj_idx)

            Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(
                bbox_dir / f'{frame_idx:06d}.png'
            )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--predictions", '-p', type=str, required=True)
    parser.add_argument("--bbox", action="store_true",
                        help="Also render a 3-D bounding-box visualization showing 6DoF pose "
                             "(saved to viz_bbox_<stem>/ next to the mesh overlay).")

    args = parser.parse_args()
    main(args)
