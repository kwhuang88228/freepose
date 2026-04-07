import os
import numpy as np
from src.pipeline.retrieval.renderer import MeshRenderer

import webdataset as wds
from pathlib import Path
import trimesh
from loguru import logger
import argparse
import multiprocessing
from tqdm import tqdm

os.environ['PYOPENGL_PLATFORM'] = 'egl'


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


def render_chunk(args):
    """Worker function: renders one chunk of 10 meshes, equivalent to one SLURM array task."""
    job_id, mesh_ids, shards_path = args

    # Set EGL platform inside the worker (needed after spawn)
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

    start = job_id * 10
    end = (job_id + 1) * 10
    meshes = mesh_ids[start:end]

    if not meshes:
        return

    max_count = 600 * 10 * 2  # 600 images per object, 10 objects, 2 modalities (rgb, depth)
    shard_path = shards_path / "shard-%06d.tar"
    shard_writer = wds.ShardWriter(str(shard_path), maxcount=max_count, start_shard=job_id, maxsize=10e9)

    renderer = MeshRenderer(600)

    for idx, mesh_id in enumerate(meshes):
        logger.info(f'[Job {job_id}] Rendering mesh {mesh_id} ({idx + 1}/{len(meshes)})')
        mesh_path = Path('data/mesh_cache').resolve() / mesh_id / f'{mesh_id}.glb.obj'
        # ensure mesh_path exists
        if not mesh_path.exists():
            mesh_path = Path('data/mesh_cache').resolve() / mesh_id / f'{mesh_id}.obj'
            if not mesh_path.exists():
                # log this mesh as missing in a file
                with open('missing_meshes.txt', 'a') as f:
                    f.write(f'{mesh_id}\n')
                continue
        mesh = trimesh.load(mesh_path)

        mesh = fix_mesh_texture(mesh)
        mesh.apply_scale(0.25)

        # Render mesh
        results = renderer.render(mesh, cull_faces=False)

        # Save results to webdataset
        for i, (rgb, depth, _) in enumerate(results):
            shard_writer.write({
                '__key__': f'{mesh_id.replace("_", "")}_{i}',
                'rgb.png': rgb.astype(np.uint8),
                'depth.png': (depth * 1000).astype(np.uint16),
            })

    shard_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filelist", type=str, default='./data/mesh_cache.txt')
    parser.add_argument("--shards_folder", type=str, default='objaverse_shards')
    parser.add_argument("--offset", type=int, default=0,
                        help="Start job_id offset (same semantics as the SLURM version)")
    parser.add_argument("--num_chunks", type=int, default=None,
                        help="Number of consecutive chunks to process from --offset "
                             "(default: all remaining chunks)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of parallel worker processes")
    args = parser.parse_args()

    shards_path = Path('./data/datasets').resolve() / args.shards_folder
    shards_path.mkdir(exist_ok=True)

    with open(args.filelist, 'r') as f:
        mesh_ids = f.read().splitlines()

    total_chunks = (len(mesh_ids) + 9) // 10  # ceiling division: one chunk per 10 meshes

    # Build the list of job_ids to process, mirroring SLURM_ARRAY_TASK_ID + offset
    start_job = args.offset
    end_job = total_chunks if args.num_chunks is None else start_job + args.num_chunks
    job_ids = list(range(start_job, min(end_job, total_chunks)))

    logger.info(f'Processing {len(job_ids)} chunk(s) '
                f'(job_ids {job_ids[0]}–{job_ids[-1]}) '
                f'with {args.num_workers} worker(s)')

    worker_args = [(job_id, mesh_ids, shards_path) for job_id in job_ids]

    # Use 'spawn' context so each worker gets a clean process — required for EGL/OpenGL safety
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(processes=args.num_workers) as pool:
        list(tqdm(pool.imap_unordered(render_chunk, worker_args),
                  total=len(worker_args), desc='chunks', unit='chunk'))
