import os
import numpy as np
from src.pipeline.retrieval.renderer import MeshRenderer

import webdataset as wds
from pathlib import Path
import trimesh
from loguru import logger
import argparse
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

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--filelist", type=str, default='./data/mesh_cache.txt')
    args.add_argument("--shards_folder", type=str, default='objaverse_shards')
    args = args.parse_args()

    shards_path = Path('./data/datasets').resolve() / args.shards_folder
    shards_path.mkdir(exist_ok=True)

    with open(args.filelist, 'r') as f:
        mesh_ids = f.read().splitlines()

    meshes = [mesh_ids[i] for i in [37508, 40249, 43066, 45852, 46020, 46030, 46031, 46032, 46033, 46034, 46035, 46036, 46023, 46015, 46025, 46026, 46027, 46028, 46029, 46018]]

    # Create webdataset
    max_count = 600*10*2 # 600 images per object, 10 objects, 2 modalities (rgb, depth)
    shard_path = shards_path / "shard-%06d.tar"

    renderer = MeshRenderer(600)

    mesh_chunks = [meshes[i:i+10] for i in range(0, len(meshes), 10)]
    for chunk_idx, chunk in tqdm(enumerate(mesh_chunks)):
        shard_writer = wds.ShardWriter(str(shard_path), maxcount=max_count, start_shard=4603 + chunk_idx, maxsize=10e9)
        logger.info(f'Writing to shard: {shards_path / ("shard-%06d.tar" % (4603 + chunk_idx))}')

        for idx, mesh_id in enumerate(chunk):
            logger.info(f'Rendering mesh {mesh_id} ({idx+1}/{len(chunk)}) [chunk {chunk_idx+1}/{len(mesh_chunks)}]')
            mesh_path = Path('data/mesh_cache').resolve() / mesh_id / f'{mesh_id}.obj'
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
