import os
import cv2
import math
import torch
import numpy as np
from pathlib import Path
from loguru import logger
import argparse
from tqdm import tqdm
import torch.multiprocessing as mp

from src.dataloader.template import WebTemplateDataset
from src.pipeline.retrieval.dino import DINOv2FeatureExtractor


def worker(rank: int, world_size: int, args: argparse.Namespace, gpu_ids: list[int]) -> None:
    gpu_id = gpu_ids[rank % len(gpu_ids)]
    device = f'cuda:{gpu_id}'

    shards_path = Path('data/datasets').resolve() / args.shards_folder
    features_path = Path('data/datasets').resolve() / f'{args.shards_folder}_{args.feature}_{args.layer}'
    features_path.mkdir(parents=True, exist_ok=True)

    filelist_path = Path('data').resolve() / args.filelist
    model = DINOv2FeatureExtractor().to(device, dtype=torch.bfloat16)
    feature_type = 'cls' if args.feature == 'cls' else 'patch'

    dataset = WebTemplateDataset(shards_path.as_posix(), filelist_path.as_posix(), crop=False)
    total = len(dataset)

    chunk_size = math.ceil(total / world_size)
    start_idx = rank * chunk_size
    end_idx = min(start_idx + chunk_size, total)

    for idx in tqdm(range(start_idx, end_idx), desc=f"Worker {rank} (GPU {gpu_id})", position=rank):
        model_name = dataset.frame_index[idx]
        out_path = features_path / f'{model_name}.npy'
        if out_path.exists():
            logger.info(f'[Worker {rank}] Skipping {model_name}, already exists')
            continue

        logger.info(f'[Worker {rank}] Processing {idx + 1} / {total}')
        sample = dataset[idx]

        if sample['templates'] is None:
            logger.warning(f'[Worker {rank}] Skipping {model_name}, no templates')
            continue

        templates = sample['templates'].to(device, dtype=torch.bfloat16)

        features = []
        for i in range(0, len(templates), args.batch_size):
            batch = templates[i:i + args.batch_size]
            features.append(model(batch, layer=args.layer, feature_type=feature_type))
        features = torch.cat(features, dim=0)

        if args.feature == 'ffa':
            avg_feats = []
            for feat, mask in zip(features, sample['masks']):
                mask_orig = mask.clone()
                mask = cv2.resize(mask.float().numpy(), (30, 30), interpolation=cv2.INTER_AREA) > 0
                avg_feat = feat[mask.flatten()]
                avg_feat = avg_feat.mean(dim=0).float().cpu().numpy()

                if np.isnan(avg_feat).any():
                    logger.warning(f'[Worker {rank}] Feature {model_name} contains NaNs')
                    logger.warning(f'[Worker {rank}] Mask sum is {mask.sum()}')
                    logger.warning(f'[Worker {rank}] Mask orig sum is {mask_orig.sum()}')
                    cv2.imwrite(f'{model_name}_mask_orig.png', mask_orig.numpy().astype(np.uint8) * 255)
                    cv2.imwrite(f'{model_name}_mask.png', mask.astype(np.uint8) * 255)
                    continue

                avg_feats.append(avg_feat)

            avg_feats = np.stack(avg_feats)
            np.save(out_path.as_posix(), avg_feats)
        else:
            np.save(out_path.as_posix(), features.float().cpu().numpy())

    logger.info(f'Worker {rank} done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shards_folder', type=str, default='objaverse_shards')
    parser.add_argument('--filelist', type=str, default='mesh_cache.csv')
    parser.add_argument('--feature', type=str, default='ffa', choices=['ffa', 'cls'])
    parser.add_argument('--layer', type=int, default=22)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of parallel workers. Defaults to the number of available GPUs.')
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError('No CUDA GPUs found. This script requires at least one GPU.')

    world_size = args.num_workers if args.num_workers is not None else num_gpus
    gpu_ids = list(range(num_gpus))

    logger.info(f'Launching {world_size} workers across {num_gpus} GPU(s): {gpu_ids}')

    mp.set_start_method('spawn', force=True)

    processes = []
    for rank in range(world_size):
        p = mp.Process(target=worker, args=(rank, world_size, args, gpu_ids))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    failed = [rank for rank, p in enumerate(processes) if p.exitcode != 0]
    if failed:
        logger.error(f'Workers {failed} exited with non-zero exit codes.')
    else:
        logger.info('All workers finished successfully.')
