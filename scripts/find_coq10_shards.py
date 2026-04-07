import tarfile
import glob
import os
from tqdm import tqdm

shard_dir = os.path.join(os.path.dirname(__file__), "../data/datasets/objaverse_shards")
tar_files = sorted(glob.glob(os.path.join(shard_dir, "shard-*.tar")), reverse=True)

print(f"Searching {len(tar_files)} shard files for entries matching '*CoQ10*'...\n")

for tar_path in tqdm(tar_files, desc="Processing shards", unit="shard"):
    try:
        with tarfile.open(tar_path, "r") as tf:
            members = tf.getnames()
            matches = [m for m in members if "CoQ10" in m]
            if matches:
                print(f"{os.path.basename(tar_path)}:")
                for m in matches:
                    print(f"  {m}")
    except Exception as e:
        print(f"Error reading {tar_path}: {e}")
