"""
Build a JSON mapping of mesh UID -> object name for all meshes in mesh_cache.
Currently handles Objaverse meshes (32-char hex UIDs).
Output: inference/metadata.json
"""

import json
import re
from pathlib import Path
from tqdm import tqdm

import objaverse

MESH_CACHE = Path(__file__).parent.parent / "data" / "mesh_cache"
OUTPUT = Path(__file__).parent / "metadata.json"

OBJAVERSE_UID_RE = re.compile(r'^[0-9a-f]{32}$')


def is_objaverse_uid(name: str) -> bool:
    return bool(OBJAVERSE_UID_RE.match(name))


def main():
    mesh_names = [p.name for p in MESH_CACHE.iterdir() if p.is_dir()]

    objaverse_uids = [n for n in mesh_names if is_objaverse_uid(n)]
    other_uids = [n for n in mesh_names if not is_objaverse_uid(n)]

    print(f"Found {len(objaverse_uids)} Objaverse UIDs, {len(other_uids)} other UIDs")

    print("Loading Objaverse annotations...")
    annotations = objaverse.load_annotations(objaverse_uids)

    metadata = {}
    missing = []
    for uid in tqdm(objaverse_uids, desc="Processing UIDs"):
        if uid in annotations and annotations[uid].get("name"):
            metadata[uid] = annotations[uid]["name"]
        else:
            metadata[uid] = None
            missing.append(uid)

    if missing:
        print(f"Warning: {len(missing)} UIDs had no annotation name")

    with open(OUTPUT, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote {len(metadata)} entries to {OUTPUT}")


if __name__ == "__main__":
    main()
