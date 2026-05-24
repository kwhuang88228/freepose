#!/usr/bin/env python3
import argparse
import glob
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", default="/share/hariharan/kh775/code/freepose/MV-SAM3D")
parser.add_argument("--sequence_id", required=True)
args = parser.parse_args()

BASE_DIR = args.base_dir
SEQUENCE_ID = args.sequence_id
OBJECT_NAME = SEQUENCE_ID.split("_")[-1]

FRAMES_DIR = f"{BASE_DIR}/scratch/frames/{SEQUENCE_ID}"
DA3_OUTPUT_DIR = f"{BASE_DIR}/scratch/da3_outputs/{SEQUENCE_ID}"


def run(cmd):
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=BASE_DIR)


run([
    "python", "scripts/run_da3.py",
    "--image_dir", f"scratch/frames/{SEQUENCE_ID}/images",
    "--output_dir", f"scratch/da3_outputs/{SEQUENCE_ID}",
])

run([
    "python", "run_inference_weighted.py",
    "--input_path", FRAMES_DIR,
    "--mask_prompt", OBJECT_NAME,
    "--da3_output", f"{DA3_OUTPUT_DIR}/da3_output.npz",
])

glb_pattern = f"{BASE_DIR}/visualization/{SEQUENCE_ID}/{OBJECT_NAME}/*/result.glb"
glb_matches = sorted(glob.glob(glb_pattern))
if not glb_matches:
    sys.exit(f"No result.glb found matching: {glb_pattern}")
glb_path = glb_matches[-1]

run([
    "python", "scratch/view_glb.py",
    glb_path,
    "--output", f"{BASE_DIR}/scratch/gs_views/{OBJECT_NAME}.png",
])
