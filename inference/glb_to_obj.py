import argparse
import os
import trimesh

parser = argparse.ArgumentParser(description='Convert GLB to OBJ')
parser.add_argument('--input_mesh_path', type=str, help='Path to the input GLB file')
parser.add_argument('--output_mesh_path', type=str, help='Path to the output OBJ file')
args = parser.parse_args()

scene_or_mesh = trimesh.load(args.input_mesh_path)

if isinstance(scene_or_mesh, trimesh.Scene):
    mesh = scene_or_mesh.dump(concatenate=True)
else:
    mesh = scene_or_mesh

mesh.export(args.output_mesh_path)