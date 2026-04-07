#!/bin/bash

set -euxo pipefail

echo "Creating datasets directory..."
mkdir -p ./data/datasets
echo "Datasets directory created."

echo "Running download_objaverse.py script..."
python scripts/download_objaverse.py
echo "download_objaverse.py script completed."

echo "Entering ./data/ directory..."
cd ./data/

echo "Downloading objaverse_shards_ffa_22.npy..."
wget --progress=bar:force https://data.ciirc.cvut.cz/public/projects/2025FreePose/objaverse_shards_ffa_22.npy
echo "objaverse_shards_ffa_22.npy download completed."

echo "Entering ./datasets/ directory..."
cd ./datasets/

echo "Downloading google_scanned_objects.zip..."
wget --progress=bar:force https://www.paris.inria.fr/archive_ylabbeprojectsdata/megapose/tars/google_scanned_objects.zip
echo "google_scanned_objects.zip download completed."

echo "Unzipping google_scanned_objects.zip..."
unzip google_scanned_objects.zip && rm google_scanned_objects.zip
echo "google_scanned_objects.zip unzipped and removed."

cd ../../

echo "Running resize_meshes.py script..."
python scripts/resize_meshes.py
echo "resize_meshes.py script completed."