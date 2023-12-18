#!/bin/bash

# set directory containing datasets
DATASET_DIR="/path/to/datasets"
mkdir -p data/cgle64

# configure CUDA environment for determinism
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# create result directory if not exist
mkdir -p results

# set datasets and latent dimensionalities
declare -a DATASETS=("c1_-020_c2_050_dT_010_s_0" "c1_-020_c2_050_dT_010_s_4" "c1_-020_c2_050_dT_010_s_6" "c1_-100_c2_130_dT_010_s_0" "c1_-100_c2_130_dT_010_s_1" "c1_-100_c2_130_dT_010_s_2")
declare -a LATENTS=("4" "8" "16")

# train models on stable and turbulent patterns varying latent dimensionality
for dset in "${DATASETS[@]}"
do
cp -r "$DATASET_DIR"/sim_regime_vary/"$dset" data/cgle64/img_align_cgle64
for ltnt in "${LATENTS[@]}"
do
python main.py "$dset"_z_"$ltnt" -z "$ltnt"
python main_viz.py "$dset"_z_"$ltnt" traversals --stack
done
rm -r data/cgle64/img_align_cgle64
done

# save results and clean up
mkdir -p postprocessing/results
mv results postprocessing/results/sim_z_vary
mkdir results
