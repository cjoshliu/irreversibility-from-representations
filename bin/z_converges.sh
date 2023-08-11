#!/bin/bash

# set directory containing datasets
DATASET_DIR="/path/to/datasets"
mkdir -p data/cgle64

# configure CUDA environment for determinism
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# create result directory if not exist
mkdir -p results

# convergence to two latent dimensions on a representative simulation dataset
cp -r "$DATASET_DIR"/sim_regime_vary/c1_-020_c2_050_dT_010_s_0 data/cgle64/img_align_cgle64
python main.py z_converges -z 5 -s 1243
python main_viz.py z_converges traversals --stack
rm -r data/cgle64/img_align_cgle64

# save results and clean up
mkdir -p postprocessing/results
mv results/z_converges postprocessing/results/z_converges