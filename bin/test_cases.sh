#!/bin/bash

# set directory containing datasets
DATASET_DIR="/path/to/datasets"
mkdir -p data/cgle64

# configure CUDA environment for determinism
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# create result directory if not exist
mkdir -p results

# example experiment
cp -r "$DATASET_DIR"/exp_orderp_vary/exp_01_crop_1 data/cgle64/img_align_cgle64
python main.py a_little_messy -c 32 32 -i 0
python main_viz.py a_little_messy all
rm -r data/cgle64/img_align_cgle64

# example simulation
cp -r "$DATASET_DIR"/sim_regime_vary/c1_-020_c2_050_dT_010_s_0 data/cgle64/img_align_cgle64
python main.py nice_and_round -i 0
python main_viz.py nice_and_round all
rm -r data/cgle64/img_align_cgle64

# save results and clean up
mkdir -p postprocessing/results
mv results postprocessing/results/test_cases
mkdir results
