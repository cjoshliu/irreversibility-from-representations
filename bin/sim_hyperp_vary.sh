#!/bin/bash

# set directory containing datasets
DATASET_DIR="/path/to/datasets"
mkdir -p data/cgle64

# configure CUDA environment for determinism
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# create result directory if not exist
mkdir -p results

# use representative simulation dataset
cp -r "$DATASET_DIR"/sim_regime_vary/c1_-020_c2_050_dT_010_s_0 data/cgle64/img_align_cgle64

# train models on one simulation using different hyperparameter choices
python main.py low_window -c 8 16
python main.py high_window -c 32 16

python main.py low_stop -c 16 8
python main.py high_stop -c 16 32

python main.py low_vaelr --lr "5e-4"
python main.py high_vaelr --lr "2e-3"

python main.py low_dlr --lr-disc "5e-5"
python main.py high_dlr --lr-disc "2e-4"

# save results and clean up
rm -r data/cgle64/img_align_cgle64
mkdir -p postprocessing/results
mv results postprocessing/results/sim_hyperp_vary
mkdir results