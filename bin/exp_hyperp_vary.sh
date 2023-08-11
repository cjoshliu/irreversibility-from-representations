#!/bin/bash

# set directory containing datasets
DATASET_DIR="/path/to/datasets"
mkdir -p data/cgle64

# configure environment for CUDA determinism
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# create result directory if not exist
mkdir -p results

# use representative experiment dataset
cp -r "$DATASET_DIR"/exp_orderp_vary/exp_01_crop_1 data/cgle64/img_align_cgle64

# train models on one experiment using different hyperparameter choices
python main.py low_window -c 16 32
python main.py high_window -c 64 32

python main.py low_stop -c 32 16
python main.py high_stop -c 32 64

python main.py low_vaelr -c 32 32 --lr "5e-4"
python main.py high_vaelr -c 32 32 --lr "2e-3"

python main.py low_dlr -c 32 32 --lr-disc "5e-5"
python main.py high_dlr -c 32 32 --lr-disc "2e-4"

# save results and clean up
rm -r data/cgle64/img_align_cgle64
mkdir -p postprocessing/results
mv results postprocessing/results/exp_hyperp_vary
mkdir results