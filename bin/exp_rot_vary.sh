#!/bin/bash

# set directory containing datasets
DATASET_DIR="/path/to/datasets"
mkdir -p data/cgle64

# configure environment for CUDA determinism
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# create result directory if not exist
mkdir -p results

# train models on angle-varying views of the same experiment
# loop over datasets with twelve training seeds
declare -a SEEDS=("1234" "1243" "1324" "1342" "1423" "1432" "2134" "2143" "2314" "2341" "2413" "2431")
for dset in "$DATASET_DIR"/exp_rot_vary/*
do
dset=${dset%*/}
cp -r $dset data/cgle64/img_align_cgle64
for seed in "${SEEDS[@]}"
do
python main.py "${dset##*/}"_"$seed" -c 32 32 -s "$seed"
done
rm -r data/cgle64/img_align_cgle64
done

# save results and clean up
mkdir -p postprocessing/results
mv results postprocessing/results/exp_rot_vary
mkdir results
