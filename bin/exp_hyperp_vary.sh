#!/bin/bash

# set directory containing datasets
DATASET_DIR="/path/to/datasets"
mkdir -p data/cgle64

# configure CUDA environment for determinism
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# create result directory if not exist
mkdir -p results

# declare training seeds
declare -a SEEDS=("1234" "1243" "1324" "1342" "1423" "1432" "2134" "2143" "2314" "2341" "2413" "2431")
# loop over datasets
for dset in "$DATASET_DIR"/exp_atp_vary/*
do
dset=${dset%*/}
cp -r "$dset" data/cgle64/img_align_cgle64
# loop over training seeds
for seed in "${SEEDS[@]}"
do

# train models on dataset using different hyperparameter choices
python main.py "${dset##*/}"_lo_window_"$seed" -c 16 32 -s "$seed"
python main.py "${dset##*/}"_hi_window_"$seed" -c 64 32 -s "$seed"

python main.py "${dset##*/}"_lo_stop_"$seed" -c 32 16 -s "$seed"
python main.py "${dset##*/}"_hi_stop_"$seed" -c 32 64 -s "$seed"

python main.py "${dset##*/}"_lo_vaelr_"$seed" -c 32 32 --lr "5e-4" -s "$seed"
python main.py "${dset##*/}"_hi_vaelr_"$seed" -c 32 32 --lr "2e-3" -s "$seed"

python main.py "${dset##*/}"_lo_dlr_"$seed" -c 32 32 --lr-disc "5e-5" -s "$seed"
python main.py "${dset##*/}"_hi_dlr_"$seed" -c 32 32 --lr-disc "2e-4" -s "$seed"

done
rm -r data/cgle64/img_align_cgle64
done

# save results and clean up
mkdir -p postprocessing/results
mv results postprocessing/results/trend_hyperp_vary
mkdir results
