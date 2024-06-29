#!/bin/bash

# set directory containing datasets
DATASET_DIR="/path/to/datasets"
mkdir -p data/cgle64

# configure CUDA environment for determinism
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# create result directory if not exist
mkdir -p results

# train models on CGL simulations varying in dispersion parameters
for dset in "$DATASET_DIR"/vary_near_critical/*
do
	dset=${dset%*/}
	cp -r $dset data/cgle64/img_align_cgle64
	python main.py "${dset##*/}"
	rm -r data/cgle64/img_align_cgle64
done

# save results and clean up
mkdir -p postprocessing/results
mv results postprocessing/results/vary_near_critical
mkdir results