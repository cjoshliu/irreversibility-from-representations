#!/bin/bash

# set directory containing datasets
DATASET_DIR="/path/to/datasets"
mkdir -p data/cgle64

# configure CUDA environment for determinism
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# create result directory if not exist
mkdir -p results

# loop over datasets
for dset in "$DATASET_DIR"/sim_regime_vary/c1_-020_c2_*_dT_010_s_*/
do
	dset=${dset%*/}
	cp -r "$dset" data/cgle64/img_align_cgle64
	
	# train models on dataset using different hyperparameter choices
	python main.py "${dset##*/}"_lo_window -c 8 16
	python main.py "${dset##*/}"_hi_window -c 32 16
	
	python main.py "${dset##*/}"_lo_stop -c 16 8
	python main.py "${dset##*/}"_hi_stop -c 16 32
	
	python main.py "${dset##*/}"_lo_vaelr --lr "5e-4"
	python main.py "${dset##*/}"_hi_vaelr --lr "2e-3"
	
	python main.py "${dset##*/}"_lo_dlr --lr-disc "5e-5"
	python main.py "${dset##*/}"_hi_dlr --lr-disc "2e-4"
	
	rm -r data/cgle64/img_align_cgle64
done

# save results and clean up
mkdir -p postprocessing/results
mv results postprocessing/results/trend_hyperp_vary
mkdir results
