#!/bin/bash

# loop over datasets with training seed 1234
for dset in preprocessing/datasets/sim_tstep_vary/*
do
	dset=${dset%*/}
	cp -r $dset data/cgle64/img_align_cgle64
	python main.py "${dset##*/}"
	python main_viz.py "${dset##*/}" all
	python main_viz.py "${dset##*/}" interpolate -c 7 -r 7
	rm -r data/cgle64/img_align_cgle64
done

# loop over datasets with training seed 1243
# datasets that have discontinuous encodings in first model
declare -a DATA_1243=("c1_-020_c2_050_dT_001_s_1")
for dset in "${DATA_1243[@]}"
do
	dset=${dset%*/}
	mv results/"${dset##*/}" results/discont_encode/"${dset##*/}"_v_1
	cp -r preprocessing/datasets/sim_tstep_vary/"$dset" data/cgle64/img_align_cgle64
	python main.py "${dset##*/}" -s 1243
	python main_viz.py "${dset##*/}" all
	python main_viz.py "${dset##*/}" interpolate -c 7 -r 7
	rm -r data/cgle64/img_align_cgle64
done

# save results and clean up
mv results postprocessing/full_results/sim_tstep_vary
mkdir -p results/discont_encode