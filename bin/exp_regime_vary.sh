#!/bin/bash

# trains models on experiments at different Rho-pathway activities
# loop over datasets with twelve training seeds

declare -a SEEDS=("1234" "1243" "1324" "1342" "1423" "1432" "2134" "2143" "2314" "2341" "2413" "2431")
for dset in preprocessing/datasets/exp_regime_vary/*
do
dset=${dset%*/}
cp -r $dset data/cgle64/img_align_cgle64
for seed in "${SEEDS[@]}"
do
python main.py "${dset##*/}"_"$seed" -c 40 40 -s "$seed"
python main_viz.py "${dset##*/}"_"$seed" all
python main_viz.py "${dset##*/}"_"$seed" interpolate -c 7 -r 7
done
rm -r data/cgle64/img_align_cgle64
done

# save results and clean up
mv results postprocessing/full_results/exp_regime_vary
mkdir -p results/discont_encode
