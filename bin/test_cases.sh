#!/bin/bash

# example experiment
cp -r preprocessing/datasets/exp_test/exp_01_crop_1 data/cgle64/img_align_cgle64
python main.py a_little_messy -c 40 40
python main_viz.py a_little_messy all
python main_viz.py a_little_messy interpolate -c 7 -r 7
rm -r data/cgle64/img_align_cgle64

# example good simulation model
cp -r preprocessing/datasets/sim_test/c1_-020_c2_050_dT_010_s_0 data/cgle64/img_align_cgle64
python main.py nice_and_round
python main_viz.py nice_and_round all
python main_viz.py nice_and_round interpolate -c 7 -r 7
rm -r data/cgle64/img_align_cgle64

# example bad simulation model
cp -r preprocessing/datasets/sim_test/c1_-060_c2_050_dT_010_s_4 data/cgle64/img_align_cgle64
python main.py discontinuous
python main_viz.py discontinuous all
python main_viz.py discontinuous interpolate -c 7 -r 7
rm -r data/cgle64/img_align_cgle64

# save results and clean up
mv results postprocessing/full_results/test_cases
mkdir -p results/discont_encode