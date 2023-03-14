#!/bin/bash

cp -r preprocessing/datasets/sim_test/c1_-020_c2_050_dT_010_s_0 data/cgle64/img_align_cgle64

# train with different hyperparameter choices
python main.py low_window -c 4 8
python main.py high_window -c 16 8

python main.py low_stop -c 8 4
python main.py high_stop -c 8 16

python main.py low_vaelr --lr "5e-4"
python main.py high_vaelr --lr "2e-3"

python main.py low_dlr --lr-disc 0.000025
python main.py high_dlr --lr-disc "1e-4"

python main.py low_anneals -a 5000
python main.py high_anneals -a 20000

python main.py low_fg --factor-G 4
python main.py high_fg --factor-G 16

# generate figures
declare -a HP_CHOICES=("low_window" "high_window" "low_stop" "high_stop" "low_vaelr" "high_vaelr" "low_dlr" "high_dlr" "low_anneals" "high_anneals" "low_fg" "high_fg")
for model in "${HP_CHOICES[@]}"
do
	python main_viz.py "$model" all
	python main_viz.py "$model" interpolate -c 7 -r 7
done

# save results and clean up
rm -r data/cgle64/img_align_cgle64
mv results postprocessing/full_results/sim_hp_vary
mkdir -p results/discont_encode