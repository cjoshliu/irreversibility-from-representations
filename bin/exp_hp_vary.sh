#!/bin/bash

# for evaluating systematic error in experiment irreversibilities due to choice of hyperparameters
cp -r preprocessing/datasets/exp_test/exp_01_crop_1 data/cgle64/img_align_cgle64

# train with different hyperparameter choices
python main.py low_window -c 20 40
python main.py high_window -c 80 40

python main.py low_stop -c 40 20
python main.py high_stop -c 40 80

python main.py low_vaelr -c 40 40 --lr "5e-4"
python main.py high_vaelr -c 40 40 --lr "2e-3"

python main.py low_dlr -c 40 40 --lr-disc 0.000025
python main.py high_dlr -c 40 40 --lr-disc "1e-4"

python main.py low_anneals -c 40 40 -a 5000
python main.py high_anneals -c 40 40 -a 20000

python main.py low_fg -c 40 40 --factor-G 4
python main.py high_fg -c 40 40 --factor-G 16

# generate figures
declare -a HP_CHOICES=("low_window" "high_window" "low_stop" "high_stop" "low_vaelr" "high_vaelr" "low_dlr" "high_dlr" "low_anneals" "high_anneals" "low_fg" "high_fg")
for model in "${HP_CHOICES[@]}"
do
	python main_viz.py "$model" all
	python main_viz.py "$model" interpolate -c 7 -r 7
done

# save results and clean up
rm -r data/cgle64/img_align_cgle64
mv results postprocessing/full_results/exp_hp_vary
mkdir -p results/discont_encode