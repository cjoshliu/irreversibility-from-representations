# Irreversibility from representations [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/cjoshliu/irreversibility-from-representations/blob/master/LICENSE) [![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7737963.svg)](https://doi.org/10.5281/zenodo.7737963)

This repository contains code for simulating complex Ginzburg-Landau (CGL) phase fields, training models to represent CGL and Rho dynamics in a low-dimensional latent space, and bounding irreversibility from latent-space representations using the Ziv-Merhav estimator.
It is modified from [simulator code by David M. Winterbottom](https://github.com/codeinthehole/codeinthehole.com/blob/58ad3d28ddefb64350ec883b291d4dbe1df096f7/www/static/tutorial/files/CGLsim2D.m) and [VAE code by Yann Dubois](https://github.com/YannDubs/disentangling-vae).

The simulator uses methods described in ["Exponential time differencing for stiff systems."](https://doi.org/10.1006/jcph.2002.6995)
The default VAE uses architecture and loss described in ["Understanding disentangling in β-VAE"](https://arxiv.org/abs/1804.03599) and ["Disentangling by factorising,"](https://arxiv.org/abs/1802.05983) respectively.
However, additional losses are implemented.
The ZM estimator uses methods described in ["Entropy production and Kullback-Leibler divergence between stationary trajectories of discrete systems."](https://doi.org/10.1103/PhysRevE.85.031129)

Table of Contents:
1. [Install](#install)
2. [Preprocess](#preprocess)
3. [Train](#train)
4. [Plot](#plot)
5. [Postprocess](#postprocess)
6. [Examples](#examples)
7. [Help](#help)

## Install
```
# clone repo
cd /path/to/repo
mkdir -p {data/cgle64,results/discont_encode,postprocessing/full_results}
# set up and activate environment
pip install -r requirements.txt
```

## Preprocess
[Training data](https://doi.org/10.5281/zenodo.7734339) are deposited on Zenodo as `datasets.zip`, which should be extracted to `preprocessing/datasets` before running scripts included in `bin`.
Alternatively, use `preprocessing/SimCGL.m` to generate your own simulation videos.
Next, modify the first cell of `preprocessing/vid_to_dset.ipynb` to segment your videos into datasets containing two-frame segments.

## Train
Copy dataset from preprocessing (or wherever else) to `data/cgle64/img_align_cgle64`. For example:
```
cp -r preprocessing/datasets/sim_test/c1_-020_c2_050_dT_010_s_0 data/cgle64/img_align_cgle64
```
Run `python main.py <model-name> <param>` to train and/or evaluate a model. For example:
```
python main.py example_model_name --lr 0.001 -e 3
```
Results are generated in `results`, but should be saved to `postprocessing/full_results`. 

### Output
* `model.pt`: Model at the end of training. 
* `model-i.pt`: Model checkpoint after `<i>` iterations. By default saves every 100.
* `specs.json`: The parameters used to run the program (default and modified with CLI).
* `training.gif`: GIF of latent traversals of the latent dimensions z at each epoch of training.
* `train_losses.log`: All (sub-)losses computed during training.
* `test_losses.log`: All (sub-)losses computed at the end of training with the model in evaluate mode.
* `pct_errs.csv`: Original-to-reconstruction L2 norm as a percentage of original-to-(vanishing-field reference) L2 norm for each observation in training set.
* `reconstruct_losses.csv`: Reconstruction loss averaged over all observations in training set at each epoch during training.
* `latent_logvars.csv`: Model-generated latent log-variances for each observation in training set.
* `latent_means.csv`: Model-generated latent means for each observation in training set. These can be used as Ziv-Merhav irreversibility-estimator inputs.

## Plot
Run `python main_viz.py <model-name> <plot-types> <param>` to plot using pretrained models in `results`. For example:
```
python main_viz.py example_model_name all -c 7
```
This will save plots in the model directory `results/example_model_name`.

## Postprocess
Add the helper functions in `postprocessing/compress_label.m` and `postprocessing/cross_parsing_label.m` to your filepath.
Run `postprocessing/kld_compression_lipschitz.m` to estimate irreversibilities from model latent means and training videos using a Ziv-Merhav estimator.

## Examples
Each script in `bin` runs a predefined experiment and saves to `postprocessing/full_results`:
* `test_cases.sh`: Checks installation.
Each of three results should have a model name that describes `postprocessing/full_results/<model-name>/trajectory.png`.
* `sim_tstep_vary.sh`, `sim_regime_vary.sh`, `exp_regime_vary.sh`: Replicate main results.
* `sim_hp_vary.sh`, `exp_hp_vary.sh`: Replicate results used for hyperparameter tuning.
* [Pre-trained results](https://doi.org/10.5281/zenodo.7734339) are deposited on Zenodo as `full_results.zip`.

## Help
```
python main.py example_model_name -h
python main_viz.py example_model_name -h
```
