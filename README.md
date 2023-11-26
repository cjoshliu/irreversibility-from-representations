# Irreversibility from representations [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/cjoshliu/irreversibility-from-representations/blob/main/LICENSE) [![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7737963.svg)](https://doi.org/10.5281/zenodo.7737963)


This repository contains code for simulating complex Ginzburg-Landau (CGL) phase fields, training models to represent CGL and Rho dynamics in a low-dimensional latent space, and bounding irreversibility from latent-space representations using the Ziv-Merhav estimator.
It is modified from [simulator code by D.M. Winterbottom](https://github.com/codeinthehole/codeinthehole.com/blob/58ad3d28ddefb64350ec883b291d4dbe1df096f7/www/static/tutorial/files/CGLsim2D.m), [VAE code by Y. Dubois](https://github.com/YannDubs/disentangling-vae), and [VAE code by E. Dupont](https://doi.org/10.48550/arXiv.1804.00104).

The simulator uses methods described in ["Exponential time differencing for stiff systems"](https://doi.org/10.1006/jcph.2002.6995).
The default VAE uses the architecture and loss described in ["Understanding disentangling in β-VAE"](https://doi.org/10.48550/arXiv.1804.03599) and ["Disentangling by factorising"](https://doi.org/10.48550/arXiv.1802.05983), respectively, though the standard VAE loss from ["Auto-encoding variational Bayes"](https://doi.org/10.48550/arXiv.1312.6114) is also implemented.
The ZM estimator uses methods described in ["Dissipation: the phase-space perspective"](https://doi.org/10.1103/PhysRevLett.98.080602) and ["Entropy production and Kullback-Leibler divergence between stationary trajectories of discrete systems."](https://doi.org/10.1103/PhysRevE.85.031129)

Table of Contents:
1. [Install](#install)
2. [Preprocess](#preprocess)
3. [Train](#train)
4. [Plot](#plot)
5. [Postprocess](#postprocess)
6. [Examples](#examples)

## Install
```
git clone https://github.com/cjoshliu/irreversibility-from-representations your/path
cd your/path
mkdir -p {data/cgle64,results,postprocessing/results}
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

## Preprocess
Launch `preprocessing/vid_to_dset.ipynb` and set the location of saved [`raw_data`](https://doi.org/10.5281/zenodo.7734339) to `SRC_DIR` in the first cell. Set `DST_DIR` to a location suitable for saving large datasets.
Alternatively, run `preprocessing/SimCGL.m` to generate your own phase-field videos.
Modify `preprocessing/sim_noise.ipynb` to add simulated measurement noise and `preprocessing/vid_to_dset.ipynb` to make datasets.

## Train
Copy dataset to `data/cgle64/img_align_cgle64`. For example:
```
cp -r "$DATASET_DIR"/sim_regime_vary/c1_-020_c2_050_dT_010_s_0 data/cgle64/img_align_cgle64
```
Run `python main.py <model-name> <param>` to train and/or evaluate a model. For example:
```
python main.py example_model_name --lr 0.001 -e 3
```
Results are generated in `results`, but should be moved to `postprocessing/results` after each experiment.

See below for details:
```
usage: main.py [-h] [-L {CRITICAL,ERROR,WARNING,INFO,DEBUG,NOTSET}] [--no-progress-bar] [--no-cuda] [-s SEED]
               [--checkpoint-every CHECKPOINT_EVERY] [-d {cgle64}] [-e EPOCHS] [-b BATCH_SIZE] [-c CUTOFF [CUTOFF ...]]
               [-i RECORDED_SAMPLE] [--lr LR] [--lr-disc LR_DISC] [-z LATENT_DIM] [-l {VAE,factor}] [--is-eval-only]
               [--no-test] [--eval-batchsize EVAL_BATCHSIZE]
               name

PyTorch implementation of VAE and FVAE.

options:
  -h, --help            show this help message and exit

General options:
  name                  Name of the model for storing and loading purposes.
  -L, --log-level {CRITICAL,ERROR,WARNING,INFO,DEBUG,NOTSET}
                        Logging levels. (default: info)
  --no-progress-bar     Disables progress bar. (default: False)
  --no-cuda             Disables CUDA training, even when a GPU is detected. (default: False)
  -s, --seed SEED       Randomization seed. Can be `None` for stochastic behavior. (default: 1234)

Training options:
  --checkpoint-every CHECKPOINT_EVERY
                        Inverse frequency in epochs at which model is saved. (default: 2000)
  -d, --dataset {cgle64}
                        Training dataset. (default: cgle64)
  -e, --epochs EPOCHS   Maximum number of epochs to train for. (default: 2000)
  -b, --batch-size BATCH_SIZE
                        Maximum batch size for training. (default: 10001)
  -c, --cutoff CUTOFF [CUTOFF ...]
                        Window of epochs for loss regression, number of epochs to plateau before break. (default: [16, 16])
  -i, --recorded-sample RECORDED_SAMPLE
                        Index of sample for which to record reconstructions at each epoch. (default: None)
  --lr LR               VAE learning rate. (default: 0.001)
  --lr-disc LR_DISC     Learning rate of factor VAE discriminator. (default: 0.0001)

Model options:
  -z, --latent-dim LATENT_DIM
                        Dimensionality of latent space. (default: 2)
  -l, --loss {VAE,factor}
                        Type of VAE loss function to use. (default: factor)

Evaluation options:
  --is-eval-only        Whether to evaluate using the precomputed model `name`. (default: False)
  --no-test             Whether not to compute the test losses.` (default: False)
  --eval-batchsize EVAL_BATCHSIZE
                        Maximum batch size for evaluation. (default: 10001)
```

### Output
* **model.pt**: Model at the end of training. 
* **model-**`i`**.pt**: Model checkpoint after `i` iterations.
* **specs.json**: Parameters used to run `main.py`.
* **training.tif**: TIFF stack of a sample reconstructed at each training epoch. Only generated if `--reconstructed-sample` is not none.
* **train_losses.log**: All (sub-)losses computed during training.
* **test_losses.log**: All (sub-)losses computed at the end of training with the model in evaluate mode.
* **reconstruct_losses.csv**: Reconstruction loss of each sample at end of training.
* **latent_logvars.csv**: Model-generated latent log-variances for each observation in training set.
* **latent_means.csv**: Model-generated latent means for each observation in training set. These can be used as Ziv-Merhav irreversibility-estimator inputs.

## Plot

Run `python main_viz.py <model-name> <plot_types> <param>` to plot using pretrained models in `results`. For example:
```
python main_viz.py example_model_name all
```
This will save plots in the model directory `results/example_model_name`. Make sure the correct dataset is in `data/cgle64/img_align_cgle64` before running!

See below for details:
```
usage: main_viz.py [-h] [-s SEED] [-z DIMS DIMS] [-f FRAMES FRAMES] [-c CENTER [CENTER ...]]
                   [-t MAX_TRAVERSAL] [-n N_CELLS] [--stack]
                   name {traversals,lattice,reconstruct,trajectory,all}
                   [{traversals,lattice,reconstruct,trajectory,all} ...]

Module for plotting trained VAE and FVAE models.

positional arguments:
  name                  Name of the model for storing and loading purposes.
  {traversals,lattice,reconstruct,trajectory,all}
                        List of plots to generate. `traversals` traverses each latent dimension
                        while keeping others at zero. `lattice` decodes lattice points in a 2D
                        subspace of latent space. `reconstruct` reconstructs the CGL or Rho video.
                        `trajectory` plots the projection of the latent trajectory onto a 2D
                        subspace of latent space. `all` plots all of the above.

options:
  -h, --help            show this help message and exit
  -s, --seed SEED       Randomization seed. Can be `None` for stochastic behavior. (default: 1234)
  -z, --dims DIMS DIMS  Indices of two dimensions spanning subspace containing lattice points or
                        trajectory projection. (default: [0, 1])
  -f, --frames FRAMES FRAMES
                        First and last indices to reconstruct as video. (default: [0, 300])
  -c, --center CENTER [CENTER ...]
                        Center of traversals or lattice. Must be broadcastable to length of latent
                        vector. (default: [0])
  -t, --max-traversal MAX_TRAVERSAL
                        Maximum displacement from center per latent dimension. (default: 2.0)
  -n, --n-cells N_CELLS
                        Steps per latent dimension. (default: 9)
  --stack               Return traversals and lattice as a stack of tiles. (default: False)
```

## Postprocess
Use `postprocessing/kld_compression_lipschitz.m` to estimate irreversibilities from model latent means and training videos using a Ziv-Merhav estimator.
Change paths at the top of the script to estimate irreversibility for a selected result.
The ruler is provided in [`raw_data`](https://doi.org/10.5281/zenodo.7734339).

## Examples
Before running scripts in `bin`, set `DATASET_DIR` to the same location as `DST_DIR` from [preprocessing](#preprocess).
Each script in `bin` runs a predefined experiment and saves to `postprocessing/results`:
* `test_cases.sh`: Checks installation. Each of two results should have a name that describes `<name>/trajectory.pdf`.
* `sim_tstep_vary.sh`, `sim_noise_vary.sh`, `sim_regime_vary.sh`: Main simulation results, varying timestep, measurement noise, and CGL regime, respectively.
* `exp_atp_vary.sh`, `exp_orderp_vary.sh`: Main experiment results, varying metabolic depletion and Rho pattern stability, respectively.
* `sim_hyperp_vary.sh`, `exp_hyperp_vary.sh`: Results used for hyperparameter tuning.
* `regime_pool.sh`, `sim_z_vary.sh`: Additional results used in supplemental materials.
