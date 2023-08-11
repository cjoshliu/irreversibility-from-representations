import argparse
import logging
import sys
import os
from torch import optim
from disvae import init_specific_model, Trainer, Evaluator
from disvae.utils.modelIO import save_model, load_model, load_metadata, sample_vals
from disvae.models.losses import LOSSES, get_loss_f
from utils.datasets import get_dataloaders, get_img_size, DATASETS
from utils.helpers import (create_safe_directory, get_device, set_seed, get_n_param,
                           get_config_section, FormatterNoDuplicate)
from utils.visualize import TrainingRecorder


CONFIG_FILE = "main.ini"
RES_DIR = "results"
LOG_LEVELS = list(logging._levelToName.values())


def parse_arguments(args_to_parse):
    """
    Parse command-line arguments.

    Parameters
    ----------
    args_to_parse: list of str
        Arguments to parse (split on whitespace).
    """
    default_config = get_config_section([CONFIG_FILE], "Custom")

    description = "PyTorch implementation of VAE and FVAE."
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=FormatterNoDuplicate)

    # General options
    general = parser.add_argument_group('General options')
    general.add_argument('name', type=str,
                         help="Name of the model for storing and loading purposes.")
    general.add_argument('-L', '--log-level', help='Logging levels.',
                         default=default_config['log_level'], choices=LOG_LEVELS)
    general.add_argument('--no-progress-bar', action='store_true',
                         default=default_config['no_progress_bar'],
                         help='Disables progress bar.')
    general.add_argument('--no-cuda', action='store_true',
                         default=default_config['no_cuda'],
                         help='Disables CUDA training, even when a GPU is detected.')
    general.add_argument('-s', '--seed', type=int, default=default_config['seed'],
                         help='Randomization seed. Can be `None` for stochastic behavior.')

    # Training options
    training = parser.add_argument_group('Training options')
    training.add_argument('--checkpoint-every',
                          type=int, default=default_config['checkpoint_every'],
                          help='Inverse frequency in epochs at which model is saved.')
    training.add_argument('-d', '--dataset', help='Training dataset.',
                          default=default_config['dataset'], choices=DATASETS)
    training.add_argument('-e', '--epochs', type=int,
                          default=default_config['epochs'],
                          help='Maximum number of epochs to train for.')
    training.add_argument('-b', '--batch-size', type=int,
                          default=default_config['batch_size'],
                          help='Maximum batch size for training.')
    training.add_argument('-c', '--cutoff', type=int, nargs='+', default=default_config['cutoff'],
                          help='Window of epochs for loss regression, number of epochs to plateau before break.')
    training.add_argument('-i', '--recorded-sample', type=int, default=default_config['recorded_sample'],
                          help='Index of sample for which to record reconstructions at each epoch.')
    training.add_argument('--lr', type=float, default=default_config['lr'],
                          help='VAE learning rate.')
    training.add_argument('--lr-disc', type=float, default=default_config['lr_disc'],
                          help='Learning rate of factor VAE discriminator.')

    # Model options
    model = parser.add_argument_group('Model options')
    model.add_argument('-z', '--latent-dim', type=int,
                       default=default_config['latent_dim'],
                       help='Dimensionality of latent space.')
    model.add_argument('-l', '--loss',
                       default=default_config['loss'], choices=LOSSES,
                       help="Type of VAE loss function to use.")

    # Evaluation options
    evaluation = parser.add_argument_group('Evaluation options')
    evaluation.add_argument('--is-eval-only', action='store_true',
                            default=default_config['is_eval_only'],
                            help='Whether to evaluate using the precomputed model `name`.')
    evaluation.add_argument('--no-test', action='store_true',
                            default=default_config['no_test'],
                            help="Whether not to compute the test losses.`")
    evaluation.add_argument('--eval-batchsize', type=int,
                            default=default_config['eval_batchsize'],
                            help='Maximum batch size for evaluation.')

    args = parser.parse_args(args_to_parse)

    return args


def main(args):
    """
    Main training and evaluation function.

    Parameters
    ----------
    args: argparse.Namespace
        Parsed arguments.
    """
    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s', "%H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.setLevel(args.log_level.upper())
    stream = logging.StreamHandler()
    stream.setLevel(args.log_level.upper())
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    set_seed(args.seed)
    device = get_device(is_gpu=not args.no_cuda)
    exp_dir = os.path.join(RES_DIR, args.name)
    logger.info('Root directory for saving and loading experiments: {}'.format(exp_dir))

    if not args.is_eval_only:
        create_safe_directory(exp_dir, logger=logger)

        # prepare data
        train_loader = get_dataloaders(args.dataset,
                                       batch_size=args.batch_size,
                                       logger=logger)
        if len(train_loader.dataset) < args.batch_size:
            args.batch_size = len(train_loader.dataset)
        if len(train_loader.dataset) < args.eval_batchsize:
            args.eval_batchsize = len(train_loader.dataset)
        logger.info('Train {} with {} samples'.format(args.dataset, len(train_loader.dataset)))

        # prepare model
        args.img_size = get_img_size(args.dataset)  # stores for metadata
        model = init_specific_model('Burgess', args.img_size, args.latent_dim)
        logger.info('Num parameters in model: {}'.format(get_n_param(model)))

        # train model
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        model = model.to(device)  # make sure trainer and visualizer on same device
        if args.recorded_sample is not None:
            training_recorder = TrainingRecorder(model, args.dataset, exp_dir,
                                                 recorded_sample=args.recorded_sample)
        else: training_recorder = None
        loss_f = get_loss_f(args.loss,
                            n_data=len(train_loader.dataset),
                            device=device,
                            **vars(args))
        trainer = Trainer(model, optimizer, loss_f,
                          device=device,
                          logger=logger,
                          save_dir=exp_dir,
                          is_progress_bar=not args.no_progress_bar,
                          training_recorder=training_recorder,
                          epochs=args.epochs)
        trainer(train_loader,
                cutoff=args.cutoff,
                checkpoint_every=args.checkpoint_every)
        args.epochs = trainer.epochs

        # save model and metadata
        save_model(trainer.model, exp_dir, metadata=vars(args))
        sample_vals(trainer.model, exp_dir)

    if not args.no_test:
        model = load_model(exp_dir, is_gpu=not args.no_cuda)
        metadata = load_metadata(exp_dir)
        test_loader = get_dataloaders(metadata['dataset'],
                                      batch_size=args.eval_batchsize,
                                      shuffle=False,
                                      logger=logger)
        loss_f = get_loss_f(args.loss,
                            n_data=len(test_loader.dataset),
                            device=device,
                            **vars(args))
        evaluator = Evaluator(model, loss_f,
                              device=device,
                              logger=logger,
                              save_dir=exp_dir,
                              is_progress_bar=not args.no_progress_bar)
        evaluator(test_loader, is_losses=not args.no_test)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
