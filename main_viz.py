import argparse
import os
import sys
from utils.helpers import FormatterNoDuplicate, set_seed, get_config_section
from utils.visualize import Visualizer
from main import RES_DIR
from disvae.utils.modelIO import load_model, load_metadata


CONFIG_FILE = 'main_viz.ini'
PLOT_TYPES = ['traversals', 'lattice', 'reconstruct', 'trajectory', 'all']


def parse_arguments(args_to_parse):
    """
    Parse command line arguments

    Parameters
    ----------
    args_to_parse: list of str, arguments to parse (split on whitespace)
    """
    default_config = get_config_section([CONFIG_FILE], "Custom")
    description = "Module for plotting trained VAE and FVAE models."
    parser = argparse.ArgumentParser(description=description, formatter_class=FormatterNoDuplicate)

    parser.add_argument('name', type=str,
                        help='Name of the model for storing and loading purposes.')
    parser.add_argument('plots', type=str, nargs='+', choices=PLOT_TYPES,
                        help='List of plots to generate. `traversals` traverses each latent dimension while keeping others at zero. `lattice` decodes lattice points in a 2D subspace of latent space. `reconstruct` reconstructs the CGL or Rho video. `trajectory` plots the projection of the latent trajectory onto a 2D subspace of latent space. `all` plots all of the above.')
    parser.add_argument('-s', '--seed', type=int, default=default_config['seed'],
                        help='Randomization seed. Can be `None` for stochastic behavior.')
    parser.add_argument('-z', '--dims', type=int, nargs=2, default=default_config['dims'],
                        help='Indices of two dimensions spanning subspace containing lattice points or trajectory projection.')
    parser.add_argument('-f', '--frames', type=int, nargs=2, default=default_config['frames'],
                        help='First and last indices to reconstruct as video.')
    parser.add_argument('-c', '--center', type=float, nargs='+', default=default_config['center'],
                        help='Center of traversals or lattice. Must be broadcastable to length of latent vector.')
    parser.add_argument('-t', '--max-traversal', type=float, default=default_config['max_traversal'],
                        help='Maximum displacement from center per latent dimension.')
    parser.add_argument('-n', '--n-cells', type=int, default=default_config['n_cells'],
                        help='Steps per latent dimension.')
    parser.add_argument('--stack', action='store_true', help='Return traversals and lattice as a stack of tiles.')

    args = parser.parse_args()
    return args


def main(args):
    """
    Main plotting function.

    Parameters
    ----------
    args: argparse.Namespace, parsed arguments
    """
    set_seed(args.seed)
    model_dir = os.path.join(RES_DIR, args.name)
    metadata = load_metadata(model_dir)
    model = load_model(model_dir)
    model.eval()
    dataset = metadata['dataset']
    viz = Visualizer(model, dataset, model_dir, args.center)

    if "all" in args.plots:
        args.plots = [p for p in PLOT_TYPES if p != "all"]

    for plot_type in args.plots:
        if plot_type == 'traversals':
            viz.traversals(args.n_cells, args.max_traversal, stack=args.stack)
        elif plot_type == 'lattice':
            viz.lattice(args.dims, args.n_cells, args.max_traversal, stack=args.stack)
        elif plot_type == 'reconstruct':
            viz.reconstruct(args.frames)
        elif plot_type == 'trajectory':
            viz.trajectory(args.dims)
        else: raise ValueError("Unknown plot: {}".format(plot_type))


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
