import argparse
import os
import sys
from utils.helpers import FormatterNoDuplicate, check_bounds, set_seed, get_config_section
from utils.visualize import Visualizer
from utils.viz_helpers import get_samples
from main import RES_DIR
from disvae.utils.modelIO import load_model, load_metadata

CONFIG_FILE = 'main_viz.ini'
PLOT_TYPES = ['generate-samples', 'data-samples', 'reconstruct', 'traversals',
              'reconstruct-traverse', 'gif-traversals', 'interpolate', 'trajectory',
              'reconstruct-vid', 'vid-traversals', 'all']


def parse_arguments(args_to_parse):
    """Parse the command line arguments.

    Parameters
    ----------
    args_to_parse: list of str
        Arguments to parse (splitted on whitespaces).
    """
    default_config = get_config_section([CONFIG_FILE], "Custom")

    description = "CLI for plotting using pretrained models of `disvae`"

    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=FormatterNoDuplicate)

    parser.add_argument('name', type=str,
                        help="Name of the model for storing and loading purposes.")
    parser.add_argument("plots", type=str, nargs='+', choices=PLOT_TYPES,
                        help="List of all plots to generate. `generate-samples`: random decoded samples. `data-samples` samples from the dataset. `reconstruct` first rnows//2 will be the original and rest will be the corresponding reconstructions. `traversals` traverses the most important rnows dimensions with ncols different samples from the prior or posterior. `reconstruct-traverse` first row for original, second are reconstructions, rest are traversals. `gif-traversals` grid of gifs where rows are latent dimensions, columns are examples, each gif shows posterior traversals. `all` runs every plot.")
    parser.add_argument('-s', '--seed', type=int, default=default_config['seed'],
                        help='Random seed. Can be `None` for stochastic behavior.')
    parser.add_argument('-r', '--n-rows', type=int, default=default_config['n_rows'],
                        help='The number of rows to visualize (if applicable).')
    parser.add_argument('-c', '--n-cols', type=int, default=default_config['n_cols'],
                        help='The number of columns to visualize (if applicable).')
    parser.add_argument('-t', '--max-traversal', default=default_config['max_traversal'],
                        type=lambda v: check_bounds(v, lb=0, is_inclusive=False,
                                                    type=float, name="max-traversal"),
                        help='The maximum displacement induced by a latent traversal. Symmetrical traversals are assumed. If `m>=0.5` then uses absolute value traversal, if `m<0.5` uses a percentage of the distribution (quantile). E.g. for the prior the distribution is a standard normal so `m=0.45` corresponds to an absolute value of `1.645` because `2m=90%%` of a standard normal is between `-1.645` and `1.645`. Note in the case of the posterior, the distribution is not standard normal anymore.')
    parser.add_argument('-i', '--idcs', type=int, nargs='+', default=default_config['idcs'],
                        help='List of indices to of images to put at the beginning of the samples.')
    parser.add_argument('-z', '--dims', type=int, nargs='+', default=default_config['dims'],
                        help='Dimensions to plot in trajectory or interpolate.')
    parser.add_argument('-f', '--frames', type=int, nargs='+', default=default_config['frames'],
                        help='First and last indices to reconstruct as video.')
    parser.add_argument('-u', '--upsample-factor', default=default_config['upsample_factor'],
                        type=lambda v: check_bounds(v, lb=1, is_inclusive=True,
                                                    type=int, name="upsample-factor"),
                        help='The scale factor with which to upsample the image (if applicable).')
    parser.add_argument('--is-show-loss', action='store_true',
                        help='Displays the loss on the figures (if applicable).')
    parser.add_argument('--is-posterior', action='store_true',
                        help='Traverses the posterior instead of the prior.')
    args = parser.parse_args()

    return args


def main(args):
    """Main function for plotting from pretrained models.

    Parameters
    ----------
    args: argparse.Namespace
        Arguments
    """
    set_seed(args.seed)
    experiment_name = args.name
    model_dir = os.path.join(RES_DIR, experiment_name)
    meta_data = load_metadata(model_dir)
    model = load_model(model_dir)
    model.eval()  # don't sample from latent: use mean
    dataset = meta_data['dataset']
    viz = Visualizer(model=model,
                     model_dir=model_dir,
                     dataset=dataset,
                     max_traversal=args.max_traversal,
                     loss_of_interest='kl_loss_',
                     upsample_factor=args.upsample_factor)
    size = (args.n_rows, args.n_cols)
    # same samples for all plots: sample max then take first `x`data  for all plots
    num_samples = args.n_cols * args.n_rows
    samples = get_samples(dataset, num_samples, idcs=args.idcs)

    if "all" in args.plots:
        args.plots = [p for p in PLOT_TYPES if p != "all"]

    for plot_type in args.plots:
        if plot_type == 'generate-samples':
            viz.generate_samples(size=size)
        elif plot_type == 'data-samples':
            viz.data_samples(samples, size=size)
        elif plot_type == "reconstruct":
            viz.reconstruct(samples, size=size)
        elif plot_type == 'traversals':
            viz.traversals(data=samples[0:1, ...] if args.is_posterior else None,
                           n_per_latent=args.n_cols,
                           n_latents=args.n_rows,
                           is_reorder_latents=True)
        elif plot_type == "reconstruct-traverse":
            viz.reconstruct_traverse(samples,
                                     is_posterior=args.is_posterior,
                                     n_latents=args.n_rows,
                                     n_per_latent=args.n_cols,
                                     is_show_text=args.is_show_loss)
        elif plot_type == "gif-traversals":
            viz.gif_traversals(samples[:args.n_cols, ...], n_latents=args.n_rows)
        elif plot_type == "interpolate":
            viz.interpolate(n_rows=args.n_rows, n_cols=args.n_cols, dims=args.dims)
        elif plot_type == "trajectory":
            viz.trajectory(dims=args.dims)
        elif plot_type == 'reconstruct-vid':
            viz.reconstruct_vid(frames=args.frames)
        elif plot_type == 'vid-traversals':
            viz.vid_traversals(frames=args.frames, dims=args.dims)
        else:
            raise ValueError("Unknown plot_type={}".format(plot_type))


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
