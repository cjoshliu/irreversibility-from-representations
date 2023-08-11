import json
import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np
import os
import pandas as pd
from skimage import io
import torch
from utils.datasets import get_dataloaders


LATENT_MEANS = 'latent_means.csv'
PLOT_NAMES = dict(traversals='traversals.tif',
                  lattice='lattice.tif',
                  reconstruct='reconstruct.tif',
                  trajectory='trajectory.pdf')
TEST_LOSSES = 'test_losses.log'
TRAINING_VID = 'training.tif'


class Visualizer():
    def __init__(self, model, dataset, model_dir, center):
        """
        Class for generating visualizations.

        Parameters
        ----------
        model: disvae.vae.VAE
        dataset: str, name of dataset
        model_dir: str, directory where model is saved and where visualizations will be saved
        center: list, center of traversals and lattice
        """
        self.model = model
        self.device = next(self.model.parameters()).device
        self.latent_dim = self.model.latent_dim
        self.model_dir = model_dir
        self.dataset = dataset
        if len(center) == 1: self.origin = center*self.latent_dim
        elif len(center) == self.latent_dim: self.origin = center
        else: raise ValueError('Cannot broadcast center {} to length {}'.format(center, self.latent_dim))

    def traversals(self, n_cells, max_traversal, stack=False):
        """
        Traverse each latent dimension while fixing others at zero.

        Parameters
        ----------
        n_cells: int, steps per latent dimension
        max_traversal: float, maximum displacement from origin per latent dimension
        stack: bool (optional), whether to return stack of decoded latent points
        """
        device = next(self.model.parameters()).device
        self.model.cpu()
        steps = np.linspace(-1.0*max_traversal, max_traversal, n_cells)
        if stack: tiles = []
        else: tiles = np.zeros((self.latent_dim, n_cells), dtype=np.dtype('object'))
        
        with open(os.path.join(self.model_dir, TEST_LOSSES)) as f:
            losses = json.load(f)
        kl_keys = np.array([key for key in losses.keys() if 'kl_loss_' in key])
        kl_vals = np.array([losses[key] for key in kl_keys])
        sorted_keys = np.flip(kl_keys[np.argsort(kl_vals)])
        sorted_dims = np.array([int(key.replace('kl_loss_', '')) for key in sorted_keys])

        for i in range(self.latent_dim):
            for j in range(n_cells):
                latent_vec = self.origin.copy()
                latent_vec[sorted_dims[i]] += steps[j]
                latent_tensor = torch.tensor([latent_vec])
                with torch.no_grad(): recon = self.model.decoder(latent_tensor.float())
                if stack: tiles.append(phase(recon))
                else: tiles[i, j] = phase(recon)
        
        if stack: tiles = np.array(tiles).astype('uint8')
        else: tiles = np.block(tiles.tolist()).astype('uint8')
        io.imsave(os.path.join(self.model_dir, PLOT_NAMES['traversals']), tiles)
        self.model.to(device)

    def lattice(self, dims, n_cells, max_traversal, stack=False):
        """
        Decode a grid of lattice points in a 2D subspace of latent space.

        Parameters
        ----------
        dims: 1 x 2 array-like, latent dimensions spanning subspace
        n_cells: int, steps per latent dimension
        max_traversal: float, maximum displacement from origin per latent dimension
        stack: bool (optional), whether to return stack of decoded latent points
        """
        device = next(self.model.parameters()).device
        self.model.cpu()
        steps = np.linspace(-1.0*max_traversal, max_traversal, n_cells)
        if stack: tiles = []
        else: tiles = np.zeros((n_cells, n_cells), dtype=np.dtype('object'))

        for i in range(n_cells):
            for j in range(n_cells):
                latent_vec = self.origin.copy()
                latent_vec[dims[0]] += steps[i]
                latent_vec[dims[1]] += steps[j]
                latent_tensor = torch.tensor([latent_vec])
                with torch.no_grad(): recon = self.model.decoder(latent_tensor.float())
                if stack: tiles.append(phase(recon))
                else: tiles[i, j] = phase(recon)

        if stack: tiles = np.array(tiles).astype('uint8')
        else: tiles = np.block(tiles.tolist()).astype('uint8')
        io.imsave(os.path.join(self.model_dir, PLOT_NAMES['lattice']), tiles)
        self.model.to(device)

    def reconstruct(self, frames):
        """
        Reconstruct source video from dataset.

        Parameters
        ----------
        frames : 1 x 2 array-like, first and last indices to reconstruct as video
        """
        device = self.device
        self.model.cpu()
        dataloader = get_dataloaders(self.dataset,
                                     batch_size=1,
                                     shuffle=False,
                                     root=os.path.join('data', self.dataset))
        start_frame = frames[0]
        final_frame = min(frames[1], len(dataloader))
        recons = []

        for i in np.arange(start_frame, final_frame):
            sample = torch.stack([dataloader.dataset[i][0]])
            with torch.no_grad(): recon = self.model(sample)[0]
            recons.append(phase(recon))
        
        recons = np.array(recons).astype('uint8')
        io.imsave(os.path.join(self.model_dir, PLOT_NAMES['reconstruct']), recons)
        self.model.to(device)

    def trajectory(self, dims):
        """
        Plots projection of trajectory onto 2D subspace of latent space.

        Parameters
        ----------
        dims: 1 x 2 array-like, latent dimensions spanning subspace
        """
        df = pd.read_csv(os.path.join(self.model_dir, LATENT_MEANS), header=None)
        x, y  = np.array(df[dims[0]]), np.array(df[dims[1]])
        colors = pl.cm.viridis(np.linspace(0, 1, len(df)))

        plt.figure()
        plt.gca().set_aspect('equal')
        colorline(x, y)
        plt.scatter(x, y, c=colors)
        plt.xlabel('Latent dimension ' + str(dims[0]))
        plt.ylabel('Latent dimension ' + str(dims[1]))
        plt.savefig(os.path.join(self.model_dir, PLOT_NAMES["trajectory"]), bbox_inches='tight')
        plt.close()
    

class TrainingRecorder:
    def __init__(self, model, dataset, model_dir, recorded_sample=0):
        """
        Records reconstructions of a sample at each training epoch.

        Parameters
        ----------
        model: disvae.vae.VAE
        dataset: str, name of dataset
        model_dir: str, directory where model is saved and where visualizations will be saved
        recorded_sample: int, index of sample to record.
        """
        self.model = model
        self.device = next(self.model.parameters()).device
        self.latent_dim = self.model.latent_dim
        self.model_dir = model_dir
        self.dataset = dataset
        self.dataloader = get_dataloaders(self.dataset,
                                          batch_size=1,
                                          shuffle=False,
                                          root=os.path.join('data', self.dataset))
        self.sample = torch.stack([self.dataloader.dataset[recorded_sample][0]]).to(self.device)
        self.recons = []
    
    def __call__(self):
        """Reconstructs the first sample when called at the end of each epoch."""
        cached_training = self.model.training
        self.model.eval()
        with torch.no_grad(): recon = self.model(self.sample)[0].cpu()
        self.recons.append(phase(recon))
        if cached_training: self.model.train()

    def save_reset(self):
        """Saves and resets reconstructions."""
        self.recons = np.array(self.recons).astype('uint8')
        io.imsave(os.path.join(self.model_dir, TRAINING_VID), self.recons)
        self.recons = []


def phase(x):
    """
    Convert tensor of complex numbers into TIFF phase field format.

    Parameters
    ----------
    x: torch.Tensor, stack of real and imaginary parts

    Returns
    -------
    phi: np.ndarray, phases of x with [-pi, pi) -> [0, 255]
    """
    if type(x) != torch.Tensor: raise TypeError("Input must be a torch.Tensor")
    if len(x.size()) == 4: # if x is a batch
        real, imag = x[0]
    elif len(x.size()) == 3: # if x is a single sample
        real, imag = x
    else: raise ValueError("Input must have real and imaginary layers")

    real = real.detach().numpy()*2.0-1.0
    imag = imag.detach().numpy()*2.0-1.0
    comp = real+1j*imag
    phi = np.rint((np.angle(comp)+np.pi)/2.0/np.pi*255.0)
    return phi


def make_segments(x, y):
    """
    Create list of line segments formatted for LineCollection.
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def colorline(x, y, z=None, cmap=pl.cm.viridis, norm=plt.Normalize(0.0, 1.0), linewidth=2, alpha=0.5):
    """
    Plot a colored line with coordinates x and y

    Note
    ----
    From matplotlib examples: http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    """

    # Equally space default colors on [0,1]
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case of `z` a single number
    if not hasattr(z, "__iter__"):  # check for numerical input
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)
    ax = plt.gca()
    ax.add_collection(lc)

    return lc