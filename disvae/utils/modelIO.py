import json
import os
import re
import torch
import numpy as np
from disvae import init_specific_model
from disvae.models.losses import _reconstruction_loss
from utils.datasets import get_dataloaders

MODEL_FILENAME = "model.pt"
META_FILENAME = "specs.json"


def pct_err(sample, recon_sample):
    """
    Parameters
    ----------
    sample : input image
    recon_sample : reconstruction of sample

    Returns
    -------
    sqrt(sum((sample-recon)^2))/sqrt(sum(sample^2))
    """
    recon_err = torch.sqrt(torch.sum(torch.square(torch.sub(sample, recon_sample)), (0, 1, 2, 3)))
    sample_amp = torch.sqrt(torch.sum(torch.square(sample), (0, 1, 2, 3)))
    return torch.div(recon_err, sample_amp)


def sample_vals(model, directory):
    """
    Save latent means and logvars to CSV

    Parameters
    ----------
    model : nn.Module
        Model

    directory : str
        Path to the directory wherein to save CSV
    """
    metadata = load_metadata(directory)
    dataset = metadata['dataset']
    dataloader = get_dataloaders(dataset, batch_size=1, shuffle=False,
                                 root=os.path.join('data/', metadata['dataset']))
    total_frames = len(dataloader.dataset)
    device = next(model.parameters()).device
    model.cpu()

    pct_errs = []
    reconstruct_losses = []
    latent_means = []
    latent_logvars = []

    for i in range(total_frames):
        sample = torch.stack([dataloader.dataset[i][0]], dim=0)
        sample = sample.to(sample.device)

        recon_sample, _, _ = model(sample)
        reconstruct_loss = _reconstruction_loss(sample, recon_sample)
        sample_pct_err = pct_err(sample, recon_sample)
        latent_mean, latent_logvar = model.encoder(sample)

        reconstruct_losses.append(reconstruct_loss.detach().numpy())
        pct_errs.append(sample_pct_err.detach().numpy())
        latent_means.append(latent_mean[0].detach().numpy())
        latent_logvars.append(latent_logvar[0].detach().numpy())

    np.savetxt(os.path.join(directory, "reconstruct_losses.csv"),
               np.array(reconstruct_losses), delimiter=",")
    np.savetxt(os.path.join(directory, "pct_errs.csv"),
               np.array(pct_errs), delimiter=",")
    np.savetxt(os.path.join(directory, "latent_means.csv"),
               np.array(latent_means), delimiter=",")
    np.savetxt(os.path.join(directory, "latent_logvars.csv"),
               np.array(latent_logvars), delimiter=",")

    model.to(device)


def save_model(model, directory, metadata=None, filename=MODEL_FILENAME):
    """
    Save a model and corresponding metadata.

    Parameters
    ----------
    model : nn.Module
        Model.

    directory : str
        Path to the directory where to save the data.

    metadata : dict
        Metadata to save.
    """
    device = next(model.parameters()).device
    model.cpu()

    if metadata is None:
        # save the minimum required for loading
        metadata = dict(img_size=model.img_size, latent_dim=model.latent_dim,
                        model_type=model.model_type)

    save_metadata(metadata, directory)

    path_to_model = os.path.join(directory, filename)
    torch.save(model.state_dict(), path_to_model)

    model.to(device)  # restore device


def load_metadata(directory, filename=META_FILENAME):
    """Load the metadata of a training directory.

    Parameters
    ----------
    directory : string
        Path to folder where model is saved. For example './experiments/mnist'.
    """
    path_to_metadata = os.path.join(directory, filename)

    with open(path_to_metadata) as metadata_file:
        metadata = json.load(metadata_file)

    return metadata


def save_metadata(metadata, directory, filename=META_FILENAME, **kwargs):
    """Load the metadata of a training directory.

    Parameters
    ----------
    metadata:
        Object to save

    directory: string
        Path to folder where to save model. For example './experiments/mnist'.

    kwargs:
        Additional arguments to `json.dump`
    """
    path_to_metadata = os.path.join(directory, filename)

    with open(path_to_metadata, 'w') as f:
        json.dump(metadata, f, indent=4, sort_keys=True, **kwargs)


def load_model(directory, is_gpu=True, filename=MODEL_FILENAME):
    """Load a trained model.

    Parameters
    ----------
    directory : string
        Path to folder where model is saved. For example './experiments/mnist'.

    is_gpu : bool
        Whether to load on GPU is available.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and is_gpu
                          else "cpu")

    metadata = load_metadata(directory)
    img_size = metadata["img_size"]
    latent_dim = metadata["latent_dim"]
    model_type = metadata["model_type"]

    path_to_model = os.path.join(directory, filename)
    model = _get_model(model_type, img_size, latent_dim, device, path_to_model)
    return model


def load_checkpoints(directory, is_gpu=True):
    """Load all checkpoint models.

    Parameters
    ----------
    directory : string
        Path to folder where model is saved. For example './experiments/mnist'.

    is_gpu : bool
        Whether to load on GPU .
    """
    checkpoints = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            results = re.search(r'.*?-([0-9].*?).pt', filename)
            if results is not None:
                epoch_idx = int(results.group(1))
                model = load_model(root, is_gpu=is_gpu, filename=filename)
                checkpoints.append((epoch_idx, model))

    return checkpoints


def _get_model(model_type, img_size, latent_dim, device, path_to_model):
    """ Load a single model.

    Parameters
    ----------
    model_type : str
        The name of the model to load. For example Burgess.
    img_size : tuple
        Numbers of pixels in the image width and height.
        For example (32, 32) or (64, 64).
    latent_dim : int
        The number of latent dimensions in the bottleneck.

    device : str
        Either 'cuda' or 'cpu'
    path_to_model : str
        Full path to the saved model on the device.
    """
    model = init_specific_model(model_type, img_size, latent_dim).to(device)
    # works with state_dict to make it independent of the file structure
    model.load_state_dict(torch.load(path_to_model), strict=False)
    model.eval()

    return model


def numpy_serialize(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))


def save_np_arrays(arrays, directory, filename):
    """Save dictionary of arrays in json file."""
    save_metadata(arrays, directory, filename=filename, default=numpy_serialize)


def load_np_arrays(directory, filename):
    """Load dictionary of arrays from json file."""
    arrays = load_metadata(directory, filename=filename)
    return {k: np.array(v) for k, v in arrays.items()}
