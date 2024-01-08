import json
import os
import torch
import numpy as np
from disvae import init_specific_model
from disvae.models.losses import _reconstruction_loss
from utils.datasets import get_dataloaders


MODEL_FILENAME = "model.pt"
META_FILENAME = "specs.json"


# def pct_err(sample, recon_sample):
#     """
#     Returns sqrt(sum((sample-recon)^2))/sqrt(sum(sample^2))

#     Parameters
#     ----------
#     sample : input image
#     recon_sample : reconstruction of sample
#     """
#     recon_err = torch.sqrt(torch.sum(torch.square(torch.sub(sample, recon_sample)), (0, 1, 2, 3)))
#     sample_amp = torch.sqrt(torch.sum(torch.square(sample), (0, 1, 2, 3)))
#     return torch.div(recon_err, sample_amp)


def sample_vals(model, directory):
    """
    Save latent means, logvars, and percent errors to CSVs.

    Parameters
    ----------
    model : nn.Module
        Model.

    directory : str
        Directory where model is saved.
    """
    metadata = load_metadata(directory)
    dataset = metadata['dataset']
    # dataloader = get_dataloaders(dataset, batch_size=1, shuffle=False,
    #                              root=os.path.join('data', metadata['dataset']))
    dataloader = get_dataloaders(dataset,
                                 batch_size=metadata['eval_batchsize'],
                                 shuffle=False,
                                 root=os.path.join('data', metadata['dataset']))
    device = next(model.parameters()).device
    model.cpu()
    model.eval()

    sample = torch.stack([dataloader.dataset[i][0] for i in range(len(dataloader.dataset))], dim=0)
    with torch.no_grad():
        recons = model(sample)[0]
        latent_means, latent_logvars = model.encoder(sample)
    reconstruct_losses = torch.vmap(_reconstruction_loss)(sample.unsqueeze(1), recons.unsqueeze(1))

    np.savetxt(os.path.join(directory, "reconstruct_losses.csv"),
               reconstruct_losses.detach().numpy().transpose(),
               delimiter=",")
    np.savetxt(os.path.join(directory, "latent_means.csv"),
               latent_means.detach().numpy(),
               delimiter=",")
    np.savetxt(os.path.join(directory, "latent_logvars.csv"),
               latent_logvars.detach().numpy(),
               delimiter=",")
    
    model.to(device)

    # # pct_errs = []
    # reconstruct_losses = []
    # latent_means = []
    # latent_logvars = []

    # for i in range(len(dataloader.dataset)):
    #     sample = torch.stack([dataloader.dataset[i][0]], dim=0)
    #     # sample = sample.to(sample.device)

    #     with torch.no_grad(): recon_sample = model(sample)[0]
    #     reconstruct_loss = _reconstruction_loss(sample, recon_sample)
    #     # sample_pct_err = pct_err(sample, recon_sample)
    #     with torch.no_grad(): latent_mean, latent_logvar = model.encoder(sample)

    #     reconstruct_losses.append(reconstruct_loss.detach().numpy())
    #     # pct_errs.append(sample_pct_err.detach().numpy())
    #     latent_means.append(latent_mean[0].detach().numpy())
    #     latent_logvars.append(latent_logvar[0].detach().numpy())

    # np.savetxt(os.path.join(directory, "reconstruct_losses.csv"),
    #            np.array(reconstruct_losses), delimiter=",")
    # # np.savetxt(os.path.join(directory, "pct_errs.csv"),
    # #            np.array(pct_errs), delimiter=",")
    # np.savetxt(os.path.join(directory, "latent_means.csv"),
    #            np.array(latent_means), delimiter=",")
    # np.savetxt(os.path.join(directory, "latent_logvars.csv"),
    #            np.array(latent_logvars), delimiter=",")

    # model.to(device)


def save_model(model, directory, metadata=None, filename=MODEL_FILENAME):
    """
    Save a model and corresponding metadata.

    Parameters
    ----------
    model : nn.Module
        Model.

    directory : str
        Directory where model is saved.

    metadata : dict
        Metadata to save.
    """
    device = next(model.parameters()).device
    model.cpu()

    if metadata is None:
        # save the minimum required for loading
        metadata = dict(img_size=model.img_size, latent_dim=model.latent_dim),
                        # model_type=model.model_type)

    save_metadata(metadata, directory)
    path_to_model = os.path.join(directory, filename)
    torch.save(model.state_dict(), path_to_model)

    model.to(device)  #restore device


def load_metadata(directory, filename=META_FILENAME):
    """
    Load metadata of a model.

    Parameters
    ----------
    directory : string
        Directory where model is saved.
    """
    path_to_metadata = os.path.join(directory, filename)

    with open(path_to_metadata) as metadata_file:
        metadata = json.load(metadata_file)

    return metadata


def save_metadata(metadata, directory, filename=META_FILENAME, **kwargs):
    """
    Save metadata of a model.

    Parameters
    ----------
    metadata:
        Object to save.

    directory: string
        Directory where model is saved.

    kwargs:
        Additional arguments to `json.dump`.
    """
    path_to_metadata = os.path.join(directory, filename)

    with open(path_to_metadata, 'w') as f:
        json.dump(metadata, f, indent=4, sort_keys=True, **kwargs)


def load_model(directory, is_gpu=True, filename=MODEL_FILENAME):
    """
    Load a trained model.

    Parameters
    ----------
    directory : string
        Directory where model is saved.

    is_gpu : bool
        Whether to load on GPU is available.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and is_gpu
                          else "cpu")

    metadata = load_metadata(directory)
    img_size = metadata["img_size"]
    latent_dim = metadata["latent_dim"]
    model_type = 'Burgess'
    # model_type = metadata["model_type"]

    path_to_model = os.path.join(directory, filename)
    model = _get_model(model_type, img_size, latent_dim, device, path_to_model)
    return model


def _get_model(model_type, img_size, latent_dim, device, path_to_model):
    """
    Load a single model.

    Parameters
    ----------
    model_type : str
        The type of model to load, e.g., Burgess
    img_size : tuple
        Numbers of pixels in the image width and height.
        Only (64, 64) is implemented.
    latent_dim : int
        The number of latent dimensions in the bottleneck.
    device : str
        Either 'cuda' or 'cpu'
    path_to_model : str
        Full path to the saved model on the device.
    """
    model = init_specific_model(model_type, img_size, latent_dim).to(device)
    # works with state_dict to be independent of file structure
    model.load_state_dict(torch.load(path_to_model), strict=False)
    model.eval()

    return model
