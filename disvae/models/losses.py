"""
Module containing losses.
"""
import abc
from .discriminator import Discriminator
import torch
from torch import optim
from torch.nn import functional as F


LOSSES = ["VAE", "factor"]
RECON_DIST = ["bernoulli"]


def get_loss_f(loss_name, **kwargs_parse):
    """Return the correct loss function given the argparse arguments."""
    if loss_name == "VAE":
        return VAELoss()
    elif loss_name == "factor":
        return FactorKLoss(kwargs_parse["device"],
                           gamma=1,
                           disc_kwargs=dict(latent_dim=kwargs_parse["latent_dim"]),
                           optim_kwargs=dict(lr=kwargs_parse["lr_disc"], betas=(0.5, 0.9)))
    else:
        assert loss_name not in LOSSES
        raise ValueError("Unknown loss: {}".format(loss_name))


class BaseLoss(abc.ABC):
    """
    Base class for losses.

    Parameters
    ----------
    record_loss_every : int, optional
        Inverse frequency in epochs at which to log losses.

    rec_dist : {"bernoulli"}, optional
        Reconstruction distribution of the likelihood on each pixel.
        Implicitly defines the reconstruction loss.
        Bernoulli corresponds to a binary cross entropy.

    steps_anneal : int, optional
        Number of annealing steps over which to add regularization.
    """

    def __init__(self, record_loss_every=1, rec_dist="bernoulli", steps_anneal=0):
        self.n_train_steps = 0
        self.record_loss_every = record_loss_every
        self.rec_dist = rec_dist
        self.steps_anneal = steps_anneal

    @abc.abstractmethod
    def __call__(self, data, recon_data, latent_dist, is_train, storer, **kwargs):
        """
        Calculates loss for a batch of data.

        Parameters
        ----------
        data : torch.Tensor
            Input data (e.g. batch of images).
            Shape: (batch_size, n_chan, height, width).

        recon_data : torch.Tensor
            Reconstructed data.
            Shape: (batch_size, n_chan, height, width).

        latent_dist : tuple of torch.tensor
            Latent means and log-variances of data, each of shape (batch_size, latent_dim).

        is_train : bool
            Whether currently in train mode.

        storer : dict
            Dictionary in which to store important variables for visualization.

        kwargs:
            Loss-specific arguments.
        """

    def _pre_call(self, is_train, storer):
        if is_train:
            self.n_train_steps += 1

        if not is_train or self.n_train_steps % self.record_loss_every == 1 or self.record_loss_every == 1:
            storer = storer
        else:
            storer = None

        return storer


class VAELoss(BaseLoss):
    """
    Compute the standard VAE loss from [1]:

    Parameters
    ----------
    kwargs:
        Additional arguments for `BaseLoss`, e.g., `rec_dist`.

    Reference
    ---------
    [1] D Kingma and M Welling. "Auto-encoding variational Bayes,"
        in Proceedings of the International Conference on Learning Representations 2014
        (ICLR, Banff, Canada, 2014), 10.48550/arXiv.1312.6114.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self, data, recon_data, latent_dist, is_train, storer, **kwargs):
        storer = self._pre_call(is_train, storer)
        rec_loss = _reconstruction_loss(data, recon_data,
                                        storer=storer,
                                        distribution=self.rec_dist)
        kl_loss = _kl_normal_loss(*latent_dist, storer)
        anneal_reg = (linear_annealing(0, 1, self.n_train_steps, self.steps_anneal)
                      if is_train else 1)
        loss = rec_loss + anneal_reg * kl_loss
        if storer is not None:
            storer['loss'].append(loss.item())
        return loss


class FactorKLoss(BaseLoss):
    """
    Compute the Factor-VAE loss from Algorithm 2 of [1].

    Parameters
    ----------
    device : torch.device

    gamma : float, optional
        Weight of the TC loss term. `gamma` in the paper.

    discriminator : disvae.discriminator.Discriminator

    optimizer_d : torch.optim

    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.

    Reference
    ---------
    [1] H Kim and A Mnih. "Disentangling by factorising,"
        in Proceedings of the 35th International Conference on Machine Learning
        (PMLR, Stockholm, Sweden, 2018), Vol. 80, pp. 2649-2658, 10.48550/arXiv.1802.05983.
    """

    def __init__(self, device,
                 gamma=1.,
                 disc_kwargs={},
                 optim_kwargs=dict(lr=5e-5, betas=(0.5, 0.9)),
                 **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.device = device
        self.discriminator = Discriminator(**disc_kwargs).to(self.device)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), **optim_kwargs)

    def __call__(self, *args, **kwargs):
        raise ValueError("Use `call_optimize` to also train the discriminator")

    def call_optimize(self, data, model, optimizer, storer):
        storer = self._pre_call(model.training, storer)

        # FVAE splits each sampled batch into two sub-batches,
        # one for the VAE and the other for the discriminator
        batch_size = data.size(dim=0)
        half_batch_size = batch_size // 2
        data = data.split(half_batch_size)
        data1 = data[0]
        data2 = data[1]

        # FVAE Loss
        recon_batch, latent_dist, latent_sample1 = model(data1)
        rec_loss = _reconstruction_loss(data1, recon_batch,
                                        storer=storer,
                                        distribution=self.rec_dist)

        kl_loss = _kl_normal_loss(*latent_dist, storer)

        d_z = self.discriminator(latent_sample1)
        # We want log(p_true/p_false). If not using logistic regression but softmax
        # then p_true = exp(logit_true) / Z; p_false = exp(logit_false) / Z
        # so log(p_true/p_false) = logit_true - logit_false
        tc_loss = (d_z[:, 0] - d_z[:, 1]).mean()
        # with sigmoid (not good results) should be `tc_loss = (2 * d_z.flatten()).mean()`

        anneal_reg = (linear_annealing(0, 1, self.n_train_steps, self.steps_anneal)
                      if model.training else 1)
        vae_loss = rec_loss + kl_loss + anneal_reg * self.gamma * tc_loss

        if storer is not None:
            storer['loss'].append(vae_loss.item())
            storer['tc_loss'].append(tc_loss.item())

        if not model.training:
            # don't backprop if evaluating
            return vae_loss

        # Compute VAE gradients
        optimizer.zero_grad()
        vae_loss.backward(retain_graph=True)

        # Discriminator loss
        # Get second sample of latent distribution
        latent_sample2 = model.sample_latent(data2)
        z_perm = _permute_dims(latent_sample2).detach()
        d_z_perm = self.discriminator(z_perm)

        # Calculate total correlation loss
        # for cross entropy the target is the index => need to be long and says
        # that it's first output for d_z and second for perm
        ones = torch.ones(half_batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros_like(ones)
        d_tc_loss = 0.5 * (F.cross_entropy(d_z, zeros) + F.cross_entropy(d_z_perm, ones))
        # with sigmoid would be :
        # d_tc_loss = 0.5 * (self.bce(d_z.flatten(), ones) + self.bce(d_z_perm.flatten(), 1 - ones))

        # Compute discriminator gradients and update
        self.optimizer_d.zero_grad()
        d_tc_loss.backward()
        optimizer.step()
        self.optimizer_d.step()
        if storer is not None:
            storer['discrim_loss'].append(d_tc_loss.item())

        return vae_loss


def _reconstruction_loss(data, recon_data, distribution="bernoulli", storer=None):
    """
    Calculates the per image reconstruction loss for a batch of data; i.e. negative log likelihood.

    Parameters
    ----------
    data : torch.Tensor
        Input data (e.g. batch of images). Shape: (batch_size, n_chan, height, width).

    recon_data : torch.Tensor
        Reconstructed data. Shape: (batch_size, n_chan, height, width).

    distribution : {"bernoulli"}
        Reconstruction distribution of the likelihood on each pixel.
        Implicitly defines the reconstruction loss.
        Bernoulli corresponds to a binary cross entropy.

    storer : dict
        Dictionary in which to store important variables for visualization.

    Returns
    -------
    loss : torch.Tensor
        Per image cross entropy (i.e. normalized per batch but not pixel and
        channel)
    """
    batch_size, n_chan, height, width = recon_data.size()

    if distribution == "bernoulli":
        loss = F.binary_cross_entropy(recon_data, data, reduction="sum")
    else:
        assert distribution not in RECON_DIST
        raise ValueError("Unknown distribution: {}".format(distribution))

    loss = loss / batch_size

    if storer is not None:
        storer['recon_loss'].append(loss.item())

    return loss


def _kl_normal_loss(mean, logvar, storer=None):
    """
    Calculates the KL divergence between
    a normal distribution with diagonal covariance 
    and a standard normal distribution.

    Parameters
    ----------
    mean : torch.Tensor
        Mean of the normal distribution. Shape: (batch_size, latent_dim).

    logvar : torch.Tensor
        Diagonal log variance of the normal distribution. Shape: (batch_size, latent_dim)

    storer : dict
        Dictionary in which to store important variables for visualization.
    """
    latent_dim = mean.size(1)
    # batch mean KLD for each latent dimension
    latent_kl = 0.5 * (-1 - logvar + mean.pow(2) + logvar.exp()).mean(dim=0)
    total_kl = latent_kl.sum()

    if storer is not None:
        storer['kl_loss'].append(total_kl.item())
        for i in range(latent_dim):
            storer['kl_loss_' + str(i)].append(latent_kl[i].item())

    return total_kl


def _permute_dims(latent_sample):
    """
    Implementation of Algorithm 1 from [1].
    Randomly permutes the sample from q(z) (latent_dist) across the batch
    for each of the latent dimensions (mean and log_var).

    Parameters
    ----------
    latent_sample: torch.Tensor
        sample from the latent dimension using the re-parameterization trick
        shape : (batch_size, latent_dim).

    Reference
    ----------
    [1] H Kim and A Mnih. "Disentangling by factorising,"
        in Proceedings of the 35th International Conference on Machine Learning
        (PMLR, Stockholm, Sweden, 2018), Vol. 80, pp. 2649-2658, 10.48550/arXiv.1802.05983.

    """
    perm = torch.zeros_like(latent_sample)
    batch_size, dim_z = perm.size()

    for z in range(dim_z):
        pi = torch.randperm(batch_size).to(latent_sample.device)
        perm[:, z] = latent_sample[pi, z]

    return perm


def linear_annealing(init, fin, step, annealing_steps):
    """Linear annealing of a parameter."""
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)
    return annealed
