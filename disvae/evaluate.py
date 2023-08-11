import torch
import logging
from timeit import default_timer
from collections import defaultdict
from tqdm import tqdm
from disvae.utils.modelIO import save_metadata


TEST_LOSSES_FILE = "test_losses.log"


class Evaluator:
    """
    Class to handle evaluation of model.

    Parameters
    ----------
    model: disvae.vae.VAE

    loss_f: disvae.models.BaseLoss
        Loss function.

    device: torch.device, optional
        Device on which to run the code.

    logger: logging.Logger, optional
        Logger.

    save_dir : str, optional
        Directory for saving logs.

    is_progress_bar: bool, optional
        Whether to use a progress bar for training.
    """

    def __init__(self, model, loss_f,
                 device=torch.device("cpu"),
                 logger=logging.getLogger(__name__),
                 save_dir="results",
                 is_progress_bar=True):

        self.device = device
        self.loss_f = loss_f
        self.model = model.to(self.device)
        self.logger = logger
        self.save_dir = save_dir
        self.is_progress_bar = is_progress_bar
        self.logger.info("Testing Device: {}".format(self.device))

    def __call__(self, data_loader, is_losses=True):
        """Compute all test losses.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        is_metrics: bool, optional
            Whether to compute and store the disentangling metrics.

        is_losses: bool, optional
            Whether to compute and store the test losses.
        """
        start = default_timer()
        is_still_training = self.model.training
        self.model.eval()

        losses = None
        if is_losses:
            self.logger.info('Computing losses...')
            losses = self.compute_losses(data_loader)
            self.logger.info('Losses: {}'.format(losses))
            save_metadata(losses, self.save_dir, filename=TEST_LOSSES_FILE)

        if is_still_training:
            self.model.train()

        self.logger.info('Finished evaluating after {:.1f} min.'.format((default_timer() - start) / 60))

        return losses

    def compute_losses(self, dataloader):
        """
        Compute all test losses.

        Parameters
        ----------
        dataloader: torch.utils.data.DataLoader
        """
        storer = defaultdict(list)
        for data, _ in tqdm(dataloader, leave=False, disable=not self.is_progress_bar):
            data = data.to(self.device)

            try:
                recon_batch, latent_dist, latent_sample = self.model(data)
                _ = self.loss_f(data, recon_batch, latent_dist, self.model.training,
                                storer, latent_sample=latent_sample)
            except ValueError:
                # for losses that use multiple optimizers (e.g. FVAE)
                _ = self.loss_f.call_optimize(data, self.model, None, storer)

            losses = {k: sum(v) / len(dataloader) for k, v in storer.items()}
            return losses
