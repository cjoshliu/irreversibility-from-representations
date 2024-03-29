import os
import torch
import logging
import numpy as np
from tqdm import trange
from scipy import stats
from timeit import default_timer
from collections import defaultdict
from disvae.utils.modelIO import save_model
from disvae.models.losses import _reconstruction_loss


TRAIN_LOSSES_LOGFILE = "train_losses.log"


class Trainer():
    """
    Class to handle training of model.

    Parameters
    ----------
    model: disvae.vae.VAE

    optimizer: torch.optim.Optimizer

    loss_f: disvae.models.BaseLoss
        Loss function.

    epochs: int, optional
        Maximum number of epochs to train the model.

    device: torch.device, optional
        Device on which to run the code.

    logger: logging.Logger, optional
        Logger.

    save_dir : str, optional
        Directory for saving logs.

    training_recorder : utils.visualize.TrainingRecorder (optional)
        Records reconstructions of a selected sample at each epoch.

    is_progress_bar: bool, optional
        Whether to use progress bar for training.
    """

    def __init__(self, model, optimizer, loss_f, epochs,
                 device=torch.device("cpu"),
                 logger=logging.getLogger(__name__),
                 save_dir="results",
                 training_recorder=None,
                 is_progress_bar=True):

        self.device = device
        self.model = model.to(self.device)
        self.loss_f = loss_f
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.is_progress_bar = is_progress_bar
        self.logger = logger
        self.losses_logger = LossesLogger(os.path.join(self.save_dir, TRAIN_LOSSES_LOGFILE))
        self.training_recorder = training_recorder
        self.logger.info("Training Device: {}".format(self.device))
        self.epochs = epochs

    def __call__(self, data_loader, cutoff=[16, 16], checkpoint_every=2000):
        """
        Trains the model.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        checkpoint_every: int, optional
            Save a checkpoint of the trained model every n epoch.
        """
        start = default_timer()
        self.model.train()

        window_size = cutoff[0]
        cutoff_epochs = cutoff[1]

        window_losses = np.zeros(window_size)
        counter = 0
        for epoch in range(self.epochs):
            storer = defaultdict(list)
            mean_epoch_loss, _ = self._train_epoch(data_loader, storer, epoch)
            self.logger.info('Epoch: {} Average loss per image: {:.2f}'.format(epoch + 1, mean_epoch_loss))
            self.losses_logger.log(epoch, storer)

            if self.training_recorder is not None:
                self.training_recorder()

            if epoch < window_size:
                window_losses[epoch] = mean_epoch_loss
            else:
                window_losses = np.roll(window_losses, -1)
                window_losses[-1] = mean_epoch_loss
                lin_res = stats.linregress(np.arange(window_size),
                                           window_losses,
                                           alternative='less')

                if lin_res.pvalue > 0.05:
                    counter += 1
                else:
                    counter = 0

                if counter == cutoff_epochs:
                    self.epochs = epoch
                    break

            if epoch % checkpoint_every == 0:
                save_model(self.model, self.save_dir,
                           filename="model-{}.pt".format(epoch))

        if self.training_recorder is not None:
            self.training_recorder.save_reset()

        self.model.eval()

        delta_time = (default_timer() - start) / 60
        self.logger.info('Finished training after {:.1f} min.'.format(delta_time))

    def _train_epoch(self, data_loader, storer, epoch):
        """
        Trains the model for one epoch.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        storer: dict
            Dictionary in which to store important variables for visualization.

        epoch: int
            Epoch number.

        Return
        ------
        mean_epoch_loss: float
            Mean loss per image.
        """
        epoch_loss = 0.
        epoch_recon_loss = 0.
        kwargs = dict(desc="Epoch {}".format(epoch + 1), leave=False,
                      disable=not self.is_progress_bar)
        with trange(len(data_loader), **kwargs) as t:
            for _, (data, _) in enumerate(data_loader):
                iter_loss, iter_recon_loss = self._train_iteration(data, storer)
                epoch_loss += iter_loss
                epoch_recon_loss += iter_recon_loss
                t.set_postfix(loss=iter_loss)
                t.update()

        mean_epoch_loss = epoch_loss / len(data_loader)
        mean_epoch_recon_loss = epoch_recon_loss / len(data_loader)
        return mean_epoch_loss, mean_epoch_recon_loss

    def _train_iteration(self, data, storer):
        """
        Trains the model for one iteration on a batch of data.

        Parameters
        ----------
        data: torch.Tensor
            A batch of data. Shape: (batch_size, channel, height, width).

        storer: dict
            Dictionary in which to store important variables for visualization.
        """
        data = data.to(self.device)

        try:
            recon_batch, latent_dist, latent_sample = self.model(data)
            loss = self.loss_f(data, recon_batch, latent_dist, self.model.training, storer,
                               latent_sample=latent_sample)
            recon_loss = _reconstruction_loss(data, recon_batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        except ValueError:
            # for losses that use multiple optimizers (e.g. Factor)
            loss = self.loss_f.call_optimize(data, self.model, self.optimizer, storer)
            recon_loss = _reconstruction_loss(data, recon_batch)

        return loss.item(), recon_loss.item()


class LossesLogger(object):
    """Class for logging losses."""

    def __init__(self, file_path_name):
        """Create a logger to store information for plotting."""
        if os.path.isfile(file_path_name):
            os.remove(file_path_name)

        self.logger = logging.getLogger("losses_logger")
        self.logger.setLevel(1)  # always store
        file_handler = logging.FileHandler(file_path_name)
        file_handler.setLevel(1)
        self.logger.addHandler(file_handler)

        header = ",".join(["Epoch", "Loss", "Value"])
        self.logger.debug(header)

    def log(self, epoch, losses_storer):
        """Write to the log file."""
        for k, v in losses_storer.items():
            log_string = ",".join(str(item) for item in [epoch, k, mean(v)])
            self.logger.debug(log_string)


# HELPERS
def mean(l):
    """Compute the mean of a list"""
    return sum(l) / len(l)
