"""
Module containing Burgess decoder.
"""
import torch
import numpy as np
from torch import nn


# Additional decoders should be named Decoder<Model>
def get_decoder(model_type):
    model_type = model_type.lower().capitalize()
    return eval("Decoder{}".format(model_type))


class DecoderBurgess(nn.Module):
    def __init__(self, img_size, latent_dim=2):
        r"""Decoder of the model proposed in [1].

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 64, 64) or (3, 64, 64).

        latent_dim : int
            Dimensionality of latent output.

        Architecture
        ------------
        Below transposed for decoder
        - 4 convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
        - 2 fully connected layers (each of 256 units)
        - Latent distribution:
            - 1 fully connected layer of 4 units (log variance and mean for 2 Gaussians)

        Reference
        ---------
        [1] CP Burgess, I Higgins, L Matthey, N Watters, G Desjardins, and A Lerchner.
            "Understanding disentangling in $\beta$-VAE,"
            in Proceedings of the 2017 NIPS Workshop on Learning Disentangled Representations
            (NeurIPS, Long Beach, CA, 2017), 10.48550/arXiv.1804.03599.
        """
        super(DecoderBurgess, self).__init__()

        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 128
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]
        self.img_size = img_size

        # Fully connected layers
        self.lin1 = nn.Linear(latent_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, np.product(self.reshape))

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        self.convT1 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.convT2 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.convT3 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.convT4 = nn.ConvTranspose2d(hid_channels, n_chan, kernel_size, **cnn_kwargs)

    def forward(self, z):
        batch_size = z.size(0)

        # Fully connected layers with ReLu activations
        x = torch.relu(self.lin1(z))
        x = torch.relu(self.lin2(x))
        x = torch.relu(self.lin3(x))
        x = x.view(batch_size, *self.reshape)

        # Convolutional layers with ReLu activations
        x = torch.relu(self.convT1(x))
        x = torch.relu(self.convT2(x))
        x = torch.relu(self.convT3(x))
        
        # Sigmoid activation for final convolutional layer
        x = torch.sigmoid(self.convT4(x))

        return x
