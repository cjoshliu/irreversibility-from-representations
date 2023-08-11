"""
Module containing multilayer-perceptron discriminator of factorizing VAE.
"""
from torch import nn
from disvae.utils.initialization import weights_init


class Discriminator(nn.Module):
    def __init__(self,
                 neg_slope=0.2,
                 latent_dim=2,
                 hidden_units=1000):
        """
        Discriminator proposed in [1].

        Parameters
        ----------
        neg_slope: float
            Leaky ReLu negative slope.

        latent_dim : int
            Dimensionality of latent variables.

        hidden_units: int
            Number of hidden units.

        Architecture
        ------------
        - 6 layer multi-layer perceptron, each with 1000 hidden units
        - Leaky ReLu activations
        - Output 2 "logits" (unnormalized log-probabilities)

        Reference
        ---------
        [1] H Kim and A Mnih. "Disentangling by factorising,"
            in Proceedings of the 35th International Conference on Machine Learning
            (PMLR, Stockholm, Sweden, 2018), Vol. 80, pp. 2649-2658, 10.48550/arXiv.1802.05983.
        """
        super(Discriminator, self).__init__()

        # Activation parameters
        self.neg_slope = neg_slope
        self.leaky_relu = nn.LeakyReLU(self.neg_slope, True)

        # Layer parameters
        self.z_dim = latent_dim
        self.hidden_units = hidden_units
        out_units = 2

        # Fully connected layers
        self.lin1 = nn.Linear(self.z_dim, hidden_units)
        self.lin2 = nn.Linear(hidden_units, hidden_units)
        self.lin3 = nn.Linear(hidden_units, hidden_units)
        self.lin4 = nn.Linear(hidden_units, hidden_units)
        self.lin5 = nn.Linear(hidden_units, hidden_units)
        self.lin6 = nn.Linear(hidden_units, out_units)

        self.reset_parameters()

    def forward(self, z):

        # Fully connected layers with leaky ReLu activations
        z = self.leaky_relu(self.lin1(z))
        z = self.leaky_relu(self.lin2(z))
        z = self.leaky_relu(self.lin3(z))
        z = self.leaky_relu(self.lin4(z))
        z = self.leaky_relu(self.lin5(z))
        z = self.lin6(z)

        return z

    def reset_parameters(self):
        self.apply(weights_init)
