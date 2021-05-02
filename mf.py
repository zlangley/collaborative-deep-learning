import torch
import torch.nn as nn

from sdae import StackedDenoisingAutoencoder


class MatrixFactorizationModel:
    def __init__(self, target_shape, latent_size):
        super().__init__()

        self.U = torch.empty((target_shape[0], latent_size))
        self.V = torch.empty((target_shape[1], latent_size))

    def predict(self):
        return self.U @ self.V.t()
