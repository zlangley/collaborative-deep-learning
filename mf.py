import torch
import torch.nn as nn


class MatrixFactorizationModel:
    def __init__(self, target_shape, latent_size):
        super().__init__()

        self.U = torch.empty((target_shape[0], latent_size))
        self.V = torch.empty((target_shape[1], latent_size))
        self.latent_size = latent_size

        nn.init.normal_(self.U, 0, 0.1)
        nn.init.normal_(self.V, 0, 0.1)

    def predict(self):
        return self.U @ self.V.t()
