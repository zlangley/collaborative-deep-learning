import torch
from torch import nn


class LatentFactorModel:
    def __init__(self, target_shape, latent_size):
        super().__init__()

        self.U = torch.empty((target_shape[0], latent_size))
        self.V = torch.empty((target_shape[1], latent_size))
        self.latent_size = latent_size

        nn.init.normal_(self.U, 0, 0.1)
        nn.init.normal_(self.V, 0, 0.1)

    def predict(self):
        return self.U @ self.V.t()

    def state_dict(self):
        return {'U': self.U, 'V': self.V}

    def load_state_dict(self, d):
        self.U = d['U']
        self.V = d['V']
