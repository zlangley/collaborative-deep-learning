import torch
from torch import nn


class MatrixFactorizationModel:
    def __init__(self, target_shape, latent_size):
        super().__init__()

        self.U = torch.empty((target_shape[0], latent_size))
        self.V = torch.empty((target_shape[1], latent_size))
        self.latent_size = latent_size

        nn.init.normal_(self.U, 0, 0.1)
        nn.init.normal_(self.V, 0, 0.1)

    def estimate(self):
        return self.U @ self.V.t()

    def state_dict(self):
        return {'U': self.U, 'V': self.V}

    def update_state_dict(self, d):
        self.U = d['U']
        self.V = d['V']
        assert self.U.shape[1] == self.V.shape[1]
        self.latent_size = self.U.shape[1]

    def compute_recall(self, test, k):
        _, indices = torch.topk(self.estimate(), k)
        gathered = test.gather(1, indices)
        recall = gathered.sum(dim=1) / test.sum(dim=1)
        return recall.mean()
