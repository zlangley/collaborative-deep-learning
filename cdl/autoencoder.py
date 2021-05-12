import torch
from torch import nn
import torch.nn.functional as F


class StackedAutoencoder(nn.Module):
    def __init__(self, autoencoder_stack):
        super().__init__()

        self.stack = nn.ModuleList(autoencoder_stack)

    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return latent, reconstructed

    def encode(self, x):
        for i, ae in enumerate(self.stack):
            x = ae.encode(x)

            if i != len(self.stack) - 1:
                x = ae._dropout(x)

        return x

    def decode(self, x):
        for i, ae in enumerate(reversed(self.stack)):
            if i != 0:
                x = ae._activation(x)

            x = ae.decode(x)

        return x


class Autoencoder(nn.Module):
    def __init__(self, in_features, latent_size, dropout=0, activation=nn.Sigmoid(), tie_weights=True):
        """
        Instantiates a Autoencoder.

        :param in_features: The number of features (rows) of the input.
        :param latent_size: The size of the latent representation.
        :param dropout: The dropout probability before decoding.
        :param activation: The activation function.
        :param tie_weights: Whether to use the same weight matrix in the encoder and decoder.
        """
        super().__init__()

        self._weight_enc = nn.Parameter(torch.Tensor(latent_size, in_features))

        if tie_weights:
            self._weight_dec = nn.Parameter(self._weight_enc.t())
        else:
            self._weight_dec = nn.Parameter(torch.Tensor(in_features, latent_size))

        self._bias_enc = nn.Parameter(torch.Tensor(latent_size))
        self._bias_dec = nn.Parameter(torch.Tensor(in_features))

        self._activation = activation
        self._dropout = nn.Dropout(dropout)

        nn.init.xavier_normal_(self._weight_enc)
        nn.init.xavier_normal_(self._weight_dec)
        nn.init.zeros_(self._bias_enc)
        nn.init.zeros_(self._bias_dec)

    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return latent, reconstructed

    def encode(self, x):
        x = F.linear(x, self._weight_enc, self._bias_enc)
        x = self._activation(x)
        return x

    def decode(self, x):
        x = self._dropout(x)
        x = F.linear(x, self._weight_dec, self._bias_dec)
        return x
