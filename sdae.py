import torch
import torch.nn as nn
import torch.nn.functional as F


class StackedDenoisingAutoencoder(nn.Module):
    def __init__(self, in_features, layer_sizes, corruption, dropout):
        super().__init__()

        dims = zip([in_features] + layer_sizes[:-1], layer_sizes)
        self.autoencoders = nn.ModuleList([
            DenoisingAutoencoder(rows, cols, corruption, dropout)
            for rows, cols in dims
        ])
        self._dropout = dropout

    # TODO: other initialization?

    def forward(self, x):
        x = self.encode(x)

        for autoencoder in reversed(self.autoencoders):
            x = F.dropout(x, self._dropout)
            x = autoencoder.decode(x)

        return x

    def encode(self, x):
        for i, autoencoder in enumerate(self.autoencoders):
            if self.training and i != 0:
                x = F.dropout(x, self._dropout)
            x = autoencoder.encode(x)

        return x


class DenoisingAutoencoder(nn.Module):
    def __init__(self, in_features, encoding_size, corruption, dropout):
        """
        Instantiates a DenoisingAutoencoder.

        :param in_features: The number of features (rows) of the input.
        :param encoding_size: The size of the encoding.
        :param corruption: The corruption probability for each element of the input.
        :param dropout: The dropout probability before decoding.
        """
        super().__init__()

        self._encoder = nn.Linear(in_features, encoding_size)
        self._dropout = nn.Dropout(dropout)
        self._decoder = nn.Linear(encoding_size, in_features)

        self._corruption = corruption

        # Tie weights.
        self._decoder.weight = nn.Parameter(self._encoder.weight.t())

        # TODO: xavier initialization in linear modules

    def forward(self, x):
        x = F.dropout(x, self._corruption)
        x = self.encode(x)
        x = self._dropout(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        x = self._encoder(x)
        x = torch.sigmoid(x)
        return x

    def decode(self, x):
        x = self._decoder(x)
        #x = torch.sigmoid(x)
        return x