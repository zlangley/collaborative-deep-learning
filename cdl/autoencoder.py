from torch import nn


class StackedAutoencoder(nn.Module):
    def __init__(self, autoencoders):
        super().__init__()

        self.autoencoders = nn.ModuleList(autoencoders)

    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return latent, reconstructed

    def encode(self, x):
        for i, ae in enumerate(self.autoencoders):
            x = ae.encode(x)

            if i != len(self.autoencoders) - 1:
                x = ae.dropout(x)

        return x

    def decode(self, x):
        for i, ae in enumerate(reversed(self.autoencoders)):
            if i != 0:
                x = ae.activation(x)

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

        encode = nn.Linear(in_features, latent_size)
        decode = nn.Linear(latent_size, in_features)

        if tie_weights:
            decode.weight.data = encode.weight.data.t()

        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        self.encode = nn.Sequential(
            encode,
            self.activation,
        )
        self.decode = nn.Sequential(
            self.dropout,
            decode,
        )

    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return latent, reconstructed
