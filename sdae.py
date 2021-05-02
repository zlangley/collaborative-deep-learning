import torch.nn as nn


class StackedDenoisingAutoencoder(nn.Module):
    def __init__(self, in_features, layer_sizes, corruption, dropout, activation=nn.Sigmoid(), tie_weights=True):
        super().__init__()

        dims = list(zip([in_features] + layer_sizes[:-1], layer_sizes))
        self.autoencoders = [
            DenoisingAutoencoder(rows, cols, corruption, dropout, activation, tie_weights=tie_weights)
            for i, (rows, cols) in enumerate(dims)
        ]

        self.encode = nn.Sequential(*[autoencoder.encode for autoencoder in self.autoencoders])
        self.decode = nn.Sequential(*[autoencoder.decode for autoencoder in reversed(self.autoencoders)])

        self.weights = [weight for ae in self.autoencoders for weight in ae.weights]
        self.biases = [bias for ae in self.autoencoders for bias in ae.biases]

    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return latent, reconstructed

    def regularization_term(self, reg):
        s = 0
        s += reg * sum(weight.square().sum() for weight in self.weights)
        s += reg * sum(bias.square().sum() for bias in self.biases)
        return s


class DenoisingAutoencoder(nn.Module):
    def __init__(self, in_features, latent_size, corruption, dropout=0, activation=nn.Sigmoid(), tie_weights=True):
        """
        Instantiates a DenoisingAutoencoder.

        :param in_features: The number of features (rows) of the input.
        :param latent_size: The size of the latent representation.
        :param corruption: The probability of corrupting (zeroing) each element of the input.
        :param dropout: The dropout probability before decoding.
        :param activiation: The activation function.
        :param tie_weights: Whether to use the same weight matrix in the encoder and decoder.
        """
        super().__init__()

        encode = nn.Linear(in_features, latent_size)
        decode = nn.Linear(latent_size, in_features)

        if tie_weights:
            decode.weight.data = encode.weight.t()
            self.weights = [encode.weight]
        else:
            self.weights = [encode.weight, decode.weight]

        self.biases = [encode.bias.data, decode.bias.data]

        for weight in self.weights:
            nn.init.xavier_uniform_(weight)

        for bias in self.biases:
            nn.init.zeros_(bias)

        self.encode = nn.Sequential(
            nn.Dropout(corruption),
            encode,
            activation,
        )
        self.decode = nn.Sequential(
            nn.Dropout(dropout),
            decode,
        )

    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return latent, reconstructed
