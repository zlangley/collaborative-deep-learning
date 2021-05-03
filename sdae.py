import torch.nn as nn


class StackedAutoencoder(nn.Module):
    def __init__(self, autoencoder_stack):
        super().__init__()

        self.autoencoders = autoencoder_stack

        encoder_modules = []
        decoder_modules = []

        for autoencoder in self.autoencoders[:-1]:
            encoder_modules.append(autoencoder._encode)
            encoder_modules.append(autoencoder._activation)
            encoder_modules.append(autoencoder._dropout)

        # No dropout at the last encoding step.
        encoder_modules.append(self.autoencoders[-1]._encode)
        encoder_modules.append(self.autoencoders[-1]._activation)
        decoder_modules.append(self.autoencoders[-1]._dropout)
        decoder_modules.append(self.autoencoders[-1]._decode)

        for autoencoder in reversed(self.autoencoders[:-1]):
            decoder_modules.append(autoencoder._activation)
            decoder_modules.append(autoencoder._dropout)
            decoder_modules.append(autoencoder._decode)

        self.encode = nn.Sequential(*encoder_modules)
        self.decode = nn.Sequential(*decoder_modules)

        self.weights = [weight for ae in self.autoencoders for weight in ae.weights]
        self.biases = [bias for ae in self.autoencoders for bias in ae.biases]

    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return latent, reconstructed


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

        self._encode = nn.Linear(in_features, latent_size)
        self._decode = nn.Linear(latent_size, in_features)
        self._activation = activation
        self._dropout = nn.Dropout(dropout)

        if tie_weights:
            self._decode.weight.data = self._encode.weight.t()
            self.weights = [self._encode.weight]
        else:
            self.weights = [self._encode.weight, self._decode.weight]

        self.biases = [self._encode.bias.data, self._decode.bias.data]

        for weight in self.weights:
            nn.init.xavier_uniform_(weight)

        for bias in self.biases:
            nn.init.zeros_(bias)

        self.encode = nn.Sequential(self._encode, self._activation)
        self.decode = nn.Sequential(self._dropout, self._decode)

    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return latent, reconstructed
