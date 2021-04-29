import torch
import torch.nn as nn
import torch.optim as optim

import data
from sdae import StackedDenoisingAutoencoder
from train import train_sdae

if __name__ == '__main__':
    dataset = data.read_mult_dat('data/citeulike-a/mult.dat')
    # dataset.shape: (16980, 8000)

    training_dataset = dataset[:15282]
    validation_dataset = dataset[:15282]

    sdae = StackedDenoisingAutoencoder(
        in_features=training_dataset.shape[1],
        encoding_sizes=[50, 50],
        corruption=0.3,
        dropout=0.2,
    )

    def loss_fn(pred, actual):
        pred = torch.clamp(pred, min=1e-16)
        actual = torch.clamp(actual, min=1e-16)
        cross_entropies = -(actual * torch.log(pred) + (1 - actual) * torch.log(1 - pred)).sum(dim=1)
        return cross_entropies.mean()

    optimizer = optim.Adam(sdae.parameters())
    train_sdae(sdae, training_dataset, loss_fn, optimizer, epochs=20, batch_size=60)
