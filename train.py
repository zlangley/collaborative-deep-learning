from collections import namedtuple
import logging

import torch
import torch.cuda
from torch import nn
from torch import linalg
from torch import autograd
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

import data
import evaluate
from optim import LatentRepresentationOptimizer

Lambdas = namedtuple('Lambdas', ['u', 'v', 'r', 'w'])


def train_model(sdae, mf, corruption, dataset, optimizer, recon_loss_fn, conf, lambdas, epochs, batch_size, device=None, max_iters=10):
    def latent_loss_fn(pred, target):
        return lambdas.v / lambdas.r * F.mse_loss(pred, target, reduction='sum') / 2

    lr_optim = LatentRepresentationOptimizer(mf, dataset.ratings, conf[0], conf[1], lambdas.u, lambdas.v)

    logging.info('Beginning training')
    for epoch in range(epochs):
        # Each epoch is one iteration of gradient descent which only updates the SDAE
        # parameters; the matrices U and V of the CDL have require_grad=False.
        # These matrices are instead updated manually by block coordinate descent.
        logging.info(f'Staring epoch {epoch + 1}/{epochs}')

        # Update U and V.
        with autograd.no_grad():
            # Don't use dropout here.
            sdae.eval()
            latent_items_target = sdae.encode(dataset.content).cpu()
            sdae.train()

        lr_optim.step(latent_items_target)

        # Update SDAE weights. Loss here only depends on SDAE outputs.
        sdae_dataset = TensorDataset(mf.V.to(device), dataset.content)
        train_autoencoder(sdae, corruption, sdae_dataset, batch_size, recon_loss_fn, latent_loss_fn, optimizer)

    sdae.eval()
    latent_items_target = sdae.encode(dataset.content).cpu()

    # Now optimize U and V completely holding the SDAE latent layer fixed.
    prev_loss = None
    for i in range(max_iters):
        loss = lr_optim.loss(latent_items_target)
        if prev_loss is not None and (prev_loss - loss) / loss < 1e-4:
            break

        lr_optim.step(latent_items_target)
        prev_loss = loss


class ContentRatingsDataset:
    def __init__(self, content, ratings):
        # content.shape: (num_items, num_item_features)
        # ratings.shape: (num_users, num_items)
        self.content = content
        self.ratings = ratings


def pretrain_sdae(sdae, corruption, dataset, optimizer, loss_fn, epochs, batch_size):
    cur_dataset = dataset

    # Layer-wise pretraining.
    for i, autoencoder in enumerate(sdae.autoencoders):
        logging.info(f'Training layer {i + 1}/{len(sdae.autoencoders)}')
        for epoch in range(epochs):
            logging.info(f'Staring epoch {epoch + 1}/{epochs}')
            train_autoencoder(autoencoder, corruption, cur_dataset, batch_size, loss_fn, None, optimizer)

        with torch.no_grad():
            autoencoder.eval()
            cur_dataset = autoencoder.encode(cur_dataset)
            autoencoder.train()

    # Fine-tuning.
    for epoch in range(epochs):
        logging.info(f'Staring epoch {epoch + 1}/{epochs}')
        train_autoencoder(sdae, corruption, dataset, batch_size, loss_fn, None, optimizer)


def train_autoencoder(autoencoder, corruption, dataset, batch_size, recon_loss_fn, latent_loss_fn, optimizer):
    dataloader = DataLoader(dataset, batch_size)
    size = len(dataloader.dataset)

    for batch, X_b in enumerate(dataloader):
        # TODO: Make CorruptedDataSet.
        if type(X_b) != list:
            # One target: reconstruction.
            recon_target = X_b
            corrupted_X_b = F.dropout(X_b, corruption)
            _, recon_pred = autoencoder(corrupted_X_b)
            loss = recon_loss_fn(recon_pred, recon_target)

            if batch % 100 == 0:
                current = batch * batch_size
                logging.info(f'  loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')
        else:
            assert len(X_b) == 2
            # Two targets: latent and reconstruction.
            latent_target, recon_target = X_b

            corrupted_recon_target = F.dropout(recon_target, corruption)
            latent_pred, recon_pred = autoencoder(corrupted_recon_target)

            latent_loss = latent_loss_fn(latent_pred, latent_target)
            recon_loss = recon_loss_fn(recon_pred, recon_target)

            loss = latent_loss + recon_loss

            if batch % 100 == 0:
                current = batch * batch_size
                logging.info(f'  loss {loss:>5f}  latent_loss: {latent_loss:>5f}  recon_loss {recon_loss:>6f}  [{current:>5d}/{size:>5d}]')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
