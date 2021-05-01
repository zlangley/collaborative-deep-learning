from collections import namedtuple
import logging

import torch
import torch.cuda
from torch import linalg
from torch import autograd
from torch.utils.data import DataLoader


Lambdas = namedtuple('Lambdas', ['u', 'v', 'n', 'w'])


def train_cdl(cdl, dataset, optimizer, conf, lambdas, epochs, batch_size, device='cpu'):
    logging.info('Beginning CDL training')
    for epoch in range(epochs):
        # Each epoch is one iteration of gradient descent which only updates the SDAE
        # parameters; the matrices U and V of the CDL have require_grad=False.
        # These matrices are instead updated manually with the coordinate ascent algorithm.
        logging.info(f'Staring epoch {epoch + 1}/{epochs}')

        sdae_dataset = list(zip(dataset.content, cdl.V.to(device)))

        # Update SDAE weights. Loss here only depends on SDAE outputs.
        def loss_fn(sdae_out, items_latent_items):
            encoding, reconstruction = sdae_out
            items, latent_items = items_latent_items

            loss = 0
            loss += sdae_pure_loss(cdl.sdae, lambdas)(reconstruction, items)
            loss += (latent_items - encoding).square().sum(dim=1).mean() * lambdas.v / 2
            return loss

        train(lambda x: cdl.sdae(x[0]), sdae_dataset, batch_size, loss_fn, optimizer)

        # Update U and V manually with coordinate ascent.
        with autograd.no_grad():
            # Don't use dropout here.
            cdl.sdae.eval()
            encoded = cdl.sdae.encode(dataset.content).cpu()

            coordinate_ascent(cdl, dataset.ratings, conf, lambdas, encoded)

            ratings = dataset.ratings.to(device)

            encoding, reconstruction = cdl.sdae(dataset.content)
            conf_mat = (conf[0] - conf[1]) * ratings + conf[1] * torch.ones_like(ratings)

            likelihood = 0
            likelihood -= sdae_pure_loss(cdl.sdae, lambdas)(reconstruction, dataset.content)
            likelihood -= (cdl.V.to(device) - encoding).square().sum(dim=1).mean() * lambdas.v / 2
            likelihood -= cdl.U.to(device).square().sum() * lambdas.u / 2
            likelihood -= (conf_mat * (ratings - cdl.predict())).square().sum(dim=1).mean() / 2
            logging.info(f'  likelihood: {likelihood}')

            cdl.sdae.train()


class ContentRatingsDataset:
    def __init__(self, content, ratings):
        # content.shape: (num_items, num_item_features)
        # ratings.shape: (num_users, num_items)
        self.content = content
        self.ratings = ratings


def coordinate_ascent(cdl, R, conf, lambdas, enc):
    """
    :param U: The latent users matrix of shape (num_users, latent_size).
    :param V: The latent items matrix of shape (num_items, latent_size).
    :param R: The ratings matrix of shape (num_users, num_items).
    :param C: The confidence matrix of shape (num_users, num_items).
    :param enc: The encodings of the content of shape (num_items, latent_size).
    """
    U = cdl.U
    V = cdl.V

    latent_size = U.shape[1]
    idu = lambdas.u * torch.eye(latent_size)
    idv = lambdas.v * torch.eye(latent_size)
    conf_a, conf_b = conf

    scaled_enc = enc * lambdas.v

    # We compute Vt @ Ci @ V with the following optimization. Recall
    # that Ci = diag(C_i1, ..., C_iJ) where C_ij is a if R_ij = 1 and
    # b otherwise. So we have
    #
    #   Vt @ Ci @ V = Vt @ diag((a - b) * Ri + b * ones) @ V
    #             = (a - b) Vt @ diag(Ri) @ V + b * Vt @ V
    #
    # Notice that since Ri is a zero-one matrix, diag(Ri) simply kills
    # the items of V that user i has does not have in her library; indeed,
    #                Vt @ diag(Ri) @ V = Wt @ Wr,
    # where W is the submatrix restricted to rows j with R_ij != 0.
    # Since W will be *much* smaller than V, it is much more efficient to
    # first extract this submatrix.

    A_base = conf_b * V.t() @ V + idu
    for j in range(len(U)):
        rated_idx = R[j].nonzero().squeeze(1)
        W = V[rated_idx, :]
        A = (conf_a - conf_b) * W.t() @ W + A_base
        b = W.t() @ R[j, rated_idx]

        U[j] = linalg.solve(A, b)

    # The same logic above applies to the users matrix.
    A_base = conf_b * U.t() @ U + idv
    for j in range(len(V)):
        rated_idx = R[:, j].nonzero().squeeze(1)
        if len(rated_idx) == 0:
            A = A_base
            b = scaled_enc[j]
        else:
            W = U[rated_idx, :]
            A = (conf_a - conf_b) * W.t() @ W + A_base
            b = W.t() @ R[rated_idx, j] + scaled_enc[j]

        V[j] = linalg.solve(A, b)


def train_sdae(sdae, dataset, loss_fn, optimizer, epochs, batch_size):
    logging.info('Beginning CDL training')
    cur_input = dataset

    # Layer-wise pretraining.
    for i, autoencoder in enumerate(sdae.autoencoders):
        logging.info(f'Training layer {i + 1}/{len(sdae.autoencoders)}')
        for epoch in range(epochs):
            logging.info(f'Staring epoch {epoch + 1}/{epochs}')
            train(lambda x: autoencoder(x)[1], cur_input, batch_size, loss_fn, optimizer)

        with torch.no_grad():
            autoencoder.eval()
            cur_input = autoencoder.encode(cur_input)
            autoencoder.train()

    # Fine-tuning.
    for epoch in range(epochs):
        logging.info(f'Staring epoch {epoch + 1}/{epochs}')
        train(lambda x: sdae(x)[1], dataset, batch_size, loss_fn, optimizer)


def train(forward, dataset, batch_size, loss_fn, optimizer):
    dataloader = DataLoader(dataset, batch_size)
    size = len(dataloader.dataset)

    for batch, X_b in enumerate(dataloader):
        pred = forward(X_b)
        loss = loss_fn(pred, X_b)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            current = batch * batch_size
            logging.info(f'  current loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')


def sdae_pure_loss(sdae, lambdas):
    def _sdae_loss(pred, actual):
        # pred = torch.clamp(pred, min=1e-16)
        # actual = torch.clamp(actual, min=1e-16)
        # cross_entropies = -(actual * torch.log(pred) + (1 - actual) * torch.log(1 - pred)).sum(dim=1)
        # return cross_entropies.mean()

        # First parameter is encoding, second is reconstruction.
        loss = 0
        for param in sdae.parameters():
            loss += (param * param).sum() * lambdas.w / 2

        loss += (pred - actual).square().sum(dim=1).mean() * lambdas.n / 2
        return loss

    return _sdae_loss
