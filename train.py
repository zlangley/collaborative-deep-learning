from collections import namedtuple
import logging

import torch
import torch.cuda
from torch import linalg
from torch import autograd
from torch.utils.data import DataLoader

import data
import evaluate
from sdae import StackedDenoisingAutoencoder

Lambdas = namedtuple('Lambdas', ['u', 'v', 'n', 'w'])

ratings_training_dataset = data.read_ratings('data/citeulike-a/cf-train-1-users.dat', 16980)
ratings_test_dataset = data.read_ratings('data/citeulike-a/cf-test-1-users.dat', 16980)


def train_model(sdae, mf, dataset, optimizer, conf, lambdas, epochs, batch_size, device='cpu'):
    with autograd.no_grad():
        # Initialize V to agree with the encodings.
        mf.V.data = sdae.encode(dataset.content).cpu()

    logging.info('Beginning training')
    for epoch in range(epochs):
        # Each epoch is one iteration of gradient descent which only updates the SDAE
        # parameters; the matrices U and V of the CDL have require_grad=False.
        # These matrices are instead updated manually by block coordinate descent.
        logging.info(f'Staring epoch {epoch + 1}/{epochs}')

        sdae_dataset = list(zip(dataset.content, mf.V.to(device)))

        # Update SDAE weights. Loss here only depends on SDAE outputs.
        def loss_fn(sdae_out, items_latent_items):
            encoding, reconstruction = sdae_out
            items, latent_items = items_latent_items

            loss = 0
            loss += sdae_pure_loss(sdae, lambdas)(reconstruction, items)
            loss += (latent_items - encoding).square().sum(dim=1).mean() * lambdas.v
            return loss

        train(lambda x: sdae(x[0]), sdae_dataset, batch_size, loss_fn, optimizer)

        # Update U and V.
        with autograd.no_grad():
            # Don't use dropout here.
            sdae.eval()
            encoded = sdae.encode(dataset.content).cpu()

            block_coordinate_descent(mf.U, mf.V, dataset.ratings, conf, lambdas, encoded)

            ratings = dataset.ratings.to(device)

            encoding, reconstruction = sdae(dataset.content)
            conf_mat = (conf[0] - conf[1]) * ratings + conf[1] * torch.ones_like(ratings)

            pred = mf.predict()

            likelihood_w = 0
            likelihood_w -= sum(weight.square().sum() for weight in sdae.weights) * lambdas.w / 2
            likelihood_w -= sum(bias.square().sum() for bias in sdae.biases) * lambdas.w / 2

            likelihood_n = -(reconstruction - dataset.content).square().sum() * lambdas.n / 2
            likelihood_v = -(mf.V.to(device) - encoding).square().sum() * lambdas.v / 2
            likelihood_u = -mf.U.square().sum() * lambdas.u / 2
            likelihood_r = -(conf_mat * (ratings - pred.to(device)).square()).sum() / 2

            likelihood = likelihood_w + likelihood_n + likelihood_v + likelihood_u + likelihood_r
            logging.info(f'  neg_likelihood={-likelihood:>5f} w={-likelihood_w:>5f} n={-likelihood_n:>5f} v={-likelihood_v:>5f} u={-likelihood_u:>5f} r={-likelihood_r:>5f}')

            recall = evaluate.recall(pred, ratings_training_dataset, 300)
            print(f'  training recall@300: {recall}')

            recall = evaluate.recall(pred, ratings_test_dataset, 300)
            print(f'  test recall@300: {recall}')

            sdae.train()


class ContentRatingsDataset:
    def __init__(self, content, ratings):
        # content.shape: (num_items, num_item_features)
        # ratings.shape: (num_users, num_items)
        self.content = content
        self.ratings = ratings


def block_coordinate_descent(U, V, R, conf, lambdas, enc):
    """
    :param U: The latent users matrix of shape (num_users, latent_size).
    :param V: The latent items matrix of shape (num_items, latent_size).
    :param R: The ratings matrix of shape (num_users, num_items).
    :param C: The confidence matrix of shape (num_users, num_items).
    :param enc: The encodings of the content of shape (num_items, latent_size).
    """
    latent_size = U.shape[1]
    conf_a, conf_b = conf

    lv_enc = enc * lambdas.v

    # We compute Vt @ Ci @ V with the following optimization. Recall
    # that Ci = diag(C_i1, ..., C_iJ) where C_ij is a if R_ij = 1 and
    # b otherwise. So we have
    #
    #   Vt @ Ci @ V = Vt @ diag((a - b) * Ri + b * ones) @ V
    #               = (a - b) Vt @ diag(Ri) @ V + b * Vt @ V
    #
    # Notice that since Ri is a zero-one matrix, diag(Ri) simply kills
    # the items of V that user i has does not have in her library; indeed,
    #                Vt @ diag(Ri) @ V = Wt @ Wr,
    # where W is the submatrix restricted to rows j with R_ij != 0.
    # Since W will be *much* smaller than V, it is much more efficient to
    # first extract this submatrix.

    A_base = conf_b * V.t() @ V + lambdas.u * torch.eye(latent_size)
    for j in range(len(U)):
        rated_idx = R[j].nonzero().squeeze(1)
        W = V[rated_idx, :]
        A = (conf_a - conf_b) * W.t() @ W + A_base
        # R[j, rated_idx] is just an all-ones vector
        b = conf_a * W.t().sum(dim=1)

        U[j] = linalg.solve(A, b)

    # The same logic above applies to the users matrix.
    A_base = conf_b * U.t() @ U + lambdas.v * torch.eye(latent_size)
    for j in range(len(V)):
        rated_idx = R[:, j].nonzero().squeeze(1)
        if len(rated_idx) == 0:
            A = A_base
            b = lv_enc[j]
        else:
            W = U[rated_idx, :]
            A = (conf_a - conf_b) * W.t() @ W + A_base
            # R[rated_idx, j] is just an all-ones vector
            b = conf_a * W.t().sum(dim=1) + lv_enc[j]

        V[j] = linalg.solve(A, b)


def train_sdae(sdae, dataset, lambdas, optimizer, epochs, batch_size):
    logging.info('Beginning CDL training')
    cur_input = dataset

    loss_fn = lambda sdae_out, actual: (sdae_out[0] - actual).square().sum(dim=1).mean() * lambdas.n

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


def sdae_pure_loss(sdae: StackedDenoisingAutoencoder, lambdas):
    def _sdae_loss(pred, actual):
        # pred = torch.clamp(pred, min=1e-16)
        # actual = torch.clamp(actual, min=1e-16)
        # cross_entropies = -(actual * torch.log(pred) + (1 - actual) * torch.log(1 - pred)).sum(dim=1)
        # return cross_entropies.mean()

        # First parameter is encoding, second is reconstruction.
        loss = 0
        loss += sum(weight.square().sum() for weight in sdae.weights) * lambdas.w
        loss += sum(bias.square().sum() for bias in sdae.biases) * lambdas.w
        loss += (pred - actual).square().sum(dim=1).mean() * lambdas.n
        return loss

    return _sdae_loss
