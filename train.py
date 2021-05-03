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

Lambdas = namedtuple('Lambdas', ['u', 'v', 'r', 'w'])

ratings_training_dataset = data.read_ratings('data/citeulike-a/cf-train-1-users.dat', 16980)
ratings_test_dataset = data.read_ratings('data/citeulike-a/cf-test-1-users.dat', 16980)


def train_model(sdae, mf, corruption, dataset, optimizer, recon_loss_fn, conf, lambdas, epochs, batch_size, device='cpu'):
    with autograd.no_grad():
        # Initialize V to agree with the encodings.
        mf.V.data = sdae.encode(dataset.content).cpu()

    logging.info('Beginning training')
    for epoch in range(epochs):
        # Each epoch is one iteration of gradient descent which only updates the SDAE
        # parameters; the matrices U and V of the CDL have require_grad=False.
        # These matrices are instead updated manually by block coordinate descent.
        logging.info(f'Staring epoch {epoch + 1}/{epochs}')

        sdae_dataset = TensorDataset(mf.V.to(device), dataset.content)

        # Update SDAE weights. Loss here only depends on SDAE outputs.
        def latent_loss_fn(pred, target):
            return lambdas.v / lambdas.r * F.mse_loss(pred, target)

        train_autoencoder(sdae, corruption, sdae_dataset, batch_size, recon_loss_fn, latent_loss_fn, optimizer)

        # Update U and V.
        with autograd.no_grad():
            # Don't use dropout here.
            sdae.eval()
            latent_pred = sdae.encode(dataset.content).cpu()

            block_coordinate_descent(mf.U, mf.V, dataset.ratings, conf, lambdas, latent_pred)

            ratings_pred = mf.predict()

            print_likelihood(
                mf=mf,
                lambdas=lambdas,
                conf=conf,
                ratings_pred=ratings_pred,
                ratings_target=dataset.ratings,
                latent_pred=latent_pred,
                latent_target=mf.V,
            )

            recall = evaluate.recall(ratings_pred, ratings_training_dataset, 300)
            print(f'  training recall@300: {recall}')

            recall = evaluate.recall(ratings_pred, ratings_test_dataset, 300)
            print(f'      test recall@300: {recall}')

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


def pretrain_sdae(sdae, corruption, dataset, optimizer, loss_fn, epochs, batch_size):
    logging.info('Beginning CDL training')
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


def print_likelihood(mf, lambdas, conf, ratings_pred, ratings_target, latent_pred, latent_target):
    conf_mat = (conf[0] - conf[1]) * ratings_target + conf[1] * torch.ones_like(ratings_target)

    likelihood_v = -F.mse_loss(latent_pred, latent_target, reduction='sum') * lambdas.v / 2
    likelihood_u = -mf.U.square().sum() * lambdas.u / 2
    likelihood_r = -(conf_mat * (ratings_target - ratings_pred).square()).sum() / 2

    likelihood = likelihood_v + likelihood_u + likelihood_r
    logging.info(
        f'  neg_likelihood={-likelihood:>5f}'
        f'  v={-likelihood_v:>5f}'
        f'  u={-likelihood_u:>5f}'
        f'  r={-likelihood_r:>5f}'
    )
