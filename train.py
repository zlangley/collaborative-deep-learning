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

        # Don't use dropout here.
        cdl.sdae.eval()
        encoded = cdl.sdae.encode(dataset.content)
        cdl.sdae.train()

        # TODO: We should probably still use batches?
        loss = cdl_loss(cdl, dataset.content, dataset.ratings, encoded, conf, lambdas, device=device)
        logging.info(f'  current loss: {loss:>7f}')

        # Update SDAE weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update U and V manually with coordinate ascent.
        with autograd.no_grad():
            coordinate_ascent(cdl, dataset.ratings, conf, lambdas, encoded.cpu())


class ContentRatingsDataset:
    def __init__(self, content, ratings):
        # content.shape: (num_items, num_item_features)
        # ratings.shape: (num_users, num_items)
        self.content = content
        self.ratings = ratings


def coordinate_ascent(cdl, R, conf, lambdas, enc, num_iters=1):
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

    for i in range(num_iters):
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
            train(autoencoder, cur_input, batch_size, loss_fn, optimizer)

        with torch.no_grad():
            cur_input = autoencoder.encode(cur_input)

    # Fine-tuning.
    for epoch in range(epochs):
        logging.info(f'Staring epoch {epoch + 1}/{epochs}')
        train(sdae, dataset, batch_size, loss_fn, optimizer)


def train(model, dataset, batch_size, loss_fn, optimizer):
    dataloader = DataLoader(dataset, batch_size)
    size = len(dataloader.dataset)

    for batch, X_b in enumerate(dataloader):
        pred = model(X_b)
        loss = loss_fn(pred, X_b)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            current = batch * len(X_b)
            logging.info(f'  current loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')


def sdae_loss(sdae, lambdas):
    def _sdae_loss(pred, actual):
        # pred = torch.clamp(pred, min=1e-16)
        # actual = torch.clamp(actual, min=1e-16)
        # cross_entropies = -(actual * torch.log(pred) + (1 - actual) * torch.log(1 - pred)).sum(dim=1)
        # return cross_entropies.mean()

        loss = 0
        for param in sdae.parameters():
            loss += (param * param).sum() * lambdas.w / 2

        loss += ((pred - actual) ** 2).sum() * lambdas.n / 2
        return loss

    return _sdae_loss


def cdl_loss(cdl, content, ratings, encoded, conf, lambdas, device='cpu'):
    cdl.U.data = cdl.U.data.to(device)
    cdl.V.data = cdl.V.data.to(device)
    ratings = ratings.to(device)

    content_pred = cdl.sdae(content)
    ratings_pred = cdl.predict().to(device)

    conf_matrix = ratings * (conf[0] - conf[1]) + conf[1] * torch.ones_like(ratings)

    loss = 0
    loss += sdae_loss(cdl.sdae, lambdas)(content_pred, content)
    loss += (cdl.U ** 2).sum() * lambdas.u / 2
    loss += (conf_matrix * (ratings - ratings_pred) ** 2).sum() / 2
    loss += ((cdl.V - encoded) ** 2).sum() * lambdas.v / 2

    cdl.U.data = cdl.U.data.cpu()
    cdl.V.data = cdl.V.data.cpu()

    return loss
