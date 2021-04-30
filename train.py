from collections import namedtuple

import torch
import torch.cuda
from torch import linalg
from torch import autograd
from torch.utils.data import DataLoader


Lambdas = namedtuple('Lambdas', ['u', 'v', 'n', 'w'])


def train_cdl(cdl, dataset, optimizer, conf, lambdas, epochs, batch_size, device='cpu'):
    print('Training CDL')
    for epoch in range(epochs):
        print('Epoch', epoch + 1)

        # Each epoch is one iteration of gradient descent (for the SDAE)
        # followed by one iteration of coordinate ascent (for U and V).

        # The parameters U and V are not updated below since they have
        # require_grad=False.  However, their values will influence the loss
        # and therefore the gradients of the SDAE.
#        print('  fine-tuning SDAE...')
#        train(cdl, dataset.content, batch_size, sdae_loss(cdl.sdae, lambdas), optimizer)

        encoded = cdl.sdae.encode(dataset.content)
        loss = cdl_loss(cdl, dataset.content, dataset.ratings, encoded, conf, lambdas, device=device)
        print(f'  loss: {loss:>7f}')

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
        VtV = conf_b * V.t() @ V
        for j in range(len(U)):
            rated_idx = R[j].nonzero().squeeze(1)
            Vr = V[rated_idx, :]
            A = VtV + (conf_a - conf_b) * Vr.t() @ Vr + idu
            b = Vr.t() @ R[j, rated_idx]
#            VC = V.t() * C[j]
#            A = VC @ V + idu
#            b = VC @ R[j]
            U[j] = linalg.solve(A, b)

        UtU = conf_b * U.t() @ U
        for j in range(len(V)):
            A = UtU + idv
            b = scaled_enc[j]

            rated_idx = R[:, j].nonzero().squeeze(1)
            if len(rated_idx):
                Ur = U[rated_idx, :]
                A += (conf_a - conf_b) * Ur.t() @ Ur
                b += Ur.t() @ R[rated_idx, j]

            #UC = cdl.U.t() * C[:, j]
            #A = UC @ U + idv
            #b = UC @ R[:, j] + scaled_enc[j]
            V[j] = linalg.solve(A, b)


def train_sdae(sdae, dataset, loss_fn, optimizer, epochs, batch_size):
    cur_input = dataset

    # Layer-wise pretraining.
    for i, autoencoder in enumerate(sdae.autoencoders):
        print(f'Training Layer', i + 1)
        for epoch in range(epochs):
            print('Epoch', epoch + 1)
            train(autoencoder, cur_input, batch_size, loss_fn, optimizer)

        with torch.no_grad():
            cur_input = autoencoder.encode(cur_input)

    # Fine-tuning.
    print(f'Fine-tuning SDAE')
    for epoch in range(epochs):
        print('Epoch', epoch + 1)
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
            print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')


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

    content_pred = cdl.sdae.decode(encoded)
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
