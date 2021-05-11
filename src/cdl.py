import logging

import torch
import torch.cuda
import torch.nn.functional as F
from torch import autograd, linalg
from torch.utils.data import DataLoader, TensorDataset

import data


# Since we use AdamW, we can rescale our losses with negligible effect on the optimization.
# Thus, we divide our autoencoder losses by lambda_n.
def AutoencoderLatentLoss(lambda_v, lambda_n):
    return lambda pred, target: lambda_v / lambda_n * F.mse_loss(pred, target)


def train_model(sdae, lfm, content, ratings, optimizer, recon_loss_fn, config, epochs, batch_size, device=None, max_iters=10):
    """
    Trains the CDL model. For best results, the SDAE should be pre-trained.

    Each training epoch consists of the following steps: (1) update V, (2)
    update U, (3) update W+ and b. In each step, we hold the parameters not
    being updated constant. When all but V is held constant, we can minimize V
    exactly. Similarly, when all but U is constant, we can minimize U exactly.
    In the last step, we batch the input and update W+ and b on each batch with
    one step of a gradient-based iterative algorithm.
    """
    lfm_optim = LatentFactorModelOptimizer(lfm, ratings, config['conf_a'], config['conf_b'], config['lambda_u'], config['lambda_v'])
    latent_loss_fn = AutoencoderLatentLoss(config['lambda_v'], config['lambda_n'])

    for epoch in range(epochs):
        logging.info(f'Starting epoch {epoch + 1}/{epochs}')

        # Update U and V.
        with autograd.no_grad():
            # Don't use dropout here.
            sdae.eval()
            latent_items_target, recon = sdae(content)
            latent_items_target = latent_items_target.cpu()
            sdae.train()

        lfm_optim.step(latent_items_target)

        if epoch % 3 == 0:
            loss = lfm_optim.loss(latent_items_target).item()
            loss += config['lambda_n'] / 2 * F.mse_loss(recon, content, reduction='sum').item()
            loss += config['lambda_w'] / 2 * (sum(w.square().sum() for w in sdae.weights) + sum(b.square().sum() for b in sdae.biases)).item()
            logging.info(f'  neg_likelihood: {loss}')

        # Update SDAE weights. Loss here only depends on SDAE outputs.
        train_cdl_autoencoder(sdae, content, lfm.V.to(device), config['corruption'], batch_size, recon_loss_fn, latent_loss_fn, optimizer)

    sdae.eval()
    latent_items_target = sdae.encode(content).cpu()

    # Now optimize U and V completely holding the SDAE latent layer fixed.
    prev_loss = None
    for i in range(max_iters):
        lfm_optim.step(latent_items_target)
        loss = lfm_optim.loss(latent_items_target)
        if prev_loss is not None and (prev_loss - loss) / loss < 1e-4:
            break

        prev_loss = loss


class LatentFactorModelOptimizer:
    """
    Computes latent user and latent item representations given a ratings target and a latent item representation target.
    """

    def __init__(self, model, ratings, conf_a, conf_b, lambda_u, lambda_v):
        self.model = model
        self.ratings = ratings

        self.conf_a = conf_a
        self.conf_b = conf_b

        self.lambda_u = lambda_u
        self.lambda_v = lambda_v

        # Precompute the nonzero entries of each row and column from the ratings matrix.
        self._ratings_nonzero_rows = [row.coalesce().indices().squeeze(0) for row in ratings]
        self._ratings_nonzero_cols = [col.coalesce().indices().squeeze(0) for col in ratings.t()]

    def step(self, latent_item_target):
        self.step_users()
        self.step_items(latent_item_target)

    def step_users(self):
        """Minimize the loss holding all but the latent users matrix constant."""
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
        U, V, conf_a, conf_b = self.model.U, self.model.V, self.conf_a, self.conf_b

        A_base = conf_b * V.t() @ V + self.lambda_u * torch.eye(self.model.latent_size)
        for j in range(len(U)):
            rated_idx = self._ratings_nonzero_rows[j]
            W = V[rated_idx, :]
            A = (conf_a - conf_b) * W.t() @ W + A_base
            # R[j, rated_idx] is just an all-ones vector
            b = conf_a * W.t().sum(dim=1)

            U[j] = linalg.solve(A, b)

    def step_items(self, latent_items_target):
        """Minimize the loss holding all but the latent items matrix constant."""
        U, V, conf_a, conf_b = self.model.U, self.model.V, self.conf_a, self.conf_b
        latent_items_target = latent_items_target * self.lambda_v

        A_base = conf_b * U.t() @ U + self.lambda_v * torch.eye(self.model.latent_size)
        for j in range(len(V)):
            rated_idx = self._ratings_nonzero_cols[j]
            if len(rated_idx) == 0:
                A = A_base
                b = latent_items_target[j]
            else:
                W = U[rated_idx, :]
                A = (conf_a - conf_b) * W.t() @ W + A_base
                # R[rated_idx, j] is just an all-ones vector
                b = conf_a * W.t().sum(dim=1) + latent_items_target[j]

            V[j] = linalg.solve(A, b)

    def loss(self, latent_items_target):
        conf_mat = self.conf_b * torch.ones(self.ratings.shape) + (self.conf_a - self.conf_b) * self.ratings

        loss_v = self.lambda_v * F.mse_loss(self.model.V, latent_items_target, reduction='sum') / 2
        loss_u = self.lambda_u * self.model.U.square().sum() / 2
        loss_r = (conf_mat * (self.model.predict() - self.ratings).square()).sum() / 2

        loss = loss_v + loss_u + loss_r
        logging.info(
            f'  lfm_neg_likelihood={loss:>5f}'
            f'  v={loss_v:>5f}'
            f'  u={loss_u:>5f}'
            f'  r={loss_r:>5f}'
        )
        return loss


def train_stacked_autoencoders(autoencoders, corruption, dataset, optimizer, loss_fn, epochs, batch_size):
    cur_dataset = dataset

    # Layer-wise pretraining.
    for i, autoencoder in enumerate(autoencoders):
        logging.info(f'Training autoencoder {i + 1}/{len(autoencoders)}')

        train_isolated_autoencoder(autoencoder, cur_dataset, corruption, epochs, batch_size, loss_fn, optimizer)

        with torch.no_grad():
            autoencoder.eval()
            cur_dataset = autoencoder.encode(cur_dataset[:])
            autoencoder.train()


def train_isolated_autoencoder(autoencoder, content, corruption, epochs, batch_size, loss_fn, optimizer):
    for epoch in range(epochs):
        logging.info(f'Starting epoch {epoch + 1}/{epochs}')

        dataset = data.TransformDataset(
            content,
            lambda x: (F.dropout(x, corruption), x),
        )
        train(lambda x: autoencoder(x)[1], dataset, loss_fn, batch_size, optimizer)


def train_cdl_autoencoder(autoencoder, content, latent_items, corruption, batch_size, recon_loss_fn, latent_loss_fn, optimizer):
    # Input to autoencoder is add_noise(item); target is (latent_item, item).
    dataset = data.TransformDataset(
        TensorDataset(latent_items, content),
        lambda x: (F.dropout(x[1], corruption), x),
    )

    def loss_fn(pred, target):
        latent_pred, recon_pred = pred
        latent_target, recon_target = target
        return recon_loss_fn(recon_pred, recon_target) + latent_loss_fn(latent_pred, latent_target)

    train(autoencoder, dataset, loss_fn, batch_size, optimizer)


def train(model, dataset, loss_fn, batch_size, optimizer):
    dataloader = DataLoader(dataset, batch_size)
    size = len(dataset)

    for i, (xb, yb) in enumerate(dataloader):
        yb_pred = model(xb)
        loss = loss_fn(yb_pred, yb)

        if i % 100 == 0:
            current = i * batch_size
            logging.info(f'  loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
