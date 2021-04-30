import torch
import torch.cuda
from torch import linalg
from torch import autograd
from torch.utils.data import DataLoader


def train_cdl(cdl, content_dataset, ratings_matrix, optimizer, conf, epochs, batch_size):
    confidence_matrix = ratings_matrix * (conf[0] - conf[1]) + conf[1] * torch.ones_like(ratings_matrix)

    with autograd.no_grad():
        encoded_dataset = cdl.sdae.encode(content_dataset)

    def cdl_loss(cdl_out, actual):
        reconstruction, ratings_pred = cdl_out

        loss = 0
        for param in cdl.sdae.parameters():
            loss += (param ** 2).sum() * cdl.lambda_w / 2

        loss += ((reconstruction - actual) ** 2).sum() * cdl.lambda_n / 2
        loss += (cdl.U ** 2).sum() * cdl.lambda_u / 2
        loss += ((cdl.V - encoded_dataset) ** 2).sum() * cdl.lambda_v / 2
        loss += (confidence_matrix * (ratings_matrix - ratings_pred) ** 2).sum() / 2

        return loss

    print('Training CDL')
    for epoch in range(epochs):
        print('Epoch', epoch + 1)

        # Each epoch is one iteration of coordinate ascent (for U and V)
        # followed by one iteration of gradient descent (for the SDAE
        # parameters).

        with autograd.no_grad():
            print('  running coordinate ascent...')
            coordinate_ascent(cdl, ratings_matrix, conf, encoded_dataset, num_iters)

        # The parameters U and V are not updated below since they have
        # require_grad=False.  However, their values will influence the loss
        # and therefore the gradients of the SDAE.
        print('  running gradient descent...')
        train(cdl, content_dataset, batch_size, cdl_loss, optimizer)


def coordinate_ascent(cdl, R, conf, enc, num_iters=1):
    """
    :param U: The latent users matrix of shape (num_users, latent_size).
    :param V: The latent items matrix of shape (num_items, latent_size).
    :param R: The ratings matrix of shape (num_users, num_items).
    :param C: The confidence matrix of shape (num_users, num_items).
    :param enc: The encodings of the content of shape (num_items, latent_size).
    """
    # Because of the iterations below, coordinate ascent is much faster on the CPU.
    orig_device = cdl.U.device

    V = cdl.V.data.to('cpu')
    U = cdl.U.data.to('cpu')
    R = R.to('cpu')
    enc = enc.to('cpu')

    latent_size = U.shape[1]
    idu = cdl.lambda_u * torch.eye(latent_size, device=R.device)
    idv = cdl.lambda_v * torch.eye(latent_size, device=R.device)
    conf_a, conf_b = conf

    scaled_enc = enc * cdl.lambda_v

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

    cdl.V.data = V.to(orig_device)
    cdl.U.data = U.to(orig_device)


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
