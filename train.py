import torch
from torch import linalg
from torch import autograd
from torch.utils.data import DataLoader


def train_cdl(cdl, content_dataset, ratings_matrix, confidence_matrix, optimizer, epochs, batch_size):
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
        if epoch == epochs - 1:
            num_iters = 20
        else:
            num_iters = 1

        with autograd.no_grad():
            coordinate_ascent(cdl, ratings_matrix, confidence_matrix, encoded_dataset, num_iters)

        # This training will only tweak the SDAE parameters, but it will use the weights
        # U and V in the loss.
        train(cdl, content_dataset, batch_size, cdl_loss, optimizer)


def coordinate_ascent(cdl, R, C, enc, num_iters):
    """
    :param U: The latent users matrix of shape (num_users, latent_size).
    :param V: The latent items matrix of shape (num_items, latent_size).
    :param R: The ratings matrix of shape (num_users, num_items).
    :param C: The confidence matrix of shape (num_users, num_items).
    :param enc: The encodings of the content of shape (num_items, latent_size).
    """
    latent_size = cdl.U.shape[1]
    idu = cdl.lambda_u * torch.eye(latent_size)
    idv = cdl.lambda_v * torch.eye(latent_size)

    scaled_enc = enc * cdl.lambda_v

    for i in range(num_iters):
        VC = torch.empty(len(cdl.U), cdl.V.shape[1], cdl.V.shape[0])
        # VC.shape: (num_users, latent_size, num_users)
        for j in range(len(cdl.U)):
            VC[j] = cdl.V.t() * C[j]

        A = torch.matmul(VC, cdl.V)
        # A.shape: (num_users, latent_size, latent_size)
        B = torch.matmul(VC, R)
        # B.shape: (num_users, latent_size, num_items)
        cdl.U = linalg.solve(A, B)

        UC = torch.tempty(len(cdl.V), cdl.U.shape[1], cdl.U.shape[0])
        for j in range(len(cdl.V)):
            UC[j] = cdl.U.t() * C[:, j]

        A = torch.matmul(UC, cdl.U)
        B = torch.matmul(UC, R.t()) + scaled_enc
        cdl.V = linalg.solve(A, B)


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
