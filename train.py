import torch
from torch.utils.data import DataLoader


def train_sdae(sdae, dataset, loss_fn, optimizer, epochs, batch_size):
    cur_input = dataset

    # First do layer-wise training.
    for i, autoencoder in enumerate(sdae.autoencoders):
        print(f'Training DAE {i}')
        train(autoencoder, cur_input, batch_size, epochs, loss_fn, optimizer)

        with torch.no_grad():
            cur_input = autoencoder.encode(cur_input)

    # Now train it end-to-end.
    print(f'Training SDAE')
    train(sdae, dataset, batch_size, epochs, loss_fn, optimizer)


def train(model, dataset, batch_size, epochs, loss_fn, optimizer):
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')

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
