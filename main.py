import torch
import torch.nn as nn
import torch.optim as optim

import data
from cdl import CollaborativeDeepLearning
from sdae import StackedDenoisingAutoencoder
from train import train_sdae, train_cdl

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda:3'
    else:
        device = 'cpu'

    content_dataset = data.read_mult_dat('data/citeulike-a/mult.dat').to(device)
    # dataset.shape: (16980, 8000)

    # FIXME: ratings data set only has 16970 articles
    content_dataset = content_dataset[:16970]

    content_training_dataset = content_dataset[:15282]
    content_validation_dataset = content_dataset[:15282]

    ratings_training_dataset = data.read_ratings('data/citeulike-a/cf-train-1-users.dat').to(device)

    cdl = CollaborativeDeepLearning(
        in_features=content_training_dataset.shape[1],
        num_users=ratings_training_dataset.shape[0],
        num_items=ratings_training_dataset.shape[1],
        layer_sizes=[200, 50],
        corruption=0.3,
        dropout=0.0,
        lambda_u=0.01,
        lambda_v=100.0,
        lambda_n=100.0,
        lambda_w=1.0,
    ).to(device)

    conf = (1, 0.01)

    def sdae_loss(pred, actual):
        # pred = torch.clamp(pred, min=1e-16)
        # actual = torch.clamp(actual, min=1e-16)
        # cross_entropies = -(actual * torch.log(pred) + (1 - actual) * torch.log(1 - pred)).sum(dim=1)
        # return cross_entropies.mean()

        loss = 0
        for param in cdl.sdae.parameters():
            loss += (param * param).sum() * cdl.lambda_w / 2

        loss += ((pred - actual) ** 2).sum() * cdl.lambda_n / 2
        return loss

    optimizer = optim.Adam(cdl.parameters())

    load_pretrain = True
    if load_pretrain:
        state_dict = torch.load('sdae.pt', map_location=device)
        cdl.sdae.load_state_dict(state_dict)
    else:
        print('Pretraining...')
        train_sdae(cdl.sdae, content_dataset, sdae_loss, optimizer, epochs=20, batch_size=60)
        torch.save(cdl.sdae.state_dict(), 'sdae.pt')

    cdl.sdae.eval()
    x = cdl.sdae(content_validation_dataset.to(device))
    print('sdae validation loss', sdae_loss(x, content_validation_dataset))

    cdl.sdae.train()
    train_cdl(cdl, content_dataset, ratings_training_dataset, optimizer, conf=(1, 0.01), epochs=10, batch_size=60)
    torch.save(cdl.state_dict(), 'cdl.pt')

    ratings_pred = cdl.U @ cdl.V.t()
    print(ratings_training_dataset[0])
    print(ratings_pred[0])