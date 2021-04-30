import argparse
import logging
import sys

import torch
import torch.optim as optim

import data
import train
from cdl import CollaborativeDeepLearning
from train import train_sdae, train_cdl


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Collaborative Deep Learning implementation.')
    parser.add_argument('command')
    parser.add_argument('--device')
    parser.add_argument('--sdae')
    parser.add_argument('--out')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    if args.command not in ['train_sdae', 'train_cdl']:
        print('unrecognized command')
        parser.print_help()
        sys.exit(1)

    if args.verbose:
        logging.basicConfig(format='%(asctime)s  %(message)s', datefmt='%I:%M:%S', level=logging.INFO)

    # Note: SDAE inputs and parameters will use the GPU if desired, but U and V
    # matrices of CDL do not go on the GPU (and therefore nor does the ratings
    # matrix).
    device = args.device or 'cpu'
    logging.info(f'Using device {device}')

    content_dataset = data.read_mult_dat('data/citeulike-a/mult.dat').to(device)
    # dataset.shape: (16980, 8000)

    # FIXME: ratings data set only has 16970 articles
    content_dataset = content_dataset[:16970]

    content_training_dataset = content_dataset[:15282]
    content_validation_dataset = content_dataset[:15282]

    ratings_training_dataset = data.read_ratings('data/citeulike-a/cf-train-1-users.dat')

    lambdas = train.Lambdas(
        u=0.01,
        v=100.0,
        n=100.0,
        w=1.0,
    )

    cdl = CollaborativeDeepLearning(
        in_features=content_training_dataset.shape[1],
        num_users=ratings_training_dataset.shape[0],
        num_items=ratings_training_dataset.shape[1],
        layer_sizes=[200, 50],
        corruption=0.3,
        dropout=0.0,
    )
    # Only move the SDAE.
    cdl.sdae.to(device)

    sdae_loss = train.sdae_loss(cdl.sdae, lambdas)
    optimizer = optim.Adam(cdl.parameters(), lr=1e-3)

    if args.command == 'train_sdae':
        save_path = args.out or 'sdae.pt'

        train_sdae(cdl.sdae, content_dataset, sdae_loss, optimizer, epochs=20, batch_size=60)
        logging.info(f'Finished training SDAE')

        torch.save(cdl.sdae.state_dict(), save_path)
        logging.info(f'Saved SDAE model to {save_path}.')
        sys.exit(0)

    if args.command == 'train_cdl':
        load_path = args.sdae or 'sdae.pt'
        save_path = args.out or 'cdl.pt'

        state_dict = torch.load(load_path, map_location=device)
        cdl.sdae.load_state_dict(state_dict)

        dataset = train.ContentRatingsDataset(content_dataset, ratings_training_dataset)
        train_cdl(cdl, dataset, optimizer, conf=(1, 0.01), lambdas=lambdas, epochs=20, batch_size=60, device=device)
        logging.info(f'Finished training CDL')

        torch.save(cdl.state_dict(), save_path)
        logging.info(f'Saved CDL model to {save_path}')
