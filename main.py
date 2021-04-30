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
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--sdae_in', default='sdae.pt')
    parser.add_argument('--sdae_out', default='sdae.pt')
    parser.add_argument('--cdl_in', default='cdl.pt')
    parser.add_argument('--cdl_out', default='cdl.pt')
    parser.add_argument('--recall', type=int, default=300)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    if args.command not in ['train_sdae', 'train_cdl', 'predict']:
        print('unrecognized command')
        parser.print_help()
        sys.exit(1)

    if args.verbose:
        logging.basicConfig(format='%(asctime)s  %(message)s', datefmt='%I:%M:%S', level=logging.INFO)

    # Note: SDAE inputs and parameters will use the GPU if desired, but U and V
    # matrices of CDL do not go on the GPU (and therefore nor does the ratings
    # matrix).
    device = args.device
    logging.info(f'Using device {device}')

    content_dataset = data.read_mult_dat('data/citeulike-a/mult.dat')
    # dataset.shape: (16980, 8000)

    # FIXME: ratings data set only has 16970 articles
    content_dataset = content_dataset[:16970]

    content_training_dataset = content_dataset[:15282]
    content_validation_dataset = content_dataset[:15282]

    ratings_training_dataset = data.read_ratings('data/citeulike-a/cf-train-1-users.dat')
    ratings_test_dataset = data.read_ratings('data/citeulike-a/cf-test-1-users.dat')

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

    sdae_loss = train.sdae_loss(cdl.sdae, lambdas)
    optimizer = optim.Adam(cdl.parameters(), lr=1e-3)

    if args.command == 'train_sdae':
        save_path = args.sdae_out

        content_dataset = content_dataset.to(device)
        cdl.sdae.to(device)

        train_sdae(cdl.sdae, content_dataset, sdae_loss, optimizer, epochs=20, batch_size=60)
        logging.info(f'Finished training SDAE')

        cdl.sdae.cpu()

        torch.save(cdl.sdae.state_dict(), save_path)
        logging.info(f'Saved SDAE model to {save_path}.')
        sys.exit(0)

    elif args.command == 'train_cdl':
        load_path = args.sdae_in
        save_path = args.cdl_out

        state_dict = torch.load(load_path)
        cdl.sdae.load_state_dict(state_dict)

        content_dataset = content_dataset.to(device)
        cdl.sdae.to(device)

        dataset = train.ContentRatingsDataset(content_dataset, ratings_training_dataset)
        train_cdl(cdl, dataset, optimizer, conf=(1, 0.01), lambdas=lambdas, epochs=20, batch_size=60, device=device)
        logging.info(f'Finished training CDL')

        cdl.sdae.cpu()

        torch.save(cdl.state_dict(), save_path)
        logging.info(f'Saved CDL model to {save_path}')

    elif args.command == 'predict':
        load_path = args.cdl_in
        recall = args.recall

        state_dict = torch.load(load_path)
        cdl.load_state_dict(state_dict)
        cdl.eval()

        pred = cdl.predict()

        _, indices = torch.sort(pred, dim=1)
        top = indices[:, :args.recall]

        gathered = ratings_test_dataset.gather(1, top)
        recall = gathered.sum(dim=1) / ratings_test_dataset.sum(dim=1)

        print(f'recall@{args.recall}: {recall.mean().item()}')
