import argparse
import logging
import sys

import torch
import torch.optim as optim

import data
import evaluate
import train
from cdl import CollaborativeDeepLearning
from train import train_sdae, train_cdl


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Collaborative Deep Learning implementation.')
    parser.add_argument('command')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--resume', action='store_true')

    parser.add_argument('--sdae_in', default='sdae.pt')
    parser.add_argument('--sdae_out', default='sdae.pt')
    parser.add_argument('--cdl_in', default='cdl.pt')
    parser.add_argument('--cdl_out', default='cdl.pt')

    parser.add_argument('--recall', type=int, default=300)

    parser.add_argument('--conf_a', type=float, default=1.0)
    parser.add_argument('--conf_b', type=float, default=0.01)

    parser.add_argument('--lambda_u', type=float, default=0.1)
    parser.add_argument('--lambda_v', type=float, default=10.0)
    parser.add_argument('--lambda_n', type=float, default=1.0)
    parser.add_argument('--lambda_w', type=float, default=1.0)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--corruption', type=float, default=0.3)

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

    logging.info('Loading dataset')
    content_dataset = data.read_mult_dat('data/citeulike-a/mult.dat', map_location=device)
    # dataset.shape: (16980, 8000)

    # FIXME: the ratings matrix only has 16970 items...
    content_dataset = content_dataset[:16970]

    content_training_dataset = content_dataset[:15282]
    content_validation_dataset = content_dataset[:15282]

    ratings_training_dataset = data.read_ratings('data/citeulike-a/cf-train-1-users.dat')
    ratings_test_dataset = data.read_ratings('data/citeulike-a/cf-test-1-users.dat')

    lambdas = train.Lambdas(
        u=args.lambda_u,
        v=args.lambda_v,
        n=args.lambda_n,
        w=args.lambda_w,
    )

    cdl = CollaborativeDeepLearning(
        in_features=content_training_dataset.shape[1],
        num_users=ratings_training_dataset.shape[0],
        num_items=ratings_training_dataset.shape[1],
        layer_sizes=[100, 50],
        corruption=args.corruption,
        dropout=args.dropout,
    )

    optimizer = optim.Adam(cdl.parameters(), lr=args.lr)

    if args.command == 'train_sdae':
        if args.resume:
            logging.info(f'Loading pre-trained SDAE from {args.sdae_in}')
            cdl.sdae.load_state_dict(torch.load(args.sdae_in))

        cdl.sdae.to(device)

        logging.info(f'Training SDAE')
        train_sdae(cdl.sdae, content_dataset, lambdas, optimizer, epochs=args.epochs, batch_size=args.batch_size)

        logging.info(f'Saving SDAE model to {args.sdae_out}.')
        cdl.sdae.cpu()
        torch.save(cdl.sdae.state_dict(), args.sdae_out)

    elif args.command == 'train_cdl':
        if args.resume:
            logging.info(f'Loading pre-trained CDL from {args.cdl_in}')
            cdl.load_state_dict(torch.load(args.cdl_in))
        else:
            logging.info(f'Loading pre-trained SDAE from {args.sdae_in}')
            cdl.sdae.load_state_dict(torch.load(args.sdae_in))

        cdl.sdae.to(device)

        logging.info(f'Training CDL')
        dataset = train.ContentRatingsDataset(content_dataset, ratings_training_dataset)
        train_cdl(cdl, dataset, optimizer, conf=(args.conf_a, args.conf_b), lambdas=lambdas, epochs=args.epochs, batch_size=args.batch_size, device=device)

        logging.info(f'Saving CDL model to {args.cdl_out}')
        cdl.sdae.cpu()
        torch.save(cdl.state_dict(), args.cdl_out)

    elif args.command == 'predict':
        load_path = args.cdl_in
        recall = args.recall

        logging.info(f'Loading CDL from {load_path}')
        state_dict = torch.load(load_path, map_location=device)
        cdl.load_state_dict(state_dict)
        cdl.eval()

        logging.info(f'Predicting')
        pred = cdl.predict()

        logging.info(f'Calculating recall@{args.recall}')

        recall = evaluate.recall(pred, ratings_test_dataset, args.recall)

        print(f'recall@{args.recall}: {recall.item()}')

    logging.info(f'Complete')
