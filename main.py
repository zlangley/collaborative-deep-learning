import scipy.io

import argparse
import logging
import sys

import torch
import torch.nn as nn
import torch.optim as optim

import data
import evaluate
import train
from mf import MatrixFactorizationModel
from sdae import Autoencoder, StackedAutoencoder
from train import pretrain_sdae, train_model


def load_model(filename, sdae, mf, map_location=None):
    checkpoint = torch.load(filename, map_location=map_location)
    sdae.load_state_dict(checkpoint['sdae'])
    mf.U = checkpoint['U']
    mf.V = checkpoint['V']


def save_model(filename, sdae, mf):
    torch.save({
        'sdae': sdae.cpu().state_dict(),
        'U': mf.U,
        'V': mf.V,
    }, filename)


def print_params(args):
    logging.info(f'Parameters:')
    logging.info(f'         a: {args.conf_a}')
    logging.info(f'         b: {args.conf_b}')
    logging.info(f'  lambda_u: {args.lambda_u}')
    logging.info(f'  lambda_v: {args.lambda_v}')
    logging.info(f'  lambda_w: {args.lambda_w}')
    logging.info(f'  lambda_r: {args.lambda_r}')


sdae_activations = {
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
}

recon_losses = {
    'mse': nn.MSELoss(),
    'cross-entropy': nn.BCEWithLogitsLoss(),
}


def regularize_sdae_loss(sdae, loss, lambda_w):
    return lambda pred, actual: loss(pred, actual) + sdae.regularization_term(lambda_w)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Collaborative Deep Learning implementation.')
    parser.add_argument('command', choices=['pretrain_sdae', 'train', 'predict'])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=0)
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
    parser.add_argument('--lambda_r', type=float, default=10000.0)
    parser.add_argument('--lambda_w', type=float, default=1.0)

    parser.add_argument('--epochs', type=int, default=10)

    # SDAE hyperparameters
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--corruption', type=float, default=0.3)
    parser.add_argument('--activation', choices=sdae_activations.keys(), default='sigmoid')
    parser.add_argument('--recon_loss', choices=recon_losses.keys(), default='cross-entropy')
    parser.add_argument('--hidden_sizes', nargs='+', type=int, default=[50])
    parser.add_argument('--latent_size', type=int, default=50)
    parser.add_argument('--no_tie_weights', action='store_true')

    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.verbose:
        logging.basicConfig(format='%(asctime)s  %(message)s', datefmt='%I:%M:%S', level=logging.INFO)

    # Note: SDAE inputs and parameters will use the GPU if desired, but U and V
    # matrices of CDL do not go on the GPU (and therefore nor does the ratings
    # matrix).
    device = args.device
    logging.info(f'Using device {device}')

    logging.info('Loading dataset')
    variables = scipy.io.loadmat("data/citeulike-a/mult_nor.mat")
    content_dataset = torch.from_numpy(variables['X']).float().to(device)
    num_items = content_dataset.shape[0]
    # dataset.shape: (16980, 8000)

    content_training_dataset = content_dataset[:15282]
    content_validation_dataset = content_dataset[:15282]

    ratings_training_dataset = data.read_ratings('data/citeulike-a/cf-train-1-users.dat', num_items)
    ratings_test_dataset = data.read_ratings('data/citeulike-a/cf-test-1-users.dat', num_items)

    lambdas = train.Lambdas(
        u=args.lambda_u,
        v=args.lambda_v,
        r=args.lambda_r,
        w=args.lambda_w,
    )

    layer_sizes = [content_training_dataset.shape[1]] + args.hidden_sizes + [args.latent_size]
    logging.info(f'Using autoencoder architecture {"x".join(map(str, layer_sizes))}')

    activation = sdae_activations[args.activation]

    autoencoders = []
    for in_features, out_features in zip(layer_sizes[:-1], layer_sizes[1:-1]):
        autoencoders.append(Autoencoder(in_features, out_features, args.dropout, activation, tie_weights=True))

    # [?] Don't use activation function for latent layer.
    # [?] Don't tie weights in latent layer.
    autoencoders.append(Autoencoder(layer_sizes[-2], layer_sizes[-1], args.dropout, nn.Identity(), tie_weights=False))

    sdae = StackedAutoencoder(autoencoder_stack=autoencoders)

    mf = MatrixFactorizationModel(
        target_shape=ratings_training_dataset.shape,
        latent_size=args.latent_size,
    )

    optimizer = optim.Adam(sdae.parameters(), lr=args.lr)

    if args.command == 'pretrain_sdae':
        print_params(args)

        if args.resume:
            logging.info(f'Loading pre-trained SDAE from {args.sdae_in}')
            sdae.load_state_dict(torch.load(args.sdae_in))

        sdae.to(device)

        logging.info(f'Pretraining SDAE with {args.recon_loss} loss')
        loss_fn = regularize_sdae_loss(sdae, recon_losses[args.recon_loss], args.lambda_w)
        pretrain_sdae(sdae, args.corruption, content_dataset, optimizer, loss_fn, epochs=args.epochs, batch_size=args.batch_size)

        logging.info(f'Saving pretrained SDAE to {args.sdae_out}.')
        sdae.cpu()
        torch.save(sdae.state_dict(), args.sdae_out)

    elif args.command == 'train':
        print_params(args)

        if args.resume:
            logging.info(f'Loading pre-trained MF model from {args.cdl_in}')
            load_model(args.cdl_in, sdae, mf)
        else:
            logging.info(f'Loading pre-trained SDAE from {args.sdae_in}')
            sdae.load_state_dict(torch.load(args.sdae_in))

        sdae.train()
        sdae.to(device)

        logging.info(f'Training with recon loss {args.recon_loss}')
        recon_loss_fn = regularize_sdae_loss(sdae, recon_losses[args.recon_loss], args.lambda_w)
        dataset = train.ContentRatingsDataset(content_dataset, ratings_training_dataset)
        train_model(sdae, mf, args.corruption, dataset, optimizer, recon_loss_fn, conf=(args.conf_a, args.conf_b), lambdas=lambdas, epochs=args.epochs, batch_size=args.batch_size, device=device)

        logging.info(f'Saving model to {args.cdl_out}')
        save_model(args.cdl_out, sdae, mf)

    elif args.command == 'predict':
        load_path = args.cdl_in
        recall = args.recall

        logging.info(f'Loading model from {load_path}')
        load_model(load_path, sdae, mf)

        logging.info(f'Predicting')
        pred = mf.predict()

        logging.info(f'Calculating recall@{args.recall}')
        recall = evaluate.recall(pred, ratings_test_dataset, args.recall)

        print(f'recall@{args.recall}: {recall.item()}')

    logging.info(f'Complete')
