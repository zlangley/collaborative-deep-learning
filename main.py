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
from lfm import LatentFactorModel
from autoencoder import Autoencoder, StackedAutoencoder
from train import pretrain_sdae, train_model


def load_model(filename, sdae, lfm, map_location=None):
    checkpoint = torch.load(filename, map_location=map_location)
    sdae.load_state_dict(checkpoint['sdae'])
    lfm.U = checkpoint['U']
    lfm.V = checkpoint['V']


def save_model(filename, sdae, lfm):
    torch.save({
        'sdae': sdae.cpu().state_dict(),
        'U': lfm.U,
        'V': lfm.V,
    }, filename)


def print_params(args):
    logging.info(f'Parameters:')
    logging.info(f'         a: {args.conf_a}')
    logging.info(f'         b: {args.conf_b}')
    logging.info(f'  lambda_u: {args.lambda_u}')
    logging.info(f'  lambda_v: {args.lambda_v}')
    logging.info(f'  lambda_w: {args.lambda_w}')
    logging.info(f'  lambda_n: {args.lambda_n}')
    logging.info(f'  lambda_r: {args.lambda_r}')


sdae_activations = {
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
}

recon_losses = {
    'mse': nn.MSELoss(reduction='sum'),
    'cross-entropy': nn.BCEWithLogitsLoss(),
}


def regularize_autoencoder_loss(autoencoder, base_loss, lambda_n, lambda_w):
    def _loss(pred, actual):
        loss = 0
        loss += lambda_n * base_loss(pred, actual)
        loss += lambda_w * sum(weight.square().sum() for weight in autoencoder.weights)
        loss += lambda_w * sum(bias.square().sum() for bias in autoencoder.biases)
        return loss / 2

    return _loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Collaborative Deep Learning implementation.')
    parser.add_argument('command', choices=['pretrain_sdae', 'train', 'predict'])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--sdae_in', default='sdae.pt')
    parser.add_argument('--sdae_out', default='sdae.pt')
    parser.add_argument('--cdl_in', default='cdl.pt')
    parser.add_argument('--cdl_out', default='cdl.pt')

    parser.add_argument('--recall', type=int, default=300)

    parser.add_argument('--conf_a', type=float, default=1.0)
    parser.add_argument('--conf_b', type=float, default=0.01)

    parser.add_argument('--lambda_u', type=float, default=0.1)
    parser.add_argument('--lambda_v', type=float, default=10.0)
    parser.add_argument('--lambda_w', type=float, default=0.1)
    parser.add_argument('--lambda_n', type=float, default=1000.0)
    parser.add_argument('--lambda_r', type=float, default=1.0)

    parser.add_argument('--epochs', type=int, default=10)

    # SDAE hyperparameters
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--corruption', type=float, default=0.3)
    parser.add_argument('--activation', choices=sdae_activations.keys(), default='sigmoid')
    parser.add_argument('--recon_loss', choices=recon_losses.keys(), default='mse')
    parser.add_argument('--hidden_sizes', nargs='+', type=int, default=[200])
    parser.add_argument('--latent_size', type=int, default=50)

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

    logging.info('Loading content dataset')
    content_dataset = data.read_mult_norm_dat('data/citeulike-a/mult_nor.mat').to(device)
    num_items, in_features = content_dataset.shape
    # content_dataset.shape: (16980, 8000)

    logging.info('Loading ratings dataset')
    ratings_training_dataset = data.read_ratings('data/citeulike-a/cf-train-1-users.dat', num_items)
    ratings_test_dataset = data.read_ratings('data/citeulike-a/cf-test-1-users.dat', num_items)

    lambdas = train.Lambdas(
        u=args.lambda_u,
        v=args.lambda_v,
        r=args.lambda_r,
        w=args.lambda_w,
    )

    layer_sizes = [in_features] + args.hidden_sizes + [args.latent_size]
    logging.info(f'Using autoencoder architecture {"x".join(map(str, layer_sizes))}')

    activation = sdae_activations[args.activation]

    autoencoders = [
        Autoencoder(in_features, out_features, args.dropout, activation, tie_weights=True)
        for in_features, out_features in zip(layer_sizes, layer_sizes[1:])
    ]
    sdae = StackedAutoencoder(autoencoder_stack=autoencoders)

    lfm = LatentFactorModel(
        target_shape=ratings_training_dataset.shape,
        latent_size=args.latent_size,
    )

    optimizer = optim.Adam(sdae.parameters(), lr=args.lr)

    if args.command == 'pretrain_sdae':
        print_params(args)

        sdae.to(device)

        content_dataset = data.random_subset(content_dataset, int(num_items * 0.8))

        logging.info(f'Pretraining SDAE with {args.recon_loss} loss')
        recon_loss_fn = regularize_autoencoder_loss(sdae, recon_losses[args.recon_loss], args.lambda_n, args.lambda_w)
        pretrain_sdae(sdae, args.corruption, content_dataset, optimizer, recon_loss_fn, epochs=args.epochs, batch_size=args.batch_size)

        logging.info(f'Saving pretrained SDAE to {args.sdae_out}.')
        sdae.cpu()
        torch.save(sdae.state_dict(), args.sdae_out)

    elif args.command == 'train':
        print_params(args)

        logging.info(f'Loading pre-trained SDAE from {args.sdae_in}')
        sdae.load_state_dict(torch.load(args.sdae_in))
        sdae.train()
        sdae.to(device)

        content_dataset = content_dataset.to(device).float()

        logging.info(f'Training with recon loss {args.recon_loss}')
        recon_loss_fn = regularize_autoencoder_loss(sdae, recon_losses[args.recon_loss], args.lambda_n, args.lambda_w)
        train_model(sdae, lfm, args.corruption, content_dataset, ratings_training_dataset, optimizer, recon_loss_fn, conf=(args.conf_a, args.conf_b), lambdas=lambdas, epochs=args.epochs, batch_size=args.batch_size, device=device)

        logging.info(f'Saving model to {args.cdl_out}')
        save_model(args.cdl_out, sdae, lfm)

    elif args.command == 'predict':
        load_path = args.cdl_in
        recall = args.recall

        logging.info(f'Loading model from {load_path}')
        load_model(load_path, sdae, lfm)

        logging.info(f'Predicting')
        pred = lfm.predict()

        logging.info(f'Calculating recall@{args.recall}')
        recall = evaluate.recall(pred, ratings_test_dataset, args.recall)

        print(f'recall@{args.recall}: {recall.item()}')

    logging.info(f'Complete')
