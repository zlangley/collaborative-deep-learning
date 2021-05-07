import argparse
import logging
import os
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
from ray import tune

import data
import evaluate
from autoencoder import Autoencoder, StackedAutoencoder
from cdl import LatentFactorModel
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


sdae_activations = {
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
}

recon_losses = {
    'mse': nn.MSELoss(),
    'cross-entropy': nn.BCEWithLogitsLoss(),
}


def make_stacked_autoencoder(in_features, hidden_sizes, latent_size, activation):
    layer_sizes = [in_features] + hidden_sizes + [latent_size]
    activation = sdae_activations[activation]

    autoencoders = [
        Autoencoder(in_features, out_features, config['dropout'], activation, tie_weights=True)
        for in_features, out_features in zip(layer_sizes, layer_sizes[1:])
    ]
    return StackedAutoencoder(autoencoder_stack=autoencoders)


def train_cdl(config, checkpoint_dir=None, data_dir=None):
    content_dataset, ratings_training_dataset, _ = data.load_data(data_dir)
    num_items, in_features = content_dataset.shape

    sdae = make_stacked_autoencoder(in_features, config['hidden_sizes'], config['latent_size'], config['activation'])

    #    logging.info(f'Using autoencoder architecture {"x".join(map(str, layer_sizes))}')

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            sdae = nn.DataParallel(sdae)
        sdae.to(device)

    lfm = LatentFactorModel(
        target_shape=ratings_training_dataset.shape,
        latent_size=config['latent_size'],
    )

    #    logging.info(f'Config: {config}')
    optimizer = optim.AdamW(sdae.parameters(), lr=config['lr'], weight_decay=config['lambda_w'])

    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, 'checkpoint')
        sdae_state, lfm_items, lfm_users, optimizer_state = torch.load(checkpoint)
        sdae_state.load_state_dict(sdae_state)
        lfm.V = lfm_items
        lfm.U = lfm_users
        optimizer.load_state_dict(optimizer_state)

    recon_loss_fn = recon_losses[config['recon_loss']]
    content_pretraining_dataset = data.random_subset(content_dataset, int(num_items * 0.8))

    #    logging.info(f'Pretraining SDAE with {args.recon_loss} loss')
    pretrain_sdae(sdae, args.corruption, content_pretraining_dataset, optimizer, recon_loss_fn, epochs=config['pretrain_epochs'], batch_size=config['batch_size'])

    #    logging.info(f'Saving pretrained SDAE to {args.sdae_out}.')
    #    torch.save(sdae.state_dict(), args.sdae_out)

    #    logging.info(f'Training with recon loss {args.recon_loss}')
    recon_loss_fn = recon_losses[config['recon_loss']]

    train_model(sdae, lfm, content_dataset, ratings_training_dataset, optimizer, recon_loss_fn, config, epochs=config['epochs'], batch_size=config['batch_size'], device=device)

#    logging.info(f'Saving model to {args.cdl_out}')
#    save_model(args.cdl_out, sdae, lfm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Collaborative Deep Learning implementation.')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--sdae_in')
    parser.add_argument('--sdae_out', default='sdae.pt')
    parser.add_argument('--cdl_in')
    parser.add_argument('--cdl_out', default='cdl.pt')

    parser.add_argument('--recall', type=int, default=300)

    parser.add_argument('--conf_a', type=float, default=1.0)
    parser.add_argument('--conf_b', type=float, default=0.01)

    parser.add_argument('--lambda_u', type=float, default=0.1)
    parser.add_argument('--lambda_v', type=float, default=10.0)
    parser.add_argument('--lambda_w', type=float, default=0.01)
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

    config = {
        'conf_a': args.conf_a,
        'conf_b': args.conf_b,
        'lambda_u': args.lambda_u,
        'lambda_v': args.lambda_v,
        'lambda_w': args.lambda_u,
        'lambda_n': args.lambda_n,
        'dropout': args.dropout,
        'corruption': args.corruption,
        'lr': args.lr,

        'pretrain_epochs': args.epochs,
        'epochs': args.epochs,
        'batch_size': args.batch_size,

        'recon_loss': args.recon_loss,
        'activation': args.activation,

        'hidden_sizes': args.hidden_sizes,
        'latent_size': args.latent_size,
    }
    data_dir = '../../data/citeulike-a'
    result = tune.run(
        partial(train_cdl, data_dir=data_dir),
        name='cdl',
        local_dir=os.getcwd(),
        config=config,
    )
    best_trial = result.get_best_trial()
    print("Best trial config: {}".format(best_trial.config))

    # Shape irrelevant because we'll override below.
    lfm = LatentFactorModel((1, 1), best_trial.config['latent_size'])

    checkpoint_path = os.path.join(best_trial.checkpoint.value, "checkpoint")

    _, lfm_items, lfm_users, _ = torch.load(checkpoint_path)
    lfm.V = lfm_items
    lfm.U = lfm_users

    logging.info(f'Predicting')
    pred = lfm.predict()

    logging.info(f'Calculating recall@{args.recall}')
    ratings_test_dataset = data.read_ratings(f'{data_dir}/cf-test-1-users.dat', 19680)
    recall = evaluate.recall(pred, ratings_test_dataset, args.recall)

    print(f'recall@{args.recall}: {recall.item()}')



