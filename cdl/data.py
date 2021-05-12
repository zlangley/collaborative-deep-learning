import torch


class TransformDataset:
    def __init__(self, dataset, transform):
        self._dataset = dataset
        self._transform = transform

    def __getitem__(self, i):
        return self._transform(self._dataset[i])

    def __len__(self):
        return len(self._dataset)


def random_subset(x, k):
    idx = torch.randperm(len(x))[:k]
    return torch.utils.data.Subset(x, idx)


def load_content_embeddings(embedding, device=None):
    x = torch.load(f'data/processed/citeulike-a/content-{embedding}.pt', map_location=device)

    if x.is_sparse:
        x = x.to_dense()

    return x


def load_cf_train_data():
    return torch.load('data/processed/citeulike-a/cf-train-1-users.pt')


def load_cf_test_data():
    return torch.load('data/processed/citeulike-a/cf-test-1-users.pt')


def save_model(sdae, lfm, filename):
    torch.save({
        'autoencoder': sdae.state_dict(),
        'latent_factor_model': lfm.state_dict(),
    }, filename)


def load_model(sdae, lfm, filename):
    d = torch.load(filename)

    if sdae is not None:
        sdae.update_state_dict(d['autoencoder'])

    if lfm is not None:
        lfm.update_state_dict(d['latent_factor_model'])
