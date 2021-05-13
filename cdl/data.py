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


def bernoulli_corrupt(x, p):
    mask = torch.rand_like(x) > p
    return x * mask


def load_content_embeddings(dataset_name, embedding, device=None):
    x = torch.load(f'data/processed/{dataset_name}/content-{embedding}.pt', map_location=device)

    if x.is_sparse:
        x = x.to_dense()

    return x


def load_cf_train_data(dataset_name):
    return torch.load(f'data/processed/{dataset_name}/cf-train-1.pt')


def load_cf_test_data(dataset_name):
    return torch.load(f'data/processed/{dataset_name}/cf-test-1.pt')


def save_model(sdae, mfm, filename):
    torch.save({
        'autoencoder': sdae.state_dict(),
        'matrix_factorization_model': mfm.state_dict(),
    }, filename)


def load_model(sdae, mfm, filename):
    d = torch.load(filename)

    if sdae is not None:
        sdae.update_state_dict(d['autoencoder'])

    if mfm is not None:
        mfm.update_state_dict(d['matrix_factorization_model'])
