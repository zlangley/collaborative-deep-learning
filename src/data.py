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


def load_content_embeddings(use_bert=True, device=None):
    if use_bert:
        return torch.load('data/processed/citeulike-a/content-bert.pt', map_location=device)
    else:
        return torch.load('data/processed/citeulike-a/content-bow.pt', map_location=device).to_dense()


def load_cf_data():
    return (
        torch.load('data/processed/citeulike-a/cf-train-1-users.pt'),
        torch.load('data/processed/citeulike-a/cf-test-1-users.pt'),
    )