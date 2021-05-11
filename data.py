import scipy.io
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


def read_ratings(filename, num_items, map_location=None):
    adj = []

    with open(filename) as f:
        for line in f:
            nums = [int(x) for x in line.split()[1:]]
            adj.append(nums)

    R = torch.zeros((len(adj), num_items), device=map_location)

    # TODO: use torch.scatter?
    for u, vs in enumerate(adj):
        for v in vs:
            R[u, v] = 1

    return R
