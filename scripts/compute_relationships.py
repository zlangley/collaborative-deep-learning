import os
import sys

import torch


def transform_file(dataset, filename_base, shape):
    indices = [[], []]

    with open(f'data/raw/{dataset}/{filename_base}-users.dat') as f:
        for i, line in enumerate(f):
            item_ids = line.split()[1:]

            for item_id in item_ids:
                indices[0].append(i)
                indices[1].append(int(item_id))

    values = [1] * len(indices[0])
    x = torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)

    base, _ = os.path.splitext(filename_base)
    torch.save(x, f'data/processed/{dataset}/{base}.pt')


if __name__ == '__main__':
    dataset_name = sys.argv[1]

    shape = {
        'citeulike-a': (5551, 16980),
        'citeulike-t': (7947, 25975),
    }

    for a in ['train', 'test']:
        for b in [1, 10]:
            transform_file(dataset_name, f'cf-{a}-{b}', shape[dataset_name])
