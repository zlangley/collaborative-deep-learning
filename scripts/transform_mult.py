import os

import torch


def transform_file(filename, shape):
    indices = [[], []]

    with open(f'data/raw/citeulike-a/{filename}') as f:
        for i, line in enumerate(f):
            item_ids = line.split()[1:]

            for item_id in item_ids:
                indices[0].append(i)
                indices[1].append(int(item_id))

    values = [1] * len(indices[0])
    x = torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)

    base, _ = os.path.splitext(filename)
    torch.save(x, f'data/processed/citeulike-a/{base}.pt')


if __name__ == '__main__':
    for a in ['train', 'test']:
        for b in [1, 10]:
            transform_file(f'cf-{a}-{b}-users.dat', (5551, 16980))
