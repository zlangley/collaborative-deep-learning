import torch


def preprocess_content():
    indices = [[], []]
    values = []

    with open('data/raw/citeulike-a/mult.dat') as f:
        for doc_id, line in enumerate(f):
            tokens = line.split()[1:]

            for token in tokens:
                word_id, cnt = tuple(map(int, token.split(':')))
                indices[0].append(doc_id)
                indices[1].append(word_id)
                values.append(cnt)

    x = torch.sparse_coo_tensor(indices, values, (16980, 8000), dtype=torch.float32).to_dense()

    maxes, _ = x.max(dim=1, keepdim=True)
    x /= maxes

    torch.save(x.to_sparse(), 'data/processed/citeulike-a/content.pt')


def preprocess_ratings_file(filename, shape):
    indices = [[], []]

    with open(f'data/raw/citeulike-a/{filename}') as f:
        for i, line in enumerate(f):
            item_ids = line.split()[1:]

            for item_id in item_ids:
                indices[0].append(i)
                indices[1].append(int(item_id))

    values = [1] * len(indices[0])

    x = torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)
    torch.save(x, f'data/processed/citeulike-a/{filename}')


if __name__ == '__main__':
    preprocess_content()

    for a in ['train', 'test']:
        for b in [1, 10]:
            for c, shape in [('users', (5551, 16980)), ('items', (16980, 5551))]:
                preprocess_ratings_file(f'cf-{a}-{b}-{c}.dat', shape)
