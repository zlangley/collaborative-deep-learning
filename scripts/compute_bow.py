import torch


if __name__ == '__main__':
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

    torch.save(x.to_sparse(), 'data/processed/citeulike-a/content-bow.pt')
