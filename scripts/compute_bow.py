import torch


def compute_bow(infile, outfile, shape):
    indices = [[], []]
    values = []

    with open(infile) as f:
        for doc_id, line in enumerate(f):
            tokens = line.split()[1:]

            for token in tokens:
                word_id, cnt = tuple(map(int, token.split(':')))
                indices[0].append(doc_id)
                indices[1].append(word_id)
                values.append(cnt)

    x = torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32).to_dense()

    maxes, _ = x.max(dim=1, keepdim=True)
    torch.clamp_min_(maxes, 1)
    x /= maxes

    torch.save(x.to_sparse(), outfile)


if __name__ == '__main__':
    compute_bow('data/raw/citeulike-a/mult.dat', 'data/processed/citeulike-a/content-bow.pt', (16980, 8000))
    compute_bow('data/raw/citeulike-t/mult.dat', 'data/processed/citeulike-t/content-bow.pt', (25975, 20000))
