import torch


def read_mult_dat(filename, map_location='cpu'):
    document_words = []

    max_id = 0

    with open(filename) as f:
        for line in f:
            bow, largest_word_id = _parse_mult_dat_line(line)
            max_id = max(max_id, largest_word_id)

            document_words.append(bow)

    X = torch.zeros((len(document_words), max_id + 1), device=map_location)
    for i, bow in enumerate(document_words):
        for word_id, count in bow.items():
            X[i, word_id] = count

    maxes, _ = X.max(dim=1)
    X = X / maxes.unsqueeze(1)

    return X


def _parse_mult_dat_line(line):
    bow = {}

    max_word_id = 0

    split = line.split()
    for token in split[1:]:
        word_id, count = map(int, token.split(':'))
        bow[word_id] = count
        max_word_id = max(max_word_id, word_id)

    return bow, max_word_id


def read_ratings(filename, num_items, map_location='cpu'):
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

