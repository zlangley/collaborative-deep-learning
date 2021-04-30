import torch


def recall(pred, test, M):
    _, indices = torch.sort(pred, dim=1, descending=True)

    top = indices[:, :M]

    gathered = test.gather(1, top)
    recall = gathered.sum(dim=1) / test.sum(dim=1)

    return recall.mean()
