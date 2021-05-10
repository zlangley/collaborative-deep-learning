import torch



def recall(pred, test, k):
    _, indices = torch.topk(pred, k)
    gathered = test.gather(1, indices)
    recall = gathered.sum(dim=1) / test.sum(dim=1)
    return recall.mean()
