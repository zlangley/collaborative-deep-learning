import argparse

from cdl import data
from cdl.lfm import LatentFactorModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Collaborative Deep Learning training')
    parser.add_argument('--recall', type=int, default=300)
    parser.add_argument('--model', default='model.pt')
    args = parser.parse_args()

    lfm = LatentFactorModel((1, 1), 1)
    data.load_model(sdae=None, lfm=lfm, filename=args.model)

    ratings_test_dataset = data.load_cf_test_data()

    recall = lfm.compute_recall(ratings_test_dataset.to_dense(), args.recall)
    print(f'recall@{args.recall}: {recall.item()}')