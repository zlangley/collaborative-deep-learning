# Collaborative Deep Learning for Recommender Systems with BERT

This repository contains an implementation of the [Collaborative Deep Learning
for Recommender Systems](http://wanghao.in/paper/KDD15_CDL.pdf) by Wang, Wang,
and Yeung.  In contrast to the experiment presented in the paper, here we
explore using BERT to embed the content rather than the bag-of-words
representation.

The default hyperparameters---the best ones we found---achieve 25.3% recall@300
for the bag-of-words embedding. The same exact hyperparameters achieve 31.5%
recall@300 by simply switching to use the BERT embedding.

## Installation

The project was developed using Python 3.8. After setting up your Python 3.8
environment, you can install the requirements with `pip`:
```
pip install -r requirements.txt
```

## Data Processing

The repository contains only raw data. To prepare the data in the format the
source code expects, you can run
```
make features
```
which will read from `data/raw/` and write to `data/processed/`.

## Training and Inference

The file `train.py` will train the CDL from the data files created in the
previous step.  To train the model and compute recall@300, you can run
```
python train.py -v
```
The `-v` flag toggles "verbose" mode. There are many more command-line flags
to customize behavior; almost every hyperparameter can be set without changing
any code. By default, the BERT embedding will be used; to use the bag-of-words
embedding, run `train.py` with the flag `--embedding bow`.

The training script will output the recall@300 when finished. It will also
save the model to disk (by default in a file `model.pt`, but configurable with
the `--out` flag). The `infer.py` program can be run to (re)compute the recall.
