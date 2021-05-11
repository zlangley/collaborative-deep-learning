import torch
import pandas as pd
from sentence_transformers import SentenceTransformer


if __name__ == '__main__':
    model = SentenceTransformer('allenai-specter')
    df = pd.read_csv('data/raw/citeulike-a/raw-data.csv')

    embeddings = model.encode(df['title'] + ' ' + df['raw.abstract'], convert_to_tensor=True)
    torch.save(embeddings, 'data/processed/citeulike-a/content-bert.pt')
