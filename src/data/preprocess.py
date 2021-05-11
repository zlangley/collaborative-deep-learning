from collections import Counter

import nltk
import pandas as pd
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

out = 'data/processed/citeulike-a/content'
k = 8000

if __name__ == '__main__':
    nltk.download('punkt')
    nltk.download('stopwords')

    stops = set(stopwords.words('english'))

    df = pd.read_csv('./data/raw/citeulike-a/raw-data.csv')
    title_abstracts = df['raw.title'] + ' ' + df['raw.abstract']

    docs = [word_tokenize(txt) for txt in title_abstracts]
    docs = [[word.lower() for word in doc if word.isalpha() and word not in stops] for doc in docs]

    cnt = Counter([word for doc in docs for word in doc])
    common = [w for w, _ in cnt.most_common(k)]

    dictionary = {}
    for i, word in enumerate(common):
        dictionary[word] = i

    data = torch.zeros(len(docs), k)
    for i, doc in enumerate(docs):
        for word in doc:
            if word in dictionary:
                data[i][dictionary[word]] += 1

    torch.save(data, out)