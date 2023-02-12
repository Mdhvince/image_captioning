import pickle
from pathlib import Path

import nltk
import pandas as pd
from nltk.tokenize import word_tokenize

nltk.download('punkt')

if __name__ == "__main__":
    root = Path("/home/medhyvinceslas/Documents/programming")
    dataset = root / "datasets/image_captioning_flickr30k_images"
    annotations = dataset / "annotations_cleaned.csv"
    word2idx_file = root / "Image_Captioning/word2idx-toy.pkl"

    df = pd.read_csv(annotations)

    tokens_list = ["<start>", "<end>", "<PAD>"]
    for cap in df["comment"].values.tolist():
        for s in eval(cap):
            s = s.lower()
            tokens = word_tokenize(s)
            tokens_list.extend(tokens)

    token_set = set(tokens_list)
    word2idx = {}
    for i, word in enumerate(token_set):
        word2idx[word] = i

    with open(word2idx_file, "wb") as handle:
        pickle.dump(word2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
