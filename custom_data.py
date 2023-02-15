import pickle
from pathlib import Path

import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import torchvision.transforms as T
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """
    Image Caption dataset
    """

    def __init__(self, csv_file, image_folder, word2idx_file, max_seq_length=20, transform=None):
        self.max_seq_length = max_seq_length

        self.df = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform

        with open(word2idx_file, 'rb') as f:
            self.word2idx = pickle.load(f)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # read an image and its list of captions
        image = mpimg.imread(self.image_folder / self.df.image_name[idx])
        captions = self.df.comment[idx]

        if not isinstance(captions, list):
            captions = eval(captions)
        assert isinstance(captions, list)

        # select one caption at random from the list and process it
        caption = captions[np.random.randint(0, len(captions))]
        caption_lw = caption.lower()
        tokens = word_tokenize(caption_lw)

        # add the start and end tokens and transform all token to integer
        caption_enriched = ['<start>']
        caption_enriched.extend([token for token in tokens])
        caption_enriched.append('<end>')
        caption_integer = [self.word2idx[i] for i in caption_enriched]

        # pad the sequence
        padded_caption = self.pad_data(caption_integer)

        sample = {'image': image, 'caption': padded_caption}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def pad_data(self, sentence_int):
        """
        sentence_int: encoded sentence into integer
        """
        padded = np.ones((self.max_seq_length,), dtype=np.int64) * self.word2idx["<PAD>"]

        if len(sentence_int) > self.max_seq_length:
            padded[:] = s[:self.max_seq_length]
        else:
            padded[:len(sentence_int)] = sentence_int

        return padded


class Normalize(object):

    def __call__(self, sample):
        image, caption = sample['image'], sample['caption']

        image_copy = np.copy(image)
        image_copy = image_copy / 255.0

        return {'image': image_copy, 'caption': caption}


class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, caption = sample['image'], sample['caption']

        img = cv2.resize(image, (self.output_size, self.output_size))

        return {'image': img, 'caption': caption}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, caption = sample['image'], sample['caption']

        # if image has no RGB color channel, add one
        if len(image.shape) == 2:
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 3)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        caption = torch.Tensor(caption).long()

        return {'image': torch.from_numpy(image), 'caption': caption}


if __name__ == "__main__":
    root = Path("/home/medhyvinceslas/Documents/programming")
    dataset = root / "datasets/image_captioning_flickr30k_images"
    annotations = dataset / "annotations_cleaned.csv"
    image_folder = dataset / "flickr30k_images"
    word2idx_file = root / "Image_Captioning/word2idx-toy.pkl"

    transform = T.Compose([
        Rescale(224),
        Normalize(),
        ToTensor()
    ])

    train_set = CustomDataset(annotations, image_folder, word2idx_file, max_seq_length=20, transform=transform)
