import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2
from PIL import Image
import ast
import pickle



class ImageCaptionDataset(Dataset):
    """Image Caption dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.captions = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        
        # take the image name contained in the csv file
        image_name = os.path.join(self.root_dir,
                                  self.captions.iloc[idx, 0])

        # read the true image based on that name
        # choice: mpimg because done with 1 line
        # with cv2, I need to read the convert from BGR2RGB
        image = mpimg.imread(image_name)
        
        # read captions
        captions = self.captions.iloc[idx, 1]
        
        sample = {'image': image, 'captions': captions}
        
        if self.transform:
            sample = self.transform(sample)

        return sample