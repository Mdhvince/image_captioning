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
from nltk.tokenize import word_tokenize, sent_tokenize
from itertools import chain



class ImageCaptionDataset(Dataset):
    """Image Caption dataset."""

    def __init__(self, csv_file, root_dir, mapper_file, max_seq_length=20, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.max_seq_length = max_seq_length
        
        print("Reading data...")
        self.df = pd.read_csv(csv_file)
        self.captions_column = self.df['captions']
        self.img_name_column = self.df['img_name']
        
        print("Calculating length...")
        self.df['length'] = self.captions_column.apply(lambda x: len(x.split()))
        self.length_column = self.df['length']
        
        self.root_dir = root_dir
        self.transform = transform
        
        print("Reading Mapper file...")
        with open('mapping.pkl', 'rb') as f:
            self.mapper_file = pickle.load(f)
        
        print("Ready !")
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # take the image name contained in the csv file
        image_name = os.path.join(self.root_dir, self.img_name_column[idx])
        
        #image_name = os.path.join(self.root_dir,
                                  #self.df.iloc[idx, 0])
        

        # read the true image based on that name
        # choice: mpimg because done with 1 line
        # with cv2, I need to read the convert from BGR2RGB
        image = mpimg.imread(image_name)
        
        # read df & transform caption to tensor
        caption = self.captions_column[idx]
        #caption = self.df.iloc[idx, 1]
        caption = caption.lower()
        tokens = word_tokenize(caption)
                
        caption = []
        caption.append('<start>')
        caption.extend([token for token in tokens])
        caption.append('<end>')
        
        # Map to integer
        caption = [self.mapper_file[i] for i in caption]
        
        #pad sequence
        caption = self.pad_data(caption)
        
        sample = {'image': image, 'caption': caption}
        
        
        if self.transform:
            sample = self.transform(sample)

        return sample
    
    
    
    def pad_data(self, s):
        padded = np.ones((self.max_seq_length,), dtype=np.int64)*self.mapper_file['<PAD>']
        
        if len(s) > self.max_seq_length:
            padded[:] = s[:self.max_seq_length]
        else: 
            padded[:len(s)] = s
            
        return padded













