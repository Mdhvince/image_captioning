import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2


class Normalize(object):

    def __call__(self, sample):
        image, captions = sample['image'], sample['captions']
        
        image_copy = np.copy(image)
        image_copy =  image_copy/255.0
        
        return {'image': image_copy, 'captions': captions}



class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, captions = sample['image'], sample['captions']
        
        img = cv2.resize(image, (self.output_size, self.output_size))
          
        return {'image': img, 'captions': captions}




class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, captions = sample['image'], sample['captions']
         
        # if image has no RGB color channel, add one
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 3)
            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        return {'image': torch.from_numpy(image), 'captions': captions}
                #'captions': torch.from_numpy(captions)}
