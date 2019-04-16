import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, models
import torch.optim as optim
import pickle

from transform import *
from custom_data import ImageCaptionDataset
from models import EncoderCNN, DecoderRNN

batch_size = 10
num_workers = 4
csv_file = 'data/results2.csv'
root_dir = 'data/flickr30k_images'
mapper_file = 'mapping.pkl'

transform = transforms.Compose([
    Rescale(224),
    Normalize(),
    ToTensor()
])

with open(mapper_file, 'rb') as f:
    vocab = pickle.load(f)

    
valid_size = 0.3


def train_valid_split(training_set, validation_size):
    """ Function that split our dataset into train and validation
        given in parameter the training set and the % of sample for validation"""
    
    # obtain training indices that will be used for validation
    num_train = len(training_set)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(validation_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    return train_sampler, valid_sampler


train_set = ImageCaptionDataset(csv_file=csv_file,
                                root_dir=root_dir,
                                mapper_file=mapper_file,
                                transform=transform)

train_sampler, valid_sampler = train_valid_split(train_set, valid_size)


train_loader = DataLoader(train_set,
                          batch_size=batch_size,
                          sampler=train_sampler,
                          num_workers=num_workers)

valid_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=batch_size,
                                           sampler=valid_sampler,
                                           num_workers=num_workers)


train_on_gpu = torch.cuda.is_available()

embed_size = 256
vocab_size = len(vocab)
hidden_size = 512

encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)



# Move to GPU, if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = encoder.to(device)
decoder = decoder.to(device)

criterion = nn.CrossEntropyLoss().to(device)

# Learnable parameters & Optimizer
params = list(decoder.parameters()) + list(encoder.embed.parameters())
optimizer = torch.optim.Adam(params, lr=0.001)



n_epochs = 2

# This is to make sure that the 1st loss is  lower than sth and
# Save the model according to this comparison
valid_loss_min = np.Inf

for epoch in range(1, n_epochs+1):
    
    # Keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    
    encoder.train()
    decoder.train()
    for data in train_loader:
        images, captions = data['image'], data['caption']
        images = images.type(torch.FloatTensor)
        images.to(device)
        captions.to(device)
        
        decoder.zero_grad()
        encoder.zero_grad()

        features = encoder(images)
        outputs = decoder(features, captions)

        loss = criterion(outputs.contiguous().view(-1, vocab_size), captions.view(-1))
        loss.backward()  
        optimizer.step()
        
        train_loss += loss.item()*images.size(0)
    
    
    encoder.eval()
    decoder.eval()
    for data in valid_loader:
        images, captions = data['image'], data['caption']
        images = images.type(torch.FloatTensor)
        images.to(device)
        captions.to(device)
        
        features = encoder(images)
        outputs = decoder(features, captions)

        loss = criterion(outputs.contiguous().view(-1, vocab_size), captions.view(-1))
        
        valid_loss += loss.item()*images.size(0)
        
        # Average losses
        train_loss = train_loss/len(train_loader)
        valid_loss = valid_loss/len(valid_loader)
        
        print(f"Epoch: {epoch} \tTraining Loss: {train_loss} \tValidation Loss: {valid_loss}")

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print(f"Validation loss decreased ({valid_loss_min} --> {valid_loss}).  Saving model ...")
            torch.save(encoder.state_dict(), f'saved_models/encoder{n_epochs}.pt')
            torch.save(decoder.state_dict(), f'saved_models/decoder{n_epochs}.pt')
            valid_loss_min = valid_loss