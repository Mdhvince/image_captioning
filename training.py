import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim

from models import EncoderCNN, DecoderRNN
from load_data import *


def train(n_epochs, train_loader, valid_loader, save_location_path):

    encoder = EncoderCNN(embed_size)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

    # Move to GPU, if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    params = list(decoder.parameters()) + list(encoder.embed.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001)

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
                torch.save(encoder.state_dict(), save_location_path+'/encoder{n_epochs}.pt')
                torch.save(decoder.state_dict(), save_location_path+'/decoder{n_epochs}.pt')
                valid_loss_min = valid_loss


if __name__ == '__main__':

    csv_file = 'data/results2.csv'
    root_dir = 'data/flickr30k_images'
    mapper_file = 'mapping.pkl'
    save_location_path = 'saved_models'
    batch_size = 10
    num_workers = 4
    valid_size = 0.3
    embed_size = 256
    vocab_size = len(vocab)
    hidden_size = 512

    train_set = create_dataset(csv_file, root_dir, mapper_file)
    train_sampler, valid_sampler = train_valid_split(train_set, valid_size)
    train_loader, valid_loader = build_lodaers(train_set, train_sampler, valid_sampler,
                                               batch_size, valid_size, num_workers,
                                               csv_file, root_dir)

    train(n_epochs=2, train_loader, valid_loader, save_location_path)