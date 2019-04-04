import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        
        #import pre trained model
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        # remove last fully connected layer
        modules = list(resnet.children())[:-1]
        
        # build the new resnet
        self.resnet = nn.Sequential(*modules)
        
        # our additional Fully connected layer with an output = the embbed size
        # to feed the rnn
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        #call resnet on our images
        features = self.resnet(images)
        
        #flatten for our additional fc layer
        features = features.view(features.size(0), -1)
        
        features = self.embed(features)
        
        return features #here is our spacial information extracted from the image with the right output size