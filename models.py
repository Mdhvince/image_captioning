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
    
    
    
    
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        
        self.hidden_dim = hidden_size
        
        # Our embedding layer
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(embed_size, self.hidden_dim, num_layers, batch_first=True)
        
        # The linear layer maps the hidden state output of the LSTM to the number of words we want:
        # vocab_size
        self.linear = nn.Linear(self.hidden_dim, vocab_size)
        
    
    def init_hidden(self, batch_size):
        """ 
        Here we need to define h0, c0 with all zeroes in order to initialize our LSTM
        Architecture
        """
        return torch.zeros((1, batch_size, self.hidden_dim)), torch.zeros((1, batch_size, self.hidden_dim))
    
    
    def forward(self, features, captions):
        
        # We don't want to take the <end> caption to make predictions of the following
        # word.
        captions = captions[:, :-1]
        
        # Make sure that features shape are :batch_size, embed_size
        batch_size = features.shape[0]
        
        # Initialize the hidden state
        self.hidden = self.init_hidden(batch_size)
        
        # Create embedded word vectors for each word in the captions
        embeddings = self.word_embeddings(captions) # embeddings new shape : (batch_size, captions length - 1, embed_size)
        
        # Stack the features and captions
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1) 
        
        lstm_out, self.hidden = self.lstm(embeddings, self.hidden) 
        
        out = self.linear(lstm_out)
        
        return out