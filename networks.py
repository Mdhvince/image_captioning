from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models

from load_data import create_dataset


class EncoderCNN(nn.Module):
    """
    Use a pre-trained CNN to extract the features.
    """
    def __init__(self, embed_size, device):
        """
        embed_size: hyperparameter representing the size of the output feature. This will also be the input size of
        the embedding layer of Decoder.
        """
        super(EncoderCNN, self).__init__()

        resnet = models.resnet50(pretrained=True)

        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        # additional Fully connected layer with an output = the embed size to be fed to the rnn
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

        self.resnet.to(device)

    def forward(self, images):
        features = self.resnet(images)

        # flatten for our additional fc layer
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, device):
        """
            - embed_size: size of the embedding layer, should match the size of the output linear layer of the CNN
            (this is the input_size of the RNN)
            - hidden_size: max number of outputs between layers
            - vocab_size: size of the vocabulary
            - num_layers: number of stacked LSTM cells (2-3 works best in practice)
        """
        super().__init__()

        self.hidden = None
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        # embedding layer and the LSTM cell
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)

        # batch_first: if True, then the input and output tensors are provided as (batch, seq, feature)
        self.lstm = nn.LSTM(embed_size, self.hidden_size, self.num_layers, batch_first=True)

        # The linear layer maps the hidden state output of the LSTM to the number of words we want: vocab_size
        self.linear = nn.Linear(self.hidden_size, vocab_size)

    def init_hidden(self, batch_size):
        """ 
        Here we need to define h0, c0 (stm and ltm) with all zeroes in order to initialize our LSTM cell
        """
        h0 = torch.zeros((self.num_layers, batch_size, self.hidden_size))
        c0 = torch.zeros((self.num_layers, batch_size, self.hidden_size))
        return h0.to(self.device), c0.to(self.device)

    def forward(self, features, captions):
        features = features.to(self.device)
        captions = captions.to(self.device)

        batch_size = features.shape[0]

        # Initialize the hidden state
        self.hidden = self.init_hidden(batch_size)

        # Create embedded word vectors for each word in the captions
        embeddings = self.word_embeddings(captions)

        # Stack the features and captions
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)

        lstm_out, self.hidden = self.lstm(embeddings, self.hidden)

        out = self.linear(lstm_out)

        out = out[:, :-1]

        return out

    def sample(self, inputs, word2idx, max_pred_length=50):
        """
        Takes an image tensor (inputs) and returns predicted sentence (list of tensor ids of max_length)
        """

        output = []

        batch_size = inputs.shape[0]

        # Initialize from LSTM
        hidden = self.init_hidden(batch_size)

        while True:

            lstm_out, hidden = self.lstm(inputs, hidden)

            out = self.linear(lstm_out)
            out = out.squeeze(1)

            # predict next word
            _, ids = torch.max(out, dim=1)

            # get the predicted word
            output.append(ids.cpu().numpy()[0].item())  # storing the word predicted

            # stop when we predict the <end> word
            if ids == word2idx["<end>"] or len(output) == max_pred_length:
                break

            # new input using the predicted word
            inputs = self.word_embeddings(ids)
            inputs = inputs.unsqueeze(1)

        return output


if __name__ == "__main__":
    root = Path("/home/medhyvinceslas/Documents/programming")
    dataset = root / "datasets/image_captioning_flickr30k_images"
    annotations = dataset / "annotations_cleaned.csv"
    image_folder = dataset / "flickr30k_images"
    word2idx_file = root / "Image_Captioning/word2idx-toy.pkl"

    max_seq_length = 20

    train_set, word2idx = create_dataset(annotations, image_folder, word2idx_file, max_seq_length)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    embed_size = 256
    hidden_size = 512
    vocab_size = len(word2idx)

    encoder = EncoderCNN(embed_size, device)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers=2, device=device)
