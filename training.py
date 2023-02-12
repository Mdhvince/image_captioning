import torch
import torch.nn as nn
from networks import EncoderCNN, DecoderRNN
from load_data import *


def train(encoder, decoder, train_loader, vocab_size, criterion, optimizer, device):
    running_loss = 0.0

    encoder.train()
    decoder.train()
    for data in train_loader:
        images, captions = data["image"], data["caption"]

        if torch.cuda.is_available():
            images = images.type(torch.cuda.FloatTensor)
        else:
            images = images.type(torch.FloatTensor)

        images.to(device)
        captions.to(device)

        decoder.zero_grad()
        encoder.zero_grad()

        features = encoder(images)
        outputs = decoder(features, captions)

        loss = criterion(outputs.contiguous().view(-1, vocab_size).to(device), captions.view(-1).to(device))
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss

def validate(encoder, decoder, valid_loader, vocab_size, criterion, device):
    running_loss = 0.0

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        for data in valid_loader:
            images, captions = data["image"], data["caption"]

            if torch.cuda.is_available():
                images = images.type(torch.cuda.FloatTensor)
            else:
                images = images.type(torch.FloatTensor)

            images.to(device)
            captions.to(device)

            features = encoder(images)
            outputs = decoder(features, captions)

            loss = criterion(outputs.contiguous().view(-1, vocab_size).to(device), captions.view(-1).to(device))

            running_loss += loss.item() * images.size(0)

    return running_loss

def learn(encoder, decoder, criterion, optimizer, loaders, encoder_path, decoder_path, vocab_size, device, n_epochs):

    valid_loss_min = np.Inf
    train_loader, valid_loader = loaders

    for epoch in range(1, n_epochs + 1):
        running_train_loss = train(encoder, decoder, train_loader, vocab_size, criterion, optimizer, device)
        running_valid_loss = validate(encoder, decoder, valid_loader, vocab_size, criterion, device)

        # Average losses
        train_loss = running_train_loss / len(train_loader)
        valid_loss = running_valid_loss / len(valid_loader)

        print(f"Epoch: {epoch} \tTraining Loss: {train_loss} \tValidation Loss: {valid_loss}")

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print(f"Validation loss decreased ({valid_loss_min} --> {valid_loss}).  Saving model ...")
            torch.save(encoder.state_dict(), encoder_path)
            torch.save(decoder.state_dict(), decoder_path)
            valid_loss_min = valid_loss


if __name__ == '__main__':
    root = Path("/home/medhyvinceslas/Documents/programming")
    dataset = root / "datasets/image_captioning_flickr30k_images"
    annotations = dataset / "annotations_cleaned.csv"
    image_folder = dataset / "flickr30k_images"
    word2idx_file = root / "Image_Captioning/word2idx.pkl"
    encoder_path = root / "Image_Captioning/weights/encoder-toy.pt"
    decoder_path = root / "Image_Captioning/weights/decoder-toy.pt"

    max_seq_length = 20
    batch_size = 2
    valid_size = 0.3
    num_workers = 2
    embed_size = 256
    hidden_size = 512
    num_layers = 2
    n_epochs = 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    train_set, word2idx = create_dataset(annotations, image_folder, word2idx_file, max_seq_length)
    loaders = build_loaders(train_set, batch_size, valid_size, num_workers)

    vocab_size = len(word2idx)
    encoder = EncoderCNN(embed_size, device)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers=num_layers, device=device)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    params = list(decoder.parameters()) + list(encoder.embed.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001)

    learn(encoder, decoder, criterion, optimizer, loaders, encoder_path, decoder_path, vocab_size, device, n_epochs)
