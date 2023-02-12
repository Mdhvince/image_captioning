import matplotlib.pyplot as plt

from custom_data import *
from networks import EncoderCNN, DecoderRNN

if __name__ == "__main__":
    root = Path("/home/medhyvinceslas/Documents/programming")
    word2idx_file = root / "Image_Captioning/word2idx.pkl"
    encoder_path = root / "Image_Captioning/weights/encoder.pt"
    decoder_path = root / "Image_Captioning/weights/decoder.pt"

    with open(word2idx_file, 'rb') as f:
        word2idx = pickle.load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}\n")

    vocab_size = len(word2idx)
    embed_size = 256
    hidden_size = 512
    num_layers = 2

    encoder = EncoderCNN(embed_size, device)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers=num_layers, device=device)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))

    encoder.eval()
    decoder.eval()

    # pre-process
    im_path = root / "datasets/image_captioning_flickr30k_images/flickr30k_images/301246.jpg"
    im_path = root / "datasets/dog_cat_classification/test/data/4.jpg"

    image = mpimg.imread(im_path)

    transform = T.Compose([
        Rescale(224), Normalize(), ToTensor()
    ])
    sample = {"image": image, "caption": [0]}

    sample = transform(sample)
    img_tensor = sample["image"]
    img_tensor = img_tensor.type(torch.FloatTensor)
    img_tensor = img_tensor.to(device)
    img_tensor = img_tensor.unsqueeze(0)

    # forward
    features = encoder(img_tensor).unsqueeze(1)
    output = decoder.sample(features, word2idx)

    # display
    idx2word = {v: k for k, v in word2idx.items()}

    assert all([isinstance(x, int) for x in output]), "items in output tensor should be integer"
    assert all(
        [x in idx2word for x in output]), "items in the output needs to correspond to an integer in the vocabulary."

    caption_string = [idx2word[i] for i in output]
    caption_string = " ".join(caption_string)

    plt.title(caption_string)
    plt.axis("off")
    plt.imshow(image)
    plt.show()
