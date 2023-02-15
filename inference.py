import configparser
import warnings

import matplotlib.pyplot as plt

from custom_data import *
from networks import EncoderCNN, DecoderRNN

warnings.filterwarnings('ignore')


def cuda_setup():
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        torch.cuda.empty_cache()

    n_gpu = torch.cuda.device_count()
    device = torch.device("cuda" if is_cuda else "cpu")
    print(f"Running on {device}\n")
    return n_gpu, device

def load_models(cfg, cfg_net, vocab_size, device):
    num_layers = cfg_net.getint("num_layers")
    embed_size = cfg_net.getint("embed_size")
    hidden_size = cfg_net.getint("hidden_size")
    encoder_path = Path(cfg.get("encoder_path"))
    decoder_path = Path(cfg.get("decoder_path"))

    encoder = EncoderCNN(embed_size, device)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers=num_layers, device=device)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))

    encoder.eval()
    decoder.eval()

    return encoder, decoder

def get_vocabulary(cfg):
    word2idx_file = Path(cfg.get("word2idx_file"))

    with open(word2idx_file, 'rb') as f:
        word2idx = pickle.load(f)

    vocab_size = len(word2idx)
    return word2idx, vocab_size

def load_transform_image(im_path, device):
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
    return image, img_tensor

def process_caption(output, word2idx):
    idx2word = {v: k for k, v in word2idx.items()}

    assert all([isinstance(x, int) for x in output]), "items in output tensor should be integer"
    assert all(
        [x in idx2word for x in output]), "items in the output needs to correspond to an integer in the vocabulary."

    caption_string = [idx2word[i] for i in output if idx2word[i] not in ["<start>", "<end>"]]
    caption_string = " ".join(caption_string)
    return caption_string


if __name__ == "__main__":
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read("config.ini")
    cfg, cfg_net = config["DEFAULT"], config["NETWORK"]

    im_path = Path(cfg.get("root")) / "datasets/dog_cat_classification/test/data/4.jpg"

    n_gpu, device = cuda_setup()
    word2idx, vocab_size = get_vocabulary(cfg)
    encoder, decoder = load_models(cfg, cfg_net, vocab_size, device)
    image, img_tensor = load_transform_image(im_path, device)

    # forward
    with torch.no_grad():
        features = encoder(img_tensor).unsqueeze(1)
        output = decoder.sample(features, word2idx)

    caption_string = process_caption(output, word2idx)

    plt.title(caption_string)
    plt.axis("off")
    plt.imshow(image)
    plt.show()
