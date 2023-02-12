import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from custom_data import *


def create_dataset(annotations, image_folder, word2idx_file, max_seq_length):
    with open(word2idx_file, 'rb') as f:
        word2idx = pickle.load(f)

    transform = transforms.Compose([
        Rescale(224),
        Normalize(),
        ToTensor()
    ])
    train_set = CustomDataset(
        annotations, image_folder, word2idx_file, max_seq_length=max_seq_length, transform=transform)
    return train_set, word2idx


def _train_valid_split(training_set, validation_size):
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


def build_loaders(train_set, batch_size, valid_size, num_workers):
    train_sampler, valid_sampler = _train_valid_split(train_set, valid_size)

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    valid_loader = DataLoader(train_set, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
    return train_loader, valid_loader


if __name__ == "__main__":
    root = Path("/home/medhyvinceslas/Documents/programming")
    dataset = root / "datasets/image_captioning_flickr30k_images"
    annotations = dataset / "annotations_cleaned.csv"
    image_folder = dataset / "flickr30k_images"
    word2idx_file = root / "Image_Captioning/word2idx-toy.pkl"

    max_seq_length = 20
    batch_size = 2
    valid_size = 0.3
    num_workers = 2

    train_set, word2idx = create_dataset(annotations, image_folder, word2idx_file, max_seq_length)
    train_loader, valid_loader = build_loaders(train_set, batch_size, valid_size, num_workers)
