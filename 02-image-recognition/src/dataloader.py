#!/usr/bin/env python3
from options import Options
import os
import pickle
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.io import read_image
from utils import show_image

def preprocess(image, opt):
    transformation = transforms.Resize((opt.isize, opt.isize))
    return transformation(image)

def serialize_all_in_dir(opt):
    # Check that necessary directories and files exist
    if not opt.serialization_target_dir.exists():
        os.mkdir(opt.serialization_target_dir)

    if not opt.serialization_source_dir.exists():
        raise FileNotFoundError("Serialization source directory not found at {}".format(opt.serialization_source_dir))
    if not opt.labels_file.exists():
        raise FileNotFoundError("Labels file not found at {}".format(opt.labels_file))
    if not opt.images_file.exists():
        raise FileNotFoundError("Images file not found at {}".format(opt.images_file))


    labels = []
    image_file_names = []
    with open(opt.datadir / opt.labels_file, 'r') as f:
        for line in f:
            labels.append(int(line[0])) # First character is the label

    with open(opt.datadir / opt.images_file, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.replace("000000000000", "") # Remove extra zeros from the middle
            image_file_names.append(line)

    #annotations = zip(labels, image_file_names)

    no_files = 0
    for n, label in enumerate(labels):
        file_name = "file{}.jpg".format(str(n + 1).zfill(4))
        image_path = opt.serialization_source_dir / file_name
        image = preprocess(read_image(str(image_path)), opt)
        file_name = file_name.replace(".jpg", ".pickle")
        with open(opt.serialization_target_dir / file_name, 'wb') as f:
            pickle.dump({opt.key_features: image, opt.key_classes: label}, f)
        no_files += 1
    print("Serialized {} files".format(no_files))

def get_dataloaders(opt) -> tuple[DataLoader, DataLoader]:
    dataset = GENKI4KDataset(opt)
    train_size = int(len(dataset) * opt.train_split)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print("Train size: {}, Test size: {}".format(len(train_dataset), len(test_dataset)))

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=opt.batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers,
                                  drop_last=True)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=opt.batch_size,
                                 shuffle=True,
                                 num_workers=opt.num_workers,
                                 drop_last=True)

    return train_dataloader, test_dataloader


class GENKI4KDataset(Dataset):
    """GENKI4K dataset."""

    def __init__(self, opt):
        self.opt = opt

        self.files = list(self.opt.serialization_target_dir.glob("*.pickle"))

        if self.opt.load_into_memory:
            for i, f in enumerate(self.files):
                self.files[i] = self._load_file(f)


    @staticmethod
    def _load_file(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.opt.load_into_memory:
            temp = self.files[idx]
        else:
            temp = self._load_file(self.files[idx])

        # Normalize
        image = (temp[self.opt.key_features] - 127.5) / 127.5
        return image, temp[self.opt.key_classes]



if __name__ == '__main__':
    opt = Options().parse()
    serialize_all_in_dir(opt)
    dataset = GENKI4KDataset(opt)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    for i, data in enumerate(dataloader):
        show_image(data[0][i])
        if i > 10:
            break
