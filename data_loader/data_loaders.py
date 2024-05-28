from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import DataLoader, Dataset
import numpy as np
import glob
import os
from PIL import Image


class tumorDataset(Dataset):
    """
    Tumor dataset
    """

    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir: the directory of the dataset
            split: "train" or "test"
            transform: pytorch transformations.
        """
        self.transform = transform

        self.files = glob.glob(os.path.join(root_dir, split, '*.jpg'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx])
        img = np.asarray(img)
        if self.transform:
            img = self.transform(img)
        return img


class tumorDataLoader(BaseDataLoader):
    """
    Tumor data loader
    """

    def __init__(self, root_dir, batch_size, shuffle=True,
                 validation_split=0.0, num_workers=1, split='train'):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.root_dir = root_dir
        self.split = split
        self.dataset = tumorDataset(
            self.root_dir, split=self.split, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
