from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import DataLoader, Dataset
import numpy as np
import glob
import os
from PIL import Image
import h5py
import torch


class tumorDataset(Dataset):
    """
    Tumor dataset
    """

    def __init__(self, data_dir, split='train', transform=None):
        """
        Args:
            data_dir: the directory of the dataset
            split: "train" or "test"
            transform: pytorch transformations.
        """
        self.transform = transform

        self.files = glob.glob(os.path.join(data_dir, split, '*.jpg'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with h5py.File(self.files[idx], 'r') as file:
            img = file['image'][:]
            mask = file['mask'][:]

        img = img.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))

        img = torch.tensor(img, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        if self.transform:
            img = self.transform(img)

        return img, mask


class tumorDataLoader(BaseDataLoader):
    """
    Tumor data loader
    """

    def __init__(self, data_dir, batch_size, shuffle=True,
                 validation_split=0.0, num_workers=1, split='train'):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.split = split
        self.dataset = tumorDataset(
            self.data_dir, split=self.split, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
