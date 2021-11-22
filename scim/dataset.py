import os
import torch
import skimage.io 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


########################################################
## Re-arrange channels from tape format to stack tensor
########################################################

def fold_channels(image):
    # Expected input image shape: (h, w * c), with h = w
    # Output image shape: (h, w, c), with h = w
    output = np.reshape(image, (image.shape[0], image.shape[0], -1), order="F").astype(np.float)
    return output / 255.


########################################################
## Dataset Class
########################################################

class SingleCellDataset(Dataset):
    """Single cell dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with metadata.
            root_dir (string): Directory with all the images.
            last_channel (int): Index of last channel to fold the image
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.metadata = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.metadata.loc[idx, "Image_Name"])
        image = skimage.io.imread(img_name)
        image = fold_channels(image)

        label = self.metadata.loc[idx, "Target"]
        #label = np.array([label])
        #label = label.astype('float').reshape(-1, 2)
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


########################################################
## Transform Class
########################################################

class MaskChannel(object):

    def __init__(self, mode="ignore"):
        assert mode in ["ignore", "drop", "apply"]
        self.mode = mode

    def __call__(self, sample):
        if self.mode == "ignore":
            # Keep all channels
            return sample
        elif self.mode == "drop":
            # Drop mask channel (last)
            img = sample["image"][:, :, 0:-1]
        elif self.mode == "apply":
            # Use last channel as a binary mask
            mask = sample["image"][:, :, -1:]
            img = sample["image"][:, :, 0:-1] * mask

        return {'image': img, 'label': sample["label"]}


