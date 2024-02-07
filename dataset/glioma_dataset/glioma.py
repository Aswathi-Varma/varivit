import os

import numpy as np
import torch
from torch import sqrt
from torch.utils.data import Dataset
import torchio as tio
import matplotlib.pyplot
from ctviewer import CTViewer
import random
BASE_PATH = '/root/data/glioma/ViT_96' 
#BASE_PATH = '/root/data/glioma/ViT_Multi'


class GliomaData(Dataset):
    def __init__(self, mode=None, filename='x_train_ssl.npy', transform=None, label_name=None, use_z_score=False):
        super(FlairData).__init__()
        file_path = os.path.join(BASE_PATH, filename)
        data_raw = np.load(file_path)
        print("data raw shape", data_raw.shape)
        self.data = data_raw.transpose([0, 4, 1, 2, 3])
        self.transform = transform
        self.use_z_score = use_z_score
        self.labels = np.load(os.path.join(BASE_PATH, label_name)) if label_name is not None else None
        
        print(f"Using z-score normalization: {use_z_score}")

    def __len__(self):
        return self.data.shape[0]

    def _normalize_data(self, volume):
        if self.use_z_score:
            return (volume - volume.mean()) / sqrt(volume.var())
        max_val, min_val = volume.max(), volume.min()
        volume = (volume - min_val) / (max_val - min_val)
        return 2 * volume - 1

    def _min_max_normalize_data(self, volume):
        max_val, min_val = volume.max(), volume.min()
        volume = (volume - min_val) / (max_val - min_val)
        return volume

    def __getitem__(self, item):
       
        volume = torch.tensor(self.data[item], dtype=torch.float)
        if self.transform is not None:
            volume = self.transform(volume)
        original_volume = self._normalize_data(volume.clone())

        if self.labels is not None:
            original_labels = torch.tensor(self.labels[item])
        return original_volume, original_labels

    def __str__(self):
        return f"Pre-train Flair MRI data"

class GliomaDataMulti(Dataset):
    def __init__(self, mode=None, image_files=None, label_files=None , transform=None, use_z_score=False):
        super(GliomaData).__init__()
        
        images = [np.load(file).transpose([0, 4, 1, 2, 3]) for file in image_files]
        labels = [np.load(file) for file in label_files]
        self.data = [image for sublist in images for image in sublist]
        self.labels = [label for sublist in labels for label in sublist]
        self.transform = transform
        self.use_z_score = use_z_score

        print(f"Using z-score normalization: {use_z_score}")

    def __len__(self):
        return len(self.data)

    def _normalize_data(self, volume):
        if self.use_z_score:
            return (volume - volume.mean()) / sqrt(volume.var())
        max_val, min_val = volume.max(), volume.min()
        volume = (volume - min_val) / (max_val - min_val)
        return 2 * volume - 1

    def _min_max_normalize_data(self, volume):
        max_val, min_val = volume.max(), volume.min()
        volume = (volume - min_val) / (max_val - min_val)
        return volume

    def __getitem__(self, item):
       
        volume = torch.tensor(self.data[item], dtype=torch.float)
        if self.transform is not None:
            volume = self.transform(volume)
        original_volume = self._normalize_data(volume.clone())

        if self.labels is not None:
            original_labels = torch.tensor(self.labels[item])

        return original_volume, original_labels

    def __str__(self):
        return f"Pre-train MRI data"

def build_dataset(mode, args=None, transforms = None, use_z_score=False):
    assert mode in ['train', 'test', 'val', 'whole'], f"Invalid Mode selected, {mode}"
    filename = f'x_{mode}_ssl.npy'
    label_name = f'y_{mode}_ssl.npy'
    return GliomaData(mode=mode, filename=filename, transform=transforms, label_name=label_name, use_z_score=use_z_score)

def build_multi_dataset(mode, sizes, args=None, transforms = None, use_z_score=False):
    assert mode in ['train', 'test', 'val', 'whole'], f"Invalid Mode selected, {mode}"
    image_files = [f"{BASE_PATH}/x_{mode}_{size}.npy" for size in sizes]
    label_files = [f"{BASE_PATH}/y_{mode}_{size}.npy" for size in sizes]

    return GliomaDataMulti(mode=mode, image_files=image_files, label_files=label_files, transform=transforms, use_z_score=use_z_score)


if __name__ == '__main__':
    transforms = [
        tio.RandomAffine(),
        tio.RandomBlur(),
        tio.RandomNoise(std=0.1),
        tio.RandomGamma(log_gamma=(-0.3, 0.3))
    ]
    transformations = tio.Compose(transforms)

    data = build_dataset(mode='whole')
    data_loader = torch.utils.data.DataLoader(data, batch_size=4)
    min_val, max_val = float("inf"), 0

    for batch_data,_, label in data_loader:
        if batch_data.max() > max_val:
            max_val = batch_data.max()
        if batch_data.min() < min_val:
            min_val = batch_data.min()
    print(f"Max value is {max_val}, min value {min_val}")
    # Also a check for other data splits
    train_data = build_dataset(mode='train')
    data_loader = torch.utils.data.DataLoader(train_data, batch_size=16)
    min_val, max_val = float("inf"), 0
    all_ones, total = 0, 0
    for batch_data,_, labels in data_loader:
        all_ones += labels.sum()
        total += labels.shape[0]
        if batch_data.max() > max_val:
            max_val = batch_data.max()
        if batch_data.min() < min_val:
            min_val = batch_data.min()
    print(f"% of ones {all_ones / total}")
    print(f"Max value is {max_val}, min value {min_val}")
    #     break
    # Checking for the validation split
    test_data = build_dataset(mode='train')
    data_loader = torch.utils.data.DataLoader(test_data, batch_size=16)
    min_val, max_val = float("inf"), 0
    for batch_data, _, labels in data_loader:
        if batch_data.max() > max_val:
            max_val = batch_data.max()
        if batch_data.min() < min_val:
            min_val = batch_data.min()
    print(f"Max value is {max_val}, min value {min_val}")
