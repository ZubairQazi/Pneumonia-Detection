import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import os
from PIL import Image

import numpy as np
import pandas as pd


class PneumoniaImageDataset(Dataset):
    def __init__(self, images, labels, transforms=None):
        self.images = images
        self.labels = labels
        self.transforms = transforms
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx])

        if self.transforms is not None:
            image = self.transforms(image)
        
        label = self.labels[idx]

        return image, label