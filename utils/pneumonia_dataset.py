import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import os
import glob
import pickle

from PIL import Image

import numpy as np
import pandas as pd

from . import ResizeAndPad, calculate_mean_std


class PneumoniaImageDataset(Dataset):
    def __init__(self, images, labels, transforms=None):
        self.images = images
        self.labels = labels
        self.transforms = transforms
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        label = self.labels[idx]

        if self.transforms is not None:
            image = self.transforms(image)
                
        return image, label
    

def load_dataset(data_path, file_extensions=['jpeg'], verbose=False):

    image_paths = []
    labels = []

    class_folders = [dir for dir in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, dir))]

    # Assign a unique label (integer) to each class folder
    class_to_label = {class_name: label for label, class_name in enumerate(class_folders)}

    for class_name in class_folders:
        class_folder_path = os.path.join(data_path, class_name)
        
        # Loop through file extensions to find image files in the class folder
        for ext in file_extensions:
            image_paths.extend(glob.glob(os.path.join(class_folder_path, f'*.{ext}')))
            labels.extend([class_to_label[class_name]] * len(glob.glob(os.path.join(class_folder_path, f'*.{ext}'))))

    if verbose:
        print("Number of images found:", len(image_paths))
        print("Number of labels found:", len(labels))

    data_transforms = transforms.Compose([
        ResizeAndPad(1200, 800),
        transforms.ToTensor(), 
    ])

    dataset = PneumoniaImageDataset(image_paths, labels, transforms=data_transforms)

    # Create a data loader to calculate mean and std
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # Calculate mean and std for normalization
    mean, std = calculate_mean_std(data_loader)

    with open('pneumonia_mean_std.pkl', 'wb') as f:
        pickle.dump((mean, std), f)

    # Update the transform to include normalization
    data_transforms = transforms.Compose([
        ResizeAndPad(600, 400),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Update the dataset with the final transform
    dataset = PneumoniaImageDataset(image_paths, labels, transforms=data_transforms)

    return dataset


if __name__ == '__main__':
    load_dataset(input('Enter dataset path: '), verbose=True)
