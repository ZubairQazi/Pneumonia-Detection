import torch

import torchvision.transforms as transforms
from PIL import Image

class ResizeAndPad:
    def __init__(self, desired_width, desired_height):
        self.desired_width = desired_width
        self.desired_height = desired_height

    def __call__(self, img):
        old_width, old_height = img.size

        # Calculate the new dimensions while maintaining the aspect ratio
        ratio = min(self.desired_width / old_width, self.desired_height / old_height)
        new_width = int(old_width * ratio)
        new_height = int(old_height * ratio)

        # Resize the image
        img = img.resize((new_width, new_height), Image.LANCZOS)

        # Create a new image with the desired width and height and paste the resized image in the center
        new_img = Image.new("RGB", (self.desired_width, self.desired_height), (0, 0, 0))
        left = (self.desired_width - new_width) // 2
        top = (self.desired_height - new_height) // 2
        new_img.paste(img, (left, top))

        return new_img
    
def calculate_mean_std(data_loader):
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0

    for images, _ in data_loader:
        batch_size, channels, height, width = images.size()
        total_images += batch_size

        # Calculate mean and std for each channel
        mean += images.mean((0, 2, 3))
        std += images.std((0, 2, 3))

    mean /= total_images
    std /= total_images

    return mean, std
