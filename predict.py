import torch
from torchvision import transforms
from PIL import Image

import sys
import pickle

from utils import ResizeAndPad

traced_model = torch.jit.load(sys.argv[1])
traced_model.eval()

with open('pneumonia_mean_std.pkl', 'rb') as f:
    mean, std = pickle.load(f)

transform = transforms.Compose([
    ResizeAndPad(600, 400),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Make a prediction on a new image
def predict(image_path):
    image = Image.open(image_path)
    prediction = traced_model(transform(image).unsqueeze(0))
    probabilities = torch.nn.functional.softmax(prediction, dim=1)
    # print(prediction)
    return torch.argmax(prediction, dim=1).item(), torch.max(probabilities).item()
    # return torch.argmax(prediction, dim=1)

# Get the image path from the command line arguments
image_path = sys.argv[2]

# Make a prediction on the image
output = predict(image_path)
prediction = 'Pneumonia' if output[0] == 0 else 'Normal'

# Print the prediction
print(f'Classification: {prediction} with a {output[1]:.2f}% probability')
