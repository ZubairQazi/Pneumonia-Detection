import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets import ImageFolder

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils.pneumonia_dataset import load_dataset


# Load the pre-trained ResNet-18 model
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)

model_path = input('Enter model path: ')
model.load_state_dict(torch.load(model_path))
model.eval()

print('Loading Test Dataset & Dataloader...')
test_dataset = load_dataset('data/test')
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Lists to store true labels and predicted labels
true_labels = []
predicted_labels = []

# Iterate through the test dataset and make predictions
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        predicted = torch.argmax(outputs, dim=1)
        true_labels.extend(labels.tolist())
        predicted_labels.extend(predicted.tolist())

# Calculate accuracy, precision, recall, and F1-score
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='macro')
recall = recall_score(true_labels, predicted_labels, average='macro')
f1 = f1_score(true_labels, predicted_labels, average='macro')

# Print the metrics
print(f'\nAccuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')