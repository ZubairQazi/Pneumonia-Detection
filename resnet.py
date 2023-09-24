import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision.models import resnet18, ResNet18_Weights

from sklearn.metrics import accuracy_score

from utils.pneumonia_dataset import load_dataset

train_dataset = load_dataset('data/train')
valid_dataset = load_dataset('data/val')

print('Loading Training Dataset...')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
print('Loading Validation Dataset...\n')
valid_loader = DataLoader(valid_dataset, batch_size=32)

# TRAINING
model = resnet18(weights=ResNet18_Weights.DEFAULT)

num_classes = 2
model.fc = nn.Linear(model.fc.in_features, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20  # Maximum number of epochs
quarter_epoch = len(train_loader) // 4
patience = 5     # Number of epochs to wait for convergence

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

prev_valid_loss = float('inf')
convergence_counter = 0

print(f'Entering training loop (device => {device})...\n')
try:
    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Print training loss for this epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
        
        # Validation
        model.eval()  # Set the model to evaluation mode
        valid_loss = 0.0
        all_preds = []
        all_labels = []

        correct = total = 0

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(valid_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                
                # Store predictions and labels for calculating accuracy
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        # Print validation loss for this epoch
        print(f"Validation Loss: {valid_loss/len(valid_loader)}")
        
        # Calculate validation accuracy at each quarter epoch
        if (i + 1) % quarter_epoch == 0:
            accuracy = accuracy_score(all_labels, all_preds)
            print(f"Validation Accuracy: {accuracy}")
        
        print()
        
        # Check for convergence
        if valid_loss >= prev_valid_loss:
            convergence_counter += 1
            if convergence_counter >= patience:
                print("Converged. Stopping training.")
                break
        else:
            convergence_counter = 0
        
        prev_valid_loss = valid_loss

    # Save the model after training
    torch.save(model.state_dict(), "pneumonia_model.pth")
    print("Model saved successfully.")

except Exception as e:
    print(f"An error occurred: {str(e)}")
    torch.save(model.state_dict(), "pneumonia_model_errored.pth")
    print("Model saved due to error.")


torch.cuda.empty_cache()