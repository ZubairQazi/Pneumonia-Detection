import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision.models import resnet18, ResNet18_Weights

from sklearn.metrics import accuracy_score

import sys

import mlflow
import mlflow.tracking

from utils import load_dataset


model_name = input('Enter model name (no extensions): ')

print('\nLoading Training Dataset & Dataloader...')
train_dataset = load_dataset('data/train')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

print('Loading Validation Dataset & Dataloader...\n')
valid_dataset = load_dataset('data/val')
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)

# TRAINING

# User-inputted model
if len(sys.argv) > 1:
    model = nn.Module()
    model.load_state_dict(torch.load(sys.argv[1]))

else:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    num_classes = 2
    model.fc = nn.Linear(model.fc.in_features, num_classes)

mlflow.log_params(model.named_parameters())

criterion = nn.CrossEntropyLoss()

learning_rate, weight_decay, num_epochs = 1e-5, 5e-4, 20
quarter_epoch = num_epochs // 4
patience = 5     # Number of epochs to wait for convergence

mlflow.log_param("learning rate", learning_rate)
mlflow.log_param("num epochs", num_epochs)
mlflow.log_param("weight decay", weight_decay)

params_1x = [param for name, param in model.named_parameters() if 'fc' not in str(name)]
optimizer = torch.optim.Adam([{'params':params_1x}, {'params': model.fc.parameters(), 'lr': learning_rate*10}], lr=learning_rate, weight_decay=weight_decay)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

prev_valid_loss = float('inf')
convergence_counter = 0

mlflow.start_run()

mlflow.log_param("model_name", model_name)

print(f'Entering training loop (device => {device}, epochs => {num_epochs})...\n')
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

        mlflow.log_metric("train_loss", running_loss/len(train_loader), epoch)
        
        # Validation
        model.eval()
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

        mlflow.log_metric("valid_loss", valid_loss/len(valid_loader), epoch)
        
        # Calculate validation accuracy at each quarter epoch
        if (i + 1) % quarter_epoch == 0:
            accuracy = accuracy_score(all_labels, all_preds)
            print(f"Validation Accuracy: {accuracy}")
            mlflow.log_metric("validation accuracy", accuracy)
        
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
    mlflow.pytorch.save_state_dict(model.state_dict(), f"models/{model_name}.pth")
    scripted_model = torch.jit.script(model)
    mlflow.pytorch.save_model(scripted_model, f"models/{model_name}_traced")
    print("Model saved successfully.")

    # Register the model in MLflow
    mlflow.pytorch.log_model(model, "model")
    mlflow.pytorch.log_model(scripted_model, "scripted_model")
    mlflow.end_run()

except Exception as e:
    print(f"An error occurred: {str(e)}")
    mlflow.pytorch.save_state_dict(model.state_dict(), f"{model_name}_errored.pth")
    mlflow.pytorch.save_model(torch.jit.script(model), f"{model_name}_traced_errored.pt")
    print("Model saved due to error.")

torch.cuda.empty_cache()