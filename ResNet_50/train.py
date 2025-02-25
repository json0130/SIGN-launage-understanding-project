import cv2
import numpy as np
import math
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define the model architecture
class ASLModel(nn.Module):
    def __init__(self, num_classes, input_size=224, channels=3):
        super(ASLModel, self).__init__()
        self.channels = channels
        self.input_size = input_size
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Prepare data generator for standardizing frames before sending them into the model
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),  # Resize for ResNet50
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create a custom dataset
class ASLDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.classes = sorted(data['label'].unique().tolist())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        label = self.classes.index(row['label'])
        pixels = row.iloc[1:].values.astype(np.uint8)
        image = pixels.reshape(28, 28, 1)  # Assuming grayscale images
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert to RGB

        if self.transform:
            image = self.transform(image)

        return image, label

def main():
    # Load the dataset
    data = pd.read_csv('../dataset.csv', low_memory=False)

    # Split the dataset into train and test
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Create datasets and data loaders
    train_dataset = ASLDataset(train_data, transform=data_transform)
    test_dataset = ASLDataset(test_data, transform=data_transform)
    batch_size = 100  # Reduced batch size for CPU training

    num_cores = os.cpu_count()
    num_workers = max(1, num_cores - 1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Initialize the model
    num_classes = len(train_dataset.classes)
    model = ASLModel(num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 30  # Reduced number of epochs for CPU training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    val_accuracies = []
    train_losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        total_batches = len(train_loader)

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            # Print progress 
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches: 
                progress = (batch_idx + 1) / total_batches * 100 
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{total_batches}] Loss: {running_loss / (batch_idx + 1):.4f} Progress: {progress:.2f}%") 
        
        train_losses.append(running_loss / len(train_loader))

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

        # Testing loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)

        print(f"Accuracy on test set: {(correct / total) * 100}%")

    # Plot validation accuracy graph 
    plt.figure(figsize=(8, 6)) 
    plt.plot(range(1, num_epochs + 1), val_accuracies, marker='o', linestyle='-') 
    plt.xlabel('Epoch') 
    plt.ylabel('Accuracy') 
    plt.title('Validation Accuracy') 
    plt.grid(True) 
    plt.show() 

    # Plot training loss graph 
    plt.figure(figsize=(8, 6)) 
    plt.plot(range(1, num_epochs + 1), train_losses, marker='o', linestyle='-') 
    plt.xlabel('Epoch') 
    plt.ylabel('Loss') 
    plt.title('Training Loss') 
    plt.grid(True) 
    plt.show()

    # save the trained model
    torch.save(model.state_dict(), 'asl_resnet_model.pth')


if __name__ == '__main__':
    main()
