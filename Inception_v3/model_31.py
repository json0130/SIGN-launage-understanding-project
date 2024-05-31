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
from torchvision.models import inception_v3
from torchvision.models import Inception_V3_Weights
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define the model architecture
class ASLModel(nn.Module):
    def __init__(self, num_classes, input_size=299, channels=3):
        super(ASLModel, self).__init__()
        self.channels = channels
        self.input_size = input_size
        self.inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.inception.aux_logits = False  # Disable auxiliary classifiers
        num_ftrs = self.inception.fc.in_features
        self.inception.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.inception(x)

def main():
    # Prepare data generator for standardizing frames before sending them into the model
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((299, 299)),  # Resize for InceptionV3
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

    # Load the dataset
    data = pd.read_csv('../dataset.csv', low_memory=False)

    # Split the dataset into train and test
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Create datasets and data loaders
    train_dataset = ASLDataset(train_data, transform=data_transform)
    test_dataset = ASLDataset(test_data, transform=data_transform)
    batch_size = 8  # Reduced batch size for CPU training

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
    num_epochs = 5  # Reduced number of epochs for CPU training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize lists to store validation accuracy and training loss for each batch
    val_accuracies = []
    train_losses = []

    # Create a figure and subplots for the live plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_batches = len(train_loader)
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Move data to device
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Print progress and update the live plot every 10 batches
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                progress = (batch_idx + 1) / total_batches * 100
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{total_batches}] Loss: {running_loss / (batch_idx + 1):.4f} Progress: {progress:.2f}%")
                
                # Update training loss plot
                train_losses.append(running_loss / (batch_idx + 1))
                ax1.clear()
                ax1.plot(train_losses)
                ax1.set_xlabel('Batch')
                ax1.set_ylabel('Loss')
                ax1.set_title('Training Loss')
                
                # Testing loop (run every 10 batches for validation accuracy)
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for images, labels in test_loader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                
                # Update validation accuracy plot
                val_accuracy = correct / total
                val_accuracies.append(val_accuracy)
                ax2.clear()
                ax2.plot(val_accuracies)
                ax2.set_xlabel('Batch')
                ax2.set_ylabel('Accuracy')
                ax2.set_title('Validation Accuracy')
                
                # Refresh the plot
                plt.tight_layout()
                plt.pause(0.1)
        
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")


    # Save the model
    model_path = "asl_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Show the final plot
    plt.show(block=False)


if __name__ == '__main__':
    main()
    