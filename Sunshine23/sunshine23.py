import cv2
import numpy as np
import torch
import os
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import transforms
from torchvision.models import mobilenet_v2
from torchvision.models import MobileNet_V2_Weights
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define the custom CNN model
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))  # Align spatial dimensions

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.adaptive_pool(x)  # Apply adaptive pooling here
        return x

class FeatureConcatModel(nn.Module):
    def __init__(self, num_classes):
        super(FeatureConcatModel, self).__init__()
        self.mobilenet_features = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).features
        self.custom_cnn = CustomCNN()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1408 * 7 * 7, 512) # Updated to match the concatenated feature map size
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        mobilenet_features = self.mobilenet_features(x)
        custom_cnn_features = self.custom_cnn(x)

        # Concatenate the feature maps along the channel dimension
        concatenated_features = torch.cat((mobilenet_features, custom_cnn_features), dim=1)

        x = self.flatten(concatenated_features)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Prepare data generator for standardizing frames before sending them into the model
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create a custom dataset (assuming you have your data in a similar format)
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
        image = pixels.reshape(28, 28, 1)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if self.transform:
            image = self.transform(image)

        return image, label


def main():
    # Load the dataset (assuming you have it in a pandas DataFrame)
    data = pd.read_csv('../dataset.csv', low_memory=False)

    # Split the dataset into train and test
    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

    # Create datasets and data loaders
    train_dataset = ASLDataset(train_data, transform=data_transform)
    test_dataset = ASLDataset(test_data, transform=data_transform)
    batch_size = 80

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    num_classes = len(train_dataset.classes)
    model = FeatureConcatModel(num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 30
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
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)

        print(f"Accuracy on test set: {(correct / total) * 100}%")

    # Save the trained model
    torch.save(model.state_dict(), 'feature_concat_model.pth')


if __name__ == '__main__':
    main()
