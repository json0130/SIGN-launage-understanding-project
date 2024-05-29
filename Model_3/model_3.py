import cv2
import numpy as np
import math
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import inception_v3
from torchvision.models import Inception_V3_Weights
from sklearn.model_selection import train_test_split


# Define the model architecture
class ASLModel(nn.Module):
    def __init__(self, num_classes, input_size=28, channels=3):
        super(ASLModel, self).__init__()
        self.channels = channels
        self.input_size = input_size
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=3, padding=1) 
        # self.inception = inception_v3(pretrained=True)
        self.inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.inception.aux_logits = False  # Disable auxiliary classifiers
        num_ftrs = self.inception.fc.in_features
        self.inception.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.inception(x)

# Prepare data generator for standardizing frames before sending them into the model
data_transform = transforms.Compose([
    transforms.ToTensor(),
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
        pixels = row[2:].values.astype(np.uint8)

        image_channels = len(pixels)  # Get the number of channels
        input_size = self.input_size


        image = cv2.resize(image, (299, 299))  # Resize for InceptionV3
        if self.transform:
            image = self.transform(image)
        else:
            image = np.reshape(pixels, (image_channels, input_size, input_size))
        return image, label

# Load the dataset
data = pd.read_csv('dataset.csv')

# Split the dataset into train and test
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Create datasets and data loaders
train_dataset = ASLDataset(train_data, transform=data_transform)
test_dataset = ASLDataset(test_data, transform=data_transform)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model
num_classes = len(train_dataset.classes)
model = ASLModel(num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

# Testing loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test set: {(correct / total) * 100}%")