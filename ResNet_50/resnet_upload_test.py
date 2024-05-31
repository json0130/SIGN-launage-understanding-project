import cv2
import torch
import pandas as pd
import torchvision.transforms as transforms
from resnet50 import ASLModel, data_transform

# Load the trained model
num_classes = 34  # Number of ASL classes
model = ASLModel(num_classes)
model.load_state_dict(torch.load('asl_resnet_model.pth'))
model.eval()

# Load CSV file containing pixel data
df = pd.read_csv('../dataset.csv')

# Define image dimensions (assuming images are square)
image_size = 28  # This needs to be set to the root of the number of pixel columns (e.g., 28x28 for 784 pixels)

# Loop through each row in the DataFrame
for index, row in df.iterrows():
    # Extract label and pixel data
    label = row['label']
    pixel_data = row[1:].values.astype('uint8')  # Excluding the label column
    image = pixel_data.reshape((image_size, image_size))  # Reshape flat pixel data to 2D
    
    # Convert grayscale to RGB by repeating the grayscale image across all three channels
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Resize the image to the required input size of the model
    image = cv2.resize(image, (224, 224))

    # Transform the image
    image_tensor = data_transform(image).unsqueeze(0)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs.data, 1)
        predicted_label = chr(ord('A') + predicted.item())  # Convert class index to letter
    
    print(f"True label: {label}, Predicted: {predicted_label}")

# Clean up any resources if necessary
