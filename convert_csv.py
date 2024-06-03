import pandas as pd
import numpy as np
from PIL import Image
import os

# Path to your CSV file
csv_file = 'dataset.csv'  # Update this path to your CSV file

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file)

# Base directory to save images
output_base_dir = 'labeled_images'
os.makedirs(output_base_dir, exist_ok=True)

# Process each row in the DataFrame
for index, row in df.iterrows():
    label = row['label']
    pixels = row.drop('label').values.astype(np.uint8)  # Drop the label column and get pixel values
    image_size = int(np.sqrt(len(pixels)))  # Assuming the image is square

    # Ensure the pixels form a square image
    if image_size * image_size != len(pixels):
        print(f"Skipping row {index} due to inconsistent pixel length.")
        continue

    # Reshape pixels to a 2D array (image)
    image_array = pixels.reshape((image_size, image_size))

    # Create an image from the array
    image = Image.fromarray(image_array, mode='L')  # 'L' mode is for grayscale

    # Directory for the current label
    label_dir = os.path.join(output_base_dir, str(label))
    os.makedirs(label_dir, exist_ok=True)

    # Save the image in the label directory
    image_path = os.path.join(label_dir, f'image_{index}.jpeg')
    image.save(image_path)

    print(f"Saved image: {image_path}")
