import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random

def csv_to_images(csvfile, output_dir, image_shape=(28, 28), num_images_per_label=8):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csvfile)

    # Find all unique labels
    unique_labels = df['label'].unique()

    total_images_saved = 0

    # Iterate over each unique label
    for label in unique_labels:
        # Select rows with the current label
        label_rows = df[df['label'] == label].head(num_images_per_label)

        # Separate labels from pixel data
        labels = label_rows['label']
        data = label_rows.drop(columns=['label'])

        # Normalize data if necessary (optional)
        data = data.values
        data = (data - data.min()) / (data.max() - data.min())  # Normalize to range 0-1

        # Convert rows into images and save them
        for index, row in enumerate(data):
            # Reshape row into the specified image shape
            image = row.reshape(image_shape)

            # Create the plot
            plt.figure(figsize=(5, 5))
            plt.imshow(image, cmap='gray')
            plt.axis('off')  # Hide axes

            # Save the image
            image_path = os.path.join(output_dir, f'label_{label}_index_{index}.png')
            plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
            plt.close()

        total_images_saved += len(data)

    print(f"Images saved to {output_dir}")
    return total_images_saved
