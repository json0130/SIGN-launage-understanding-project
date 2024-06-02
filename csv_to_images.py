import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random

def csv_to_images(csvfile, output_dir, num_images, image_shape=(28, 28)):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csvfile)

    if num_images == 0: 
        num_images = df.shape[0]
    else:
        num_images = min(num_images, df.shape[0])

    # Separate labels from pixel data
    labels = df['label']
    data = df.drop(columns=['label'])

    # Normalize data if necessary (optional)
    data = data.values
    data = (data - data.min()) / (data.max() - data.min())  # Normalize to range 0-1

    count = 0
    # Convert random rows into images and save them
    while count < num_images:
        # Choose a random index
        random_index = random.randint(0, df.shape[0] - 1)
        # Get the row at the random index
        row = data[random_index]
        # Reshape row into the specified image shape
        image = row.reshape(image_shape)

        # Create the plot
        plt.figure(figsize=(5, 5))
        plt.imshow(image, cmap='gray')
        plt.axis('off')  # Hide axes

        # Save the image
        label = labels[random_index]
        image_path = os.path.join(output_dir, f'image_{random_index}_label_{label}.png')
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        count += 1

    print(f"Images saved to {output_dir}")

# Example usage:
# csv_to_images("sign_mnist_alpha_digits_test_train.csv", 'images', 30)
