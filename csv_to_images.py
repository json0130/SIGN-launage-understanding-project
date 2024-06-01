import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def csv_to_images(csvfile, output_dir, num_images, image_shape=(28, 28)):
     # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csvfile)

    if num_images == 0: 
        num_images = df.shape[0]
    else:
        num_images = num_images

    # Separate labels from pixel data
    labels = df['label']
    data = df.drop(columns=['label'])

    # Normalize data if necessary (optional)
    data = data.values
    data = (data - data.min()) / (data.max() - data.min())  # Normalize to range 0-1

    count = 0
    # Convert each row into an image and save it
    for i, row in enumerate(data):
        if count == num_images:
            break
        # Reshape row into the specified image shape
        image = row.reshape(image_shape)

        # Create the plot
        plt.figure(figsize=(5, 5))
        plt.imshow(image, cmap='gray')
        plt.axis('off')  # Hide axes

        # Save the image
        label = labels[i]
        image_path = os.path.join(output_dir, f'image_{i}_label_{label}.png')
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        count += 1


    print(f"Images saved to {output_dir}")

#csv_to_images("sign_mnist_alpha_digits_test_train.csv", 'images', 30)
