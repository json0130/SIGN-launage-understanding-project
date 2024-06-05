import os
import numpy as np
import pandas as pd
from PIL import Image

def image_to_pixel_array(image_path, size=(28, 28)):
    image = Image.open(image_path).convert('L')
    image = image.resize(size, Image.Resampling.LANCZOS)
    return np.array(image).flatten()

def images_to_csv(image_dir, output_csv):
    data = []
    for label in sorted(os.listdir(image_dir)):
        label_dir = os.path.join(image_dir, label)
        if os.path.isdir(label_dir):
            for image_file in os.listdir(label_dir):
                if image_file.endswith('.jpeg'):
                    image_path = os.path.join(label_dir, image_file)
                    pixel_array = image_to_pixel_array(image_path)
                    row = [int(label)] + pixel_array.tolist()
                    data.append(row)

    columns = ['label'] + [f'pixel{i+1}' for i in range(784)]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_csv, index=False)

image_dir = './labeled_images'
output_csv = 'output.csv'
images_to_csv(image_dir, output_csv)
