import cv2
import numpy as np
import csv
import os

def image_to_pixel_list(img):
    # Convert to grayscale if not already
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize to 28x28
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Flatten to 1D array and return as list
    return img.flatten().tolist()

def append_image_to_csv(input_file, label, output_file):
    # Read the JPEG image
    img = cv2.imread(input_file)
    
    # Check if the image was successfully loaded
    if img is None:
        print(f"Error: Unable to load image {input_file}")
        return
    
    # Convert image to pixel list
    pixel_list = image_to_pixel_list(img)
    
    # Open the CSV file in append mode
    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Create the row: label followed by pixel values
        row = [label] + pixel_list
        
        # Write the row
        writer.writerow(row)
    
    print(f"Image {input_file} (label: {label}) has been appended to {output_file}")

def process_image_directory(base_dir, output_file, recursive=True):
    # Create the output file with header if it doesn't exist
    if not os.path.exists(output_file):
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ['label'] + [f'pixel{i+1}' for i in range(784)]
            writer.writerow(header)
    
    # Function to process a directory
    def process_dir(directory):
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                if recursive:
                    process_dir(item_path)
                continue
            
            # Check if it's a JPEG file
            if item.lower().endswith(('.jpg', '.jpeg')):
                # Try to extract label from directory name
                dir_name = os.path.basename(directory)
                try:
                    label = int(dir_name)
                    if 0 <= label <= 35:  # Valid ASL label range
                        append_image_to_csv(item_path, label, output_file)
                except ValueError:
                    # Directory name is not a valid label, skip this image
                    print(f"Skipping {item_path} - parent directory '{dir_name}' is not a valid label")
    
    # Start processing from the base directory
    process_dir(base_dir)

# Example usage
base_dir = "asl_dataset_digits"  # Replace with your actual path
output_file = "dataset.csv"  # Replace with your actual path

process_image_directory(base_dir, output_file)