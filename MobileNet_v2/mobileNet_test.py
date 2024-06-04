import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from train import ASLModel, data_transform, ASLDataset
from collections import deque

# Load the trained model
model = ASLModel(num_classes=36)  # Assuming 26 letters + 10 digits
model.load_state_dict(torch.load('asl_mobilenet_model.pth'))
model.eval()

# Create an instance of the ASLDataset class
dataset = ASLDataset(data=None)  # Pass None as data since we don't need it for testing

# Create a deque to store the latest 20 frame results
frame_results = deque(maxlen=20)

# Real-time testing using webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam
while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Preprocess the frame for display
    display_frame = frame.copy()
    
    # Convert frame to RGB and resize for model prediction
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224))
    frame = data_transform(frame).unsqueeze(0)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(frame)
        _, predicted = torch.max(outputs.data, 1)
        predicted_label = list(dataset.label_map.values())[predicted.item()]  # Get the corresponding alphabet or number
    
    # Append the predicted label to the frame results deque
    frame_results.append(predicted_label)
    
    # Get the most frequent label from the latest 20 frame results
    most_frequent_label = max(set(frame_results), key=frame_results.count)
    
    # Display the predicted label for the current frame on the command line
    print(f"Current Frame: {predicted_label}")
    
    # Display the most frequent label (20-frame moving average) on the command line
    print(f"Moving Average (20 frames): {most_frequent_label}")
    print("--------------------------------------------------")
    
    # Display the most frequent label on the display frame
    cv2.putText(display_frame, most_frequent_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Webcam', display_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()