import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from sunshine23 import FeatureConcatModel, data_transform

# Load the trained model
num_classes = 34
model = FeatureConcatModel(num_classes)
model.load_state_dict(torch.load('feature_concat_model.pth'))
model.eval()

# Create a dictionary to map class indices to letters or numbers
class_mapping = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
    26: '0', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9'
}

# Real-time testing using webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # flip the frame horizontally
    frame = cv2.flip(frame, 1)
    
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
        predicted_class = predicted.item()
        predicted_label = class_mapping[predicted_class]  # Get the label from the dictionary

    # Display the predicted label on the display frame
    cv2.putText(display_frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Webcam', display_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
