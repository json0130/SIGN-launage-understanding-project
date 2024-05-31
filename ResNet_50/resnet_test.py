import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from resnet50 import ASLModel, data_transform

# Load the trained model
num_classes = 26  # Number of ASL classes
model = ASLModel(num_classes)
model.load_state_dict(torch.load('asl_resnet_model.pth'))
model.eval()

# Real-time testing using webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Preprocess the frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224))
    frame = data_transform(frame).unsqueeze(0)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(frame)
        _, predicted = torch.max(outputs.data, 1)
        predicted_label = chr(ord('A') + predicted.item())  # Convert class index to letter
    
    # Display the predicted label on the frame
    cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Webcam', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


