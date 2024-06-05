import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from train import ASLModel, data_transform

def run():
    # Load the trained model
    num_classes = 34  # Number of ASL classes
    model = ASLModel(num_classes)
    
    # Load the model on CPU
    model.load_state_dict(torch.load('asl_resnet_model.pth', map_location=torch.device('cpu')))
    model.eval()

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
            predicted_label = chr(ord('A') + predicted.item())  # Convert class index to letter
        
        # Display the predicted label on the display frame
        cv2.putText(display_frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Webcam', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    run()

if __name__ == '__main__':
    main()
