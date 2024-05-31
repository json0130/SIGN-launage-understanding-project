import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from train import ASLModel, data_transform
import pyrealsense2 as rs

# Load the trained model
num_classes = 26  # Number of ASL classes
model = ASLModel(num_classes)
model.load_state_dict(torch.load('asl_resnet_model.pth'))
model.eval()

# Initialize the Intel RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

try:
    while True:
        # Get a frame from the RealSense camera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert the frame from RealSense format to OpenCV format
        frame = np.asanyarray(color_frame.get_data())

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

        cv2.imshow('RealSense', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the RealSense pipeline and release resources
    pipeline.stop()
    cv2.destroyAllWindows()