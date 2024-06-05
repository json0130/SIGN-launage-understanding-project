import cv2
import torch
import torchvision.transforms as transforms
from train import ASLModel, data_transform

# Function to predict the label of an input image
def predict_image(image_path,model_file):

    # Load the trained model
    num_classes = 34  # Number of ASL classes
    model = ASLModel(num_classes)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    # Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not open or find the image: {image_path}")

    # Preprocess the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image_tensor = data_transform(image).unsqueeze(0)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_probabilities, predicted_classes = torch.topk(probabilities, k=5)
        predicted_probabilities = predicted_probabilities.squeeze().tolist()
        predicted_classes = predicted_classes.squeeze().tolist()
        labels = [chr(ord('A') + idx) for idx in predicted_classes]
        results = list(zip(labels, predicted_probabilities))
    
    return results

