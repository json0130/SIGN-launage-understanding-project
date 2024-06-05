import cv2
import torch
import torchvision.transforms as transforms
from train import ASLModel, data_transform
from model_scripts.train_mobilenetv2 import ASLModel as MobileNetV2Model
from model_scripts.train_inceptionv3 import ASLModel as InceptionV3Model
from model_scripts.train_resnet50 import ASLModel as ResNet50Model
from model_scripts.train_sunshine23 import FeatureConcatModel


# Function to predict the label of an input image
def predict_image(image_path,model_file):
    # Determine model type from file name
    if 'mobilenet' in model_file:
        model_type = 'MobileNetV2'
    elif 'inceptionv3' in model_file:
        model_type = 'InceptionV3'
    elif 'resnet' in model_file:
        model_type = 'ResNet50'
    elif 'feature_concat' in model_file:
        model_type = 'FeatureConcat'
    else:
        raise ValueError(f"Unknown model type in file name: {model_file}")

    # Load the appropriate model architecture based on model type
    num_classes = 36  # Number of ASL classes

    if model_type == 'MobileNetV2':
        model = MobileNetV2Model(num_classes)
    elif model_type == 'InceptionV3':
        model = InceptionV3Model(num_classes)
    elif model_type == 'ResNet50':
        model = ResNet50Model(num_classes)
    elif model_type == 'Sunshine23':
        model = FeatureConcatModel(num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load
    model.load_state_dict(torch.load(model_file))
    model.eval()

    # Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not open or find the image: {image_path}")

    # Preprocess the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))  # Ensure this matches the input size used during training
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = data_transform(image).unsqueeze(0)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_probabilities, predicted_classes = torch.topk(probabilities, k=5)
        predicted_probabilities = predicted_probabilities.squeeze().tolist()
        predicted_classes = predicted_classes.squeeze().tolist()
        
        # Define the labels for the 36 classes
        labels = [chr(i) for i in range(65, 91)] + [str(i) for i in range(10)]  # A-Z and 0-9
        predicted_labels = [labels[idx] for idx in predicted_classes]
        
        results = list(zip(predicted_labels, predicted_probabilities))
    
    return results

