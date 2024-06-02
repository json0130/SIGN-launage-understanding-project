# Sign Language Understanding Project

The Sign Language Understanding Project is a cutting-edge system that leverages machine learning techniques to analyze and interpret sign language in real-time. By utilizing three state-of-the-art Convolutional Neural Network (CNN) models—Inception v3, ResNet 50, and SUNSHINE23—, the project aims to provide accurate and efficient sign language recognition.

## Features

### PyQt Interface
- **User-Friendly Design**: The project boasts an intuitive and user-friendly interface built with PyQt, empowering users with a wide range of features. The interface provides a seamless interaction experience with the sign language understanding system.

- **Model Selection**: Users can easily select the desired model for sign language recognition, choosing from Inception v3, ResNet 50, or SUNSHINE23. The interface allows users to configure hyperparameters such as epoch and batch size, enabling customization based on their requirements.

- **Estimated Training Time**: The interface provides an estimated training time based on the user's model selection and hyperparameter choices. This feature helps users make informed decisions and plan their training sessions accordingly.

- **Training Control**: Users can start or stop the training process anytime through the interface. This allows convenient control over the training pipeline and enables users to adapt to their schedules and resources.

### Backend
- **Python-Based Architecture**: The project's backend is developed entirely in Python, ensuring a robust and efficient foundation for the sign language understanding system. The backend seamlessly integrates with the user-friendly interface, enabling smooth control of all functionalities.

- **Command-Line Free**: The backend architecture eliminates the need for users to run any code through the command line. All interactions and configurations can be performed directly through the intuitive interface, streamlining the user experience and making the system accessible to users with varying technical expertise.

- **Real-Time Sign Language Understanding**: The backend supports real-time sign language recognition using the user's webcam. It processes the video feed in real-time, applying the selected model to interpret and understand the sign language gestures performed by the user.

- **Dataset Management**: The backend handles the management of the sign language dataset. It allows users to utilize the provided dataset by uploading images of signs or expand the dataset by adding their own sign language images. The backend ensures proper organization and labelling of the dataset for effective training and recognition.

### Models
- **Inception v3**: The Inception v3 model is one of the three powerful CNN models integrated into the Sign Language Understanding Project. It offers the highest accuracy among the available models but requires a longer training time. For optimal performance, a GPU machine is recommended when using Inception v3.

- **ResNet 50**: ResNet 50 is another CNN model included in the project. It provides the fastest training time and is lightweight, making it suitable for CPU machines. ResNet 50 balances accuracy and efficiency, allowing users to train the model quickly without requiring specialized hardware.

- **SUNSHINE23**: SUNSHINE23 is a custom model developed specifically for the Sign Language Understanding Project. It implements feature concatenation, combining the MobileNet_v2 architecture with a custom CNN. By leveraging feature concatenation and additional connecting layers, SUNSHINE23 achieves higher accuracy in sign language recognition than traditional approaches.

These models are seamlessly integrated into the project, allowing users to choose the most suitable option based on their requirements, available resources, and desired balance between accuracy and training time.

## Installation

To set up and run the Sign Language Understanding Project, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/COMPSYS302/project-python-sunshine-group-23.git
   ```

2. Navigate to the project directory:
   ```
   cd project-python-sunshine-group-23
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the PYQT using the following command:
   ```
   python3 CNN_project.py
   ```

5. Once you launch the application, you can choose the preferred training model
6. Select the preferred setting and start the training.
7. Now you can test your model using the dataset image or using the webcam.

## Usage

1. Launch the Sign Language Understanding application.

2. Ensure that your webcam is connected and functioning properly.

3. Position yourself in front of the webcam, ensuring your hands are visible.

4. Perform sign language gestures; the application will analyze and interpret them in real-time.

5. The interpreted sign language will be displayed on the application's interface.

## Contributing

We welcome contributions to enhance the Sign Language Understanding Project. To contribute, please follow these guidelines:

1. Fork the repository and create your branch:
   ```
   git checkout -b feature/your-feature
   ```

2. Make your changes and test thoroughly.

3. Commit your changes with descriptive commit messages:
   ```
   git commit -m "Add feature: your feature description"
   ```

4. Push to your forked repository:
   ```
   git push origin feature/your-feature
   ```

5. Open a pull request, detailing the changes you made and their benefits.

We appreciate your contributions and look forward to collaborating with you to make the Sign Language Understanding Project even better.
