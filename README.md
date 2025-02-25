# Sign Language Understanding Project

The Sign Language Understanding Project is a cutting-edge system that leverages machine learning techniques to analyze and interpret sign language in real-time. By utilizing four state-of-the-art Convolutional Neural Network (CNN) models—Inception v3, ResNet 50, MobileNet v2 and SUNSHINE23—, the project aims to provide accurate and efficient sign language recognition.

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

- **ResNet 50**: ResNet 50 is another CNN model included in the project. It balances accuracy and training time, making it a good choice for users who want to perform well without extensive computational resources. ResNet 50 can be trained on GPU and CPU machines, although GPU usage is recommended for faster training.

- **MobileNet v2**: MobileNet v2 is the lightest and fastest model among the three included in the project. Its minor parameters make it highly efficient and suitable for deployment on resource-constrained devices such as mobile phones or embedded systems. Despite its lightweight nature, MobileNet v2 still achieves competitive accuracy in sign language recognition tasks. It is the ideal choice for users who prioritize speed and efficiency over slight gains in accuracy.
  
- **SUNSHINE23**: SUNSHINE23 is a custom model developed for the Sign Language Understanding Project. It implements feature concatenation, combining the MobileNet_v2 architecture with a custom CNN. By leveraging feature concatenation and additional connecting layers, SUNSHINE23 aims to enhance the accuracy of sign language recognition while maintaining the efficiency benefits of the MobileNet_v2 architecture. This model balances accuracy and speed, making it a versatile choice for various sign language recognition scenarios.

These models are seamlessly integrated into the project, allowing users to choose the most suitable option based on their requirements, available resources, and desired balance between accuracy, training time, and model size. Whether you prioritize high accuracy, fast training, or lightweight deployment, the Sign Language Understanding Project offers a model that can cater to your needs.

## Setup & Requirements

To set up the environment for the Sign Language Understanding Project, follow these steps:

1. **Python Installation**:
   - Ensure that you have Python installed on your system. 
   - You can download Python from the official website: [https://www.python.org/downloads/](https://www.python.org/downloads/)

2. **Anaconda Installation (Optional)**:
   - While not mandatory, we recommend using Anaconda to manage your Python environment and dependencies.
   - Anaconda provides a convenient way to create and maintain isolated environments for the project.
   - Download and install Anaconda from the official website: [https://www.anaconda.com/products/individual](https://www.anaconda.com/products/individual)

3. **Create a Virtual Environment (Optional)**:
   - If you are using Anaconda, create a new virtual environment for the project using the following command:
     ```
     conda create --name sign_language_env python=3.9
     ```
   - Activate the virtual environment:
     ```
     conda activate sign_language_env
     ```
4. **Clone the Project Repository**:
   - Clone the Sign Language Understanding Project repository from GitHub using the following command:
     ```
     git clone https://github.com/COMPSYS302/project-python-sunshine-group-23.git
     ```
   - Navigate to the project directory:
     ```
     cd project-python-sunshine-group-23
     ```

5. **Install Required Libraries**:
   - Install the necessary libraries using pip. Run the following commands:
     ```
     pip install -r requirements.txt
     ```
   - These commands will install PyTorch, OpenCV, PyQt5, NumPy and other dependencies with the specified versions.

6. **Run the Application**:
    - Navigate to the App directory:
     ```
     cd project-python-sunshine-group-23/App
     ```
   - Launch the Sign Language Understanding application by running the main script:
     ```
     python ASL_Trainer.py
     ```
   - The application will start, and you can interact with the GUI to perform sign language recognition tasks.
  
    *Note: You have to copy `pretrained_models` folder and `dataset.csv` into the project root directory: `/project-python-sunshine-group-23` 
   - These files can be found in our canvas submitted zip file.

8. Select the preferred setting and start the training.

9. Now, you can test your model using the dataset image or using the webcam.


**System Requirements**:
- GPU: NVIDIA GPU with CUDA support (recommended for faster training)

**Additional Notes**:
- If you encounter any issues during the installation process, refer to the documentation of the respective libraries for troubleshooting guidance.
- Make sure you have the latest version of pip installed to avoid compatibility issues.
- If you are using a GPU, ensure that you have the appropriate NVIDIA drivers and CUDA toolkit installed.

By following these setup instructions and meeting the system requirements, you will have a complete environment ready for running the Sign Language Understanding Project. The combination of Python, PyTorch, OpenCV, PyQt5, and NumPy provides a powerful and flexible framework for developing and deploying sign language recognition models.

## Usage

#Training your model

**Data**
1. Launch the Sign Language Understanding application via aforementioned step 6.
2. On startup, you will be on the 'Data' tab. From here, select 'Upload Dataset' and navigate to your .csv file, then confirm to complete upload.
3. After a short delay, your dataset will be displayed on the GUI as images. From here, you can search to filter images via their label

**Train**

5. Next, select the 'Train' tab. Here you can select your model via the dropdown menu.
6. On model selection, the hyperparameters will auto default to recommended settings for that model. Adjust these per your use.
7. Click the 'Start Training' button to begin training the selected model, with the configured hyperparamters, on the previously uploaded dataset. This will open a new window.
8. The new training window has 2 plots which update every epoch to display Training Loss and Validation Accuracy. Underneath, there is a progress bar which updates every loop. From here, clicking the 'Stop Training' button will terminate training and save the model up to that point under:
```
./App/user_trained_models/your_model_here.pth
```

**Test**

8. Next, you can navigate to the 'Test' tab to test your model. Here, at the top of the GUI window, click 'Load Model' and find your .pth file. This file should be inside the above mentioned directory.
9. Then, select the Test set by clicking the respective button. After a short delay, your test set should be displayed as images on the GUI.
10. Click on any image to use your loaded model and make prediction.



**Live stream prediction**
1. Ensure that your webcam is connected and functioning properly.
2. Position yourself in front of the webcam, ensuring your hands are visible.
3. Navigate to the 'Test' tab and select 'Live Testing' to launch the live prediction script
4. OR click on 'Open Webcam' to take a still photo.
5. Perform sign language gestures; the application will analyze and interpret them in real-time.
6. The interpreted sign language will be displayed on the application's interface.
