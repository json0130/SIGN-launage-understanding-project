# Sign Language Understanding Project

This project aims to analyze and interpret sign language using three different machine learning models: Inception v3, YOLOv8, and SUNSHINE23. The system can recognize and understand sign language in real-time by leveraging the user's webcam input.

## Features

- User-friendly interface built with PyQt
- Robust backend developed in Python
- Integration of three powerful models for sign language analysis:
  - Inception v3
  - YOLOv8
  - SUNSHINE23

## Installation

To set up and run the Sign Language Understanding Project, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/your-username/sign-language-understanding.git
   ```

2. Navigate to the project directory:
   ```
   cd sign-language-understanding
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Collect and prepare the sign language dataset. Ensure that the dataset is properly labeled and organized.

5. Train the models by running the following command:
   ```
   python train.py
   ```

6. Once the models are trained, you can launch the application:
   ```
   python main.py
   ```

## Usage

1. Launch the Sign Language Understanding application.

2. Ensure that your webcam is connected and functioning properly.

3. Position yourself in front of the webcam, making sure your hands are visible.

4. Perform sign language gestures; the application will analyze and interpret them in real time.

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
