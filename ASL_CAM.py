    
# importing required libraries 
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
import os 
import sys 
import time 
import numpy as np
  
# Main window class 
class Camera(QMainWindow): 
    def __init__(self): 
        super().__init__() 
        self.setGeometry(100, 100, 800, 600) 
        self.setStyleSheet("background : #333333;") 
        # Get available cameras 
        self.available_cameras = QCameraInfo.availableCameras() 
  
        # if no camera found 
        if not self.available_cameras: 
            # exit the code 
            sys.exit() 
  
        # Creat status bar 
        self.status = QStatusBar() 
        self.status.setStyleSheet("background : white;") 
        # Add status bar to the main window 
        self.setStatusBar(self.status) 
        # Path to save 
        script_dir = os.path.dirname(__file__)
        folder_path = "test_images" 
        self.save_path = os.path.join(script_dir, folder_path)

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
  
        # Creating QCameraViewfinder object 
        self.viewfinder = QCameraViewfinder() 
        # showing this viewfinder 
        self.viewfinder.show() 
        self.setCentralWidget(self.viewfinder) 
        # Set the default camera. 
        self.select_camera(0) 
  
        # Creating a tool bar 
        toolbar = QToolBar("Camera Tool Bar") 
        self.addToolBar(toolbar) 

        # Creat photo action
        click_action = QAction("Click photo", self) 
        click_action.setStatusTip("This will capture picture") 
        click_action.setToolTip("Capture picture") 
        click_action.triggered.connect(self.click_photo) 
        # Add to tool bar 
        toolbar.addAction(click_action) 
  
        # Create combo box for selecting camera 
        camera_selector = QComboBox() 
        camera_selector.setStatusTip("Choose camera to take pictures") 
        camera_selector.setToolTip("Select Camera") 
        camera_selector.setToolTipDuration(2500) 
        # Adding items to combo box 
        camera_selector.addItems([camera.description() for camera in self.available_cameras]) 
        # Call the select camera method 
        camera_selector.currentIndexChanged.connect(self.select_camera) 
        # Add to tool bar 
        toolbar.addWidget(camera_selector) 
        toolbar.setStyleSheet("background : white;") 
  
        self.setWindowTitle("PyQt5 Cam") 
        self.show() 
  
    def select_camera(self, i): 
        # Set selected camera 
        self.camera = QCamera(self.available_cameras[i]) 
        self.camera.setViewfinder(self.viewfinder) 
        self.camera.setCaptureMode(QCamera.CaptureStillImage) 

        # Start the camera 
        self.camera.error.connect(lambda: self.alert(self.camera.errorString())) 
        self.camera.start() 

        # Create QCameraImageCapture object 
        self.capture = QCameraImageCapture(self.camera) 
        self.capture.error.connect(lambda error_msg, error, msg: self.alert(msg)) 
        self.capture.imageCaptured.connect(lambda d,
                                           i: self.status.showMessage("Image Captured: "+ str(self.save_seq))) 
  
        # getting current camera name 
        self.current_camera_name = self.available_cameras[i].description() 
        # initial save sequence 
        self.save_seq = 0
  
    def click_photo(self): 
        # Time stamp
        timestamp = time.strftime("%d-%b-%Y-%H_%M_%S")
        # Create file path for the image
        img_filename = os.path.join(self.save_path, "%s-%04d-%s.jpg" % (self.current_camera_name, self.save_seq, timestamp))
    
        # Capture the image
        self.capture.capture(img_filename)

        # Increment the sequence
        self.save_seq += 1
  
    def alert(self, msg): 
        # error message 
        error = QErrorMessage(self) 
        error.showMessage(msg) 
  
if __name__ == "__main__" : 
  App = QApplication(sys.argv) 
  window = Camera() 
  sys.exit(App.exec()) 