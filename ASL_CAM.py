    
# importing required libraries 
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
import os 
import sys 
import time 
import cv2
import numpy as np
  
# Main window class 
class Camera(QMainWindow): 
    triggered = pyqtSignal()  # Define a custom signal

    # constructor 
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
        self.save_path = "" 
  
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

        # Creat changing save folder action 
        change_folder_action = QAction("Change save location", self) 
        change_folder_action.setStatusTip("Change folder where picture will be saved saved.") 
        change_folder_action.setToolTip("Change save location") 
        change_folder_action.triggered.connect(self.change_folder) 
        # Addto the tool bar 
        toolbar.addAction(change_folder_action) 
  
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
  
        # setting window title 
        self.setWindowTitle("PyQt5 Cam") 
        # showing the main window 
        self.show() 
  
    def select_camera(self, i): 
        # getting the selected camera 
        self.camera = QCamera(self.available_cameras[i]) 
        # setting view finder to the camera 
        self.camera.setViewfinder(self.viewfinder) 
        # setting capture mode to the camera 
        self.camera.setCaptureMode(QCamera.CaptureStillImage) 
        # if any error occur show the alert 
        self.camera.error.connect(lambda: self.alert(self.camera.errorString())) 
        # start the camera 
        self.camera.start() 
        # creating a QCameraImageCapture object 
        self.capture = QCameraImageCapture(self.camera) 
        # showing alert if error occur 
        self.capture.error.connect(lambda error_msg, error, msg: self.alert(msg)) 
        # when image captured showing message 
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
        #replace all / with \ in the path
        img_filename = img_filename.replace("/", "\\")
        # Capture the image
        self.capture.capture(img_filename)

        # Read the captured image using OpenCV
        img = cv2.imread(img_filename)
        if img is None:
            print(f"Error: Unable to load image from {img_filename}")
            return
        # Crop the image into a square from the center
        h, w = img.shape[:2]
        min_dim = min(h, w)
        start_x = (w - min_dim) // 2
        start_y = (h - min_dim) // 2
        cropped_img = img[start_y:start_y+min_dim, start_x:start_x+min_dim]
        # Resize to 28x28
        resized_img = cv2.resize(cropped_img, (28, 28))
        # Convert to grayscale
        gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        # Flatten the image
        flattened_img = gray_img.flatten()
        
        # Save as CSV
        csv_filename = os.path.join(self.save_path, "%s-%04d-%s.csv" % (self.current_camera_name, self.save_seq, timestamp))
        np.savetxt(csv_filename, flattened_img, delimiter=",")

        # Increment the sequence
        self.save_seq += 1
  
    def change_folder(self): 
        # open the dialog to select path 
        path = QFileDialog.getExistingDirectory(self,  
                                                "Picture Location", "") 
        # if path is selected 
        if path: 
            # update the path 
            self.save_path = path 
            # update the sequence 
            self.save_seq = 0
  
    def alert(self, msg): 
        # error message 
        error = QErrorMessage(self) 
        error.showMessage(msg) 
  
if __name__ == "__main__" : 
  App = QApplication(sys.argv) 
  window = Camera() 
  sys.exit(App.exec()) 
