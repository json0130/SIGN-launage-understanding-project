import os
import math
import resnet_app_test
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QThread
from ASL_Training import Ui_training_session
from ASL_Results import Ui_Results
from ASL_CAM import Camera
from csv_to_images import csv_to_images
from ClickableQLabel import ClickableLabel, ClickLabel
from PIL import Image

from model_scripts import train_mobilenetv2, train_inceptionv3, train_resnet50, train_sunshine23

class TrainingWorker(QThread):
    update_plot = pyqtSignal(list, list, int)
    update_progress = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, train_func, batch_size, epochs, train_test_ratio, image_file):
        super().__init__()
        self.train_func = train_func
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_test_ratio = train_test_ratio
        self.image_file = image_file
        self.stop_requested = False

    def run(self):
        self.train_func(self.image_file, self.batch_size, self.epochs, self.train_test_ratio, self.update_plot, self.update_progress, self)
        self.finished.emit()

    def stop(self):
        self.stop_requested = True

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        self.image_file = None

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(600, 800)
        MainWindow.setMinimumSize(QtCore.QSize(600, 880))
        MainWindow.setStyleSheet("QMainWindow {background: #4d4d4d;}\n")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 600, 880))
        self.tabWidget.setMinimumSize(QtCore.QSize(600, 880))
        self.tabWidget.setMaximumSize(QtCore.QSize(600, 880))
        self.tabWidget.setAutoFillBackground(False)
        self.tabWidget.setStyleSheet("QTabWidget::pane { border: 1px solid #1F1F1F; background: #1F1F1F;}\n"
                                     "QTabBar::tab { background: #333333; color: #ffffff; padding: 10px;}\n"
                                     "QTabBar::tab:selected { background: #1F1F1F; color: #FFFFFF;}")
        self.first_time_train_tab = True

        # Creating a tool bar 
        toolbar = QToolBar("Camera Tool Bar") 
        toolbar.setStyleSheet("QToolBar { background: #333333; color: white; }"
                              "QToolBar::separator { background: white; }"
                              "QToolButton { background: #333333; color: white; border: none; padding: 5px; }"
                              "QToolButton:hover { background: #4d4d4d; }"
                              "QToolButton:pressed { background: #1F1F1F; }")
        MainWindow.addToolBar(toolbar)

        # Create Load model action
        self.model_file = None
        load_action = QAction("Load Model", MainWindow) 
        load_action.setStatusTip("This will load a model") 
        load_action.setToolTip("Load Model") 
        load_action.triggered.connect(self.loadModel) 
        # Add to tool bar 
        toolbar.addAction(load_action) 
        
# =================================|| Data Tab ||=================================
        # Create a Tab for Data
        self.tabWidget.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.tabWidget.setDocumentMode(False)
        self.load = QtWidgets.QWidget() 
        # Create a Frame for buttons
        self.Data_button_frame = QtWidgets.QFrame(self.load)
        self.Data_button_frame.setGeometry(QtCore.QRect(30, 50, 540, 60))
        self.Data_button_frame.setMinimumSize(QtCore.QSize(540, 0))
        self.Data_button_frame.setStyleSheet("QFrame {background-color: #333333;}")
        self.Data_button_frame.setFrameShape(QtWidgets.QFrame.Panel)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.Data_button_frame)

        spacerItem = QtWidgets.QSpacerItem(191, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem)
        # Create Button
        self.data_file_button = QtWidgets.QPushButton(self.Data_button_frame, clicked= lambda: self.uploadFiles())
        self.data_file_button.setMinimumSize(QtCore.QSize(120, 40))
        self.data_file_button.setBaseSize(QtCore.QSize(120, 40))
        self.data_file_button.setStyleSheet("QPushButton {background-color: #345CC1; color: white; border: none; padding: 8px 16px;}\n"
                                            "QPushButton:hover { background-color: #2A4BA0;}\n"
                                            "QPushButton:pressed { background-color: #1E3C8C;}")
        self.data_button_title = QtWidgets.QLabel(self.load)
        self.data_button_title.setGeometry(QtCore.QRect(30, 30, 181, 16))
        self.data_button_title.setStyleSheet("QLabel {color: white; font-weight: bold;}")
        # Create button
        self.horizontalLayout_3.addWidget(self.data_file_button)
        spacerItem1 = QtWidgets.QSpacerItem(191, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem1)

        # Create a Title for the images                          
        self.data_image_title = QtWidgets.QLabel(self.load)
        self.data_image_title.setGeometry(QtCore.QRect(30, 125, 91, 16))
        self.data_image_title.setStyleSheet("QLabel {color: white; font-weight: bold;}\n")  

        #Create a Search bar for the data set
        self.data_search = QtWidgets.QLineEdit(self.load)
        self.data_search.setGeometry(QtCore.QRect(30, 147, 200, 25))
        self.data_search.setStyleSheet("QLineEdit {background-color: #333333; color: white; border: 1px solid #345CC1;}")
        self.data_search.setPlaceholderText("Search")
        self.data_search.textChanged.connect(self.perform_search)
        self.data_search.setClearButtonEnabled(True)                           

        # Create Scroll Area
        self.scroll = QtWidgets.QScrollArea(self.load)
        self.scroll.setGeometry(QtCore.QRect(30, 180, 541, 580))
        self.scroll.setWidgetResizable(True)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Create a list to hold the image widgets
        self.data_image_widgets = []

        # Create a widget to hold the grid layout 
        self.grid_widget = QtWidgets.QWidget()
        # Create a GridLayout
        self.data_grid = QtWidgets.QGridLayout()
        self.data_grid.setSpacing(10)
        # Put the GridLayout in the Frame
        self.scroll.setWidget(self.grid_widget)
    
# =================================|| Train Tab ||=================================

        # Create a dictionary to store default values for each model
        self.model_defaults = {
            "InceptionV3": {"batch_size": 32, "epochs": 30, "train_ratio": 80},
            "MobileNet_V2": {"batch_size": 64, "epochs": 30, "train_ratio": 80},
            "ResNet_50": {"batch_size": 64, "epochs": 30, "train_ratio": 80},
            "Sunshine23": {"batch_size": 64, "epochs": 30, "train_ratio": 80}
        }
        
        self.tabWidget.addTab(self.load, "")
        self.train = QtWidgets.QWidget()

        # Create a Frame for the model selection
        self.train_options_frame = QtWidgets.QFrame(self.train)
        self.train_options_frame.setGeometry(QtCore.QRect(30, 260, 531, 451))
        self.train_options_frame.setStyleSheet("QFrame {background-color: #333333;}")
        self.train_options_frame.setFrameShape(QtWidgets.QFrame.Panel)
        self.gridLayout_2 = QtWidgets.QGridLayout(self.train_options_frame)
        self.train_batch = QtWidgets.QSpinBox(self.train_options_frame)
        self.train_batch.setMaximum(500)
        self.train_batch.setSingleStep(10)
        self.gridLayout_2.addWidget(self.train_batch, 7, 1, 1, 1)
        self.train_batch_label = QtWidgets.QLabel(self.train_options_frame)
        self.train_batch_label.setStyleSheet("QLabel {color: white;}")
        self.gridLayout_2.addWidget(self.train_batch_label, 8, 1, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem2, 11, 1, 1, 1)
        # Create a slider for the training ratio
        self.train_slider = QtWidgets.QSlider(self.train_options_frame)
        self.train_slider.setStyleSheet("QSlider::groove:horizontal {height: 4px; background: #F0F0F0; margin: 2px 0; border-radius: 4px;}\n"
                                        "QSlider::handle:horizontal {background: #345CC1; width: 24px; height: 18px; margin: -5px 0;}\n"
                                        "QSlider::handle:horizontal:hover {background: #2A4BA0;}\n"
                                        "QSlider::handle:horizontal:pressed {background: #1E3C8C;}")
        self.train_slider.setMaximum(99)
        self.train_slider.setOrientation(QtCore.Qt.Horizontal)
        self.gridLayout_2.addWidget(self.train_slider, 1, 1, 1, 1)
        # Create a SpinBox for epoch
        self.train_epoch = QtWidgets.QSpinBox(self.train_options_frame)
        self.train_epoch.setMaximum(200)
        self.train_epoch.setSingleStep(10)
        self.gridLayout_2.addWidget(self.train_epoch, 4, 1, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem3, 1, 0, 1, 1)
        self.train_start_button = QtWidgets.QPushButton(self.train_options_frame, clicked= lambda: self.startTraining())
        self.train_start_button.setMinimumSize(QtCore.QSize(0, 60))
        self.train_start_button.setStyleSheet("QPushButton {background-color: #345CC1; color: white; border: none; padding: 8px 16px;}\n"   
                                              "QPushButton:hover {background-color: #2A4BA0;}\n"
                                              "QPushButton:pressed {background-color: #1E3C8C;}\n")
        self.gridLayout_2.addWidget(self.train_start_button, 10, 1, 1, 1)
        self.epoch_label = QtWidgets.QLabel(self.train_options_frame)
        self.epoch_label.setStyleSheet("QLabel {color: white;}")
        self.gridLayout_2.addWidget(self.epoch_label, 5, 1, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem4, 1, 2, 1, 1)
        spacerItem5 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem5, 9, 1, 1, 1)
        self.train_batch_title = QtWidgets.QLabel(self.train_options_frame)
        self.train_batch_title.setMinimumSize(QtCore.QSize(0, 50))
        self.train_batch_title.setStyleSheet("QLabel {color: white; font-weight: bold;}")
        self.gridLayout_2.addWidget(self.train_batch_title, 6, 1, 1, 1)
        self.train_epoch_title = QtWidgets.QLabel(self.train_options_frame)
        self.train_epoch_title.setMinimumSize(QtCore.QSize(0, 50))
        self.train_epoch_title.setStyleSheet("QLabel {color: white; font-weight: bold;}")
        self.gridLayout_2.addWidget(self.train_epoch_title, 3, 1, 1, 1)
        self.train_ratio_title = QtWidgets.QLabel(self.train_options_frame)
        self.train_ratio_title.setMinimumSize(QtCore.QSize(0, 40))
        self.train_ratio_title.setStyleSheet("QLabel {color: white; font-weight: bold;}")
        self.gridLayout_2.addWidget(self.train_ratio_title, 0, 1, 1, 1)
        self.train_ratio_label = QtWidgets.QLabel(self.train_options_frame)
        self.train_ratio_label.setStyleSheet("QLabel {color: white;}")
        self.gridLayout_2.addWidget(self.train_ratio_label, 2, 1, 1, 1)
        self.train_model_frame = QtWidgets.QFrame(self.train)
        self.train_model_frame.setGeometry(QtCore.QRect(30, 50, 528, 101))
        self.train_model_frame.setStyleSheet("QFrame { background-color: #333333;}")
        self.train_model_frame.setFrameShape(QtWidgets.QFrame.Panel)
        self.gridLayout_3 = QtWidgets.QGridLayout(self.train_model_frame)

        # model selection combobox
        self.train_combobox = QtWidgets.QComboBox(self.train_model_frame)
        self.train_combobox.setMinimumSize(QtCore.QSize(60, 30))
        self.gridLayout_3.addWidget(self.train_combobox, 2, 1, 1, 1)
        self.train_combobox.addItems(["Select your Model","InceptionV3", "MobileNet_V2", "ResNet_50", "Sunshine23"])

        #Connect the currentIndexChanged signal to a function
        self.train_combobox.currentIndexChanged.connect(self.update_model_defaults)

        # Mapping model names to script paths
        self.model_scripts = {
            "InceptionV3": train_inceptionv3.train,
            "MobileNet_V2": train_mobilenetv2.train,
            "ResNet_50": train_resnet50.train,
            "Sunshine23": train_sunshine23.train,
        }

        # Disable the start button initially
        self.train_start_button.setEnabled(False)

        # Connect the combobox change to enable/disable button
        self.train_combobox.currentIndexChanged.connect(self.check_model_selection)

        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem6, 2, 7, 1, 1)
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem7, 2, 0, 1, 1)
        self.train_model_title = QtWidgets.QLabel(self.train_model_frame)
        self.train_model_title.setMinimumSize(QtCore.QSize(0, 30))
        self.train_model_title.setStyleSheet("QLabel {color: white; font-weight: bold;}")
        self.gridLayout_3.addWidget(self.train_model_title, 0, 1, 1, 1)
        spacerItem8 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_3.addItem(spacerItem8, 3, 1, 1, 1)
        self.train_line_bottom = QtWidgets.QFrame(self.train)
        self.train_line_bottom.setGeometry(QtCore.QRect(30, 230, 531, 16))
        self.train_line_bottom.setFrameShape(QtWidgets.QFrame.HLine)
        self.train_line_top = QtWidgets.QFrame(self.train)
        self.train_line_top.setGeometry(QtCore.QRect(30, 160, 531, 16))
        self.train_line_top.setFrameShape(QtWidgets.QFrame.HLine)
# =================================|| Test Tab ||=================================

        self.tabWidget.addTab(self.train, "")
        self.test = QtWidgets.QWidget()
        self.test_button_frame = QtWidgets.QFrame(self.test)
        self.test_button_frame.setGeometry(QtCore.QRect(30, 50, 540, 60))
        self.test_button_frame.setStyleSheet("QFrame {background-color: #333333;}")
        self.test_button_frame.setFrameShape(QtWidgets.QFrame.Panel)
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.test_button_frame)
        # Create Button to access files for testing
        self.test_file_button = QtWidgets.QPushButton(self.test_button_frame, clicked= lambda: self.selectToTest())
        self.test_file_button.setMinimumSize(QtCore.QSize(80, 40))
        self.test_file_button.setStyleSheet("QPushButton {background-color: #345CC1; color: white; border: none; padding: 8px 16px;}\n"
                                              "QPushButton:hover {background-color: #2A4BA0;}\n"
                                              "QPushButton:pressed {background-color: #1E3C8C;}")
        self.horizontalLayout.addWidget(self.test_file_button)
        # Create Button to access webcam for testing
        self.test_webcam_button = QtWidgets.QPushButton(self.test_button_frame, clicked= lambda: self.openWebcam())
        self.test_webcam_button.setMinimumSize(QtCore.QSize(80, 40))
        self.test_webcam_button.setStyleSheet("QPushButton {background-color: #345CC1; color: white; border: none; padding: 8px 16px;}\n"
                                              "QPushButton:hover {background-color: #2A4BA0;}\n"
                                              "QPushButton:pressed {background-color: #1E3C8C;}")
        self.horizontalLayout.addWidget(self.test_webcam_button)
        # Create Button to access webcam for live testing
        self.test_live_button = QtWidgets.QPushButton(self.test_button_frame, clicked= lambda: self.openLivecam())
        self.test_live_button.setMinimumSize(QtCore.QSize(80, 40))
        self.test_live_button.setStyleSheet("QPushButton {background-color: #345CC1; color: white; border: none; padding: 8px 16px;}\n"
                                              "QPushButton:hover {background-color: #2A4BA0;}\n"
                                              "QPushButton:pressed {background-color: #1E3C8C;}")
        self.horizontalLayout.addWidget(self.test_live_button)
       
        self.test_button_title = QtWidgets.QLabel(self.test)
        self.test_button_title.setGeometry(QtCore.QRect(30, 30, 141, 16))
        self.test_button_title.setStyleSheet("QLabel {color: white; font-weight: bold;}")

        # Create a Title for the Scroll area
        self.test_image_title = QtWidgets.QLabel(self.test)
        self.test_image_title.setGeometry(QtCore.QRect(30, 125, 91, 16))
        self.test_image_title.setStyleSheet("QLabel {color: white; font-weight: bold;}\n")

        # Create a refresh clickable label
        self.refresh_label = ClickLabel(self.test)
        self.refresh_label.setGeometry(QtCore.QRect(120, 125, 140, 16))
        self.refresh_label.setStyleSheet("QLabel {color: white; font-weight: bold;}\n")
        self.refresh_label.clicked.connect(self.refresh_test_grid)

        #Create a Search bar for the data set
        self.test_search = QtWidgets.QLineEdit(self.test)
        self.test_search.setGeometry(QtCore.QRect(30, 147, 200, 25))
        self.test_search.setStyleSheet("QLineEdit {background-color: #333333; color: white; border: 1px solid #345CC1;}")
        self.test_search.setPlaceholderText("Search")
        self.test_search.textChanged.connect(self.perform_test_search)
        self.test_search.setClearButtonEnabled(True)  

        # Create Scroll Area 
        self.test_scroll = QtWidgets.QScrollArea(self.test)
        self.test_scroll.setGeometry(QtCore.QRect(30, 180, 541, 580))
        self.test_scroll.setWidgetResizable(True)
        self.test_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.test_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.test_image_widgets = []

        # Create a widget to hold the grid layout 
        self.test_grid_widget = QtWidgets.QWidget()
        # Create a GridLayout
        self.test_grid = QtWidgets.QGridLayout()
        self.test_grid.setSpacing(10)
        # Put the GridLayout in the Frame
        self.test_scroll.setWidget(self.test_grid_widget)
        
        self.tabWidget.addTab(self.test, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # added
        self.training_plot_window = None

# =================================|| Functions ||=================================
    # Function to load model
    def loadModel(self):
        model_file = QtWidgets.QFileDialog.getOpenFileNames(None, 'Open file', 'c:\\',"Model files (*.pth)")

        if model_file and model_file[0]:
            self.model_file = model_file[0][0]

    def alphabetcheck(self, input):
        #check if input is longer than 1 character
        if len(input) > 1:
            return input

        #checks if a string is a letter and if it is in the alphabet it converts it into a number from 0-25
        #input is always lower case. 
        if input.isalpha():
            print(ord(input)-97)
            return str(ord(input)-97)
        else:
            return input
    
    def perform_search(self):
        file_path = 'images'
        # Check if the file path exists
        if not os.path.exists(file_path):
            return
        
        search_text = self.data_search.text().strip().lower()
        search_text = self.alphabetcheck(search_text)

        # Check if seach test is empty
        if not search_text:
            for label, image_file in self.data_image_widgets:
                label.setVisible(True)
            return

        for label, image_file in self.data_image_widgets:
            if not image_file.startswith('label_' + search_text +'_'):
                label.setVisible(False)
            else:
                label.setVisible(True)

    def perform_test_search(self):
        file_path = 'test_images'
        # Check if the file path exists
        if not os.path.exists(file_path):
            return
        
        search_text = self.test_search.text().strip().lower()
        search_text = self.alphabetcheck(search_text)

        # Check if seach test is empty
        if not search_text:
            for label, image_file in self.test_image_widgets:
                label.setVisible(True)
            return

        for label, image_file in self.test_image_widgets:
            if not image_file.startswith('label_' + search_text +'_'):
                label.setVisible(False)
            else:
                label.setVisible(True)
            
    def uploadFiles(self):
        #accessing the file path to pictures
        self.image_file = QtWidgets.QFileDialog.getOpenFileNames(None, 'Open file', 'c:\\',"Image files (*.csv)")
        self.loadImages()
        self.image_file = self.image_file[0][0]

    def loadImages(self):
        if self.image_file and self.image_file[0]:
            file_path = self.image_file[0][0]
            
            # Clean the grid layout
            while self.data_grid.count():
                child = self.data_grid.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()

            # Setting up the grid layout
            number_of_images = csv_to_images(file_path, 'images', (28, 28))
            number_of_images = math.ceil(number_of_images/4)
            positions = [(i, j) for i in range(number_of_images) for j in range(4)]
            self.data_image_widgets.clear()

            for position, image in zip(positions, os.listdir('images')):
                image_path = os.path.join('images', image)
                image_data = open(image_path, 'rb').read()  # Read image data as bytes
                
                # Create a QLabel to hold images  
                Label = QtWidgets.QLabel()
                Label.setFixedSize(90, 90)

                # Create a QPixmap to display images                 
                qp = QPixmap()
                qp.loadFromData(image_data)
                qp = qp.scaled(90, 90, QtCore.Qt.KeepAspectRatio)
                Label.setPixmap(qp)

                 # Add the QLabel to the GridLayout
                self.data_grid.addWidget(Label, *position)  
                self.data_image_widgets.append((Label, image))

            self.grid_widget.setLayout(self.data_grid)       
                 
    def selectToTest(self):
         #accessing the file path to pictures
        file = QtWidgets.QFileDialog.getOpenFileNames(None, 'Open file', 'c:\\',"Image files (*.csv)")
        self.update_test_grid(file)

    def refresh_test_grid(self):
        self.update_test_grid('null', True)
    
    def update_test_grid(self, file, skip=False):
        if file and file[0] and not skip:
            file_path = file[0][0]

            # Check if the file path exists
            if not os.path.exists(file_path):
                return

            csv_to_images(file_path, 'test_images', (28, 28))

        file_size = len(os.listdir('test_images'))
        self.test_image_widgets.clear()
        
        # Clean the grid layout
        while self.test_grid.count():
            child = self.test_grid.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        positions = [(i,j) for i in range(math.ceil(file_size/4)) for j in range(4)]
        for position, image in zip(positions, os.listdir('test_images')):
            image_path = os.path.join('test_images', image)

            # if the image's name does not start with 'converted_', convert into 28 x 28 grey scale image
            if not image.startswith('label_'):
                img = Image.open(image_path)
                
                # Formatting the image
                img = img.convert('L')
                img = img.resize((28, 28))
                left = int(img.size[0]/2-80/2)
                upper = int(img.size[1]/2-80/2)
                right = left +80
                lower = upper + 80
                img.crop((left, upper,right,lower))
                
                # Overwrite the original image
                img.save(image_path)

            # Read image data as bytes
            image_data = open(image_path, 'rb').read()  
            
            # Create a ClickLabel to hold images  
            ClickLabel = ClickableLabel(image_path)
            ClickLabel.setFixedSize(90, 90)

            # Create a QPixmap to display images                 
            self.qp = QPixmap()
            self.qp.loadFromData(image_data)
            self.qp = self.qp.scaled(90, 90, QtCore.Qt.KeepAspectRatio)
            ClickLabel.setPixmap(self.qp)

            # Connect the click signal
            ClickLabel.clicked.connect(self.labelClicked)

            #check if the image is already in the list
            if (ClickLabel, image) not in self.test_image_widgets:
                self.test_image_widgets.append((ClickLabel, image))

            # Add the ClickLabel to the GridLayout
            self.test_grid.addWidget(ClickLabel, *position) 
            self.test_image_widgets.append((ClickLabel, image))

        self.test_grid_widget.setLayout(self.test_grid)

    def check_model_selection(self):
        selected_model = self.train_combobox.currentText()
        if selected_model == "Select your Model":
            self.train_start_button.setEnabled(False)
        else:
            self.train_start_button.setEnabled(True)

    # Function that opens another window 
    def startTraining(self):

        selected_model = self.train_combobox.currentText()

        print(self.train_start_button)

        if selected_model == "Select your Model":
            return

        train_function = self.model_scripts[selected_model]

        batch_size = self.train_batch.value()
        epochs = self.train_epoch.value()
        train_test_ratio = self.train_slider.value() / 100.0

        # Check if dataset file path is set
        if not self.image_file:
            print("No dataset file selected")
            # Show error message
            error_msg = QMessageBox()
            error_msg.setIcon(QMessageBox.Critical)
            error_msg.setText("No dataset file selected")
            error_msg.setInformativeText("Please upload a dataset file before starting the training.")
            error_msg.setWindowTitle("Error")
            error_msg.exec_()
            return

        # Create and start the worker thread
        self.worker = TrainingWorker(train_function, batch_size, epochs, train_test_ratio, self.image_file)
        self.worker.update_plot.connect(self.updatePlot)
        self.worker.update_progress.connect(self.updateProgress)
        self.worker.finished.connect(self.onTrainingFinished)
        self.worker.start()

        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_training_session()
        self.ui.setupUi(self.window, self.worker)
        self.window.show()
        self.training_plot_window = self.window
    
    def update_model_defaults(self, index):
            selected_model = self.train_combobox.itemText(index)
            defaults = self.model_defaults.get(selected_model)

            if defaults:
                batch_size = defaults["batch_size"]
                epochs = defaults["epochs"]
                train_ratio = defaults["train_ratio"]

                # Update the widgets with the default values
                self.train_batch.setValue(batch_size)
                self.train_epoch.setValue(epochs)
                self.train_slider.setValue(train_ratio)
        
    def updatePlot(self, train_losses, val_accuracies, epoch):
        if self.training_plot_window:
            self.ui.update_plot(train_losses, val_accuracies, epoch)

    def updateProgress(self, value):
        if self.training_plot_window:
            self.ui.update_progress(value)

    def onTrainingFinished(self):
        print("Training completed successfully")
        # Show information message
        info_msg = QMessageBox()
        info_msg.setIcon(QMessageBox.Information)
        info_msg.setText("Training completed successfully")
        info_msg.setInformativeText("The training process has finished.")
        info_msg.setWindowTitle("Training Completed")
        info_msg.exec_()

    def openWebcam(self):
        # Check if an instance of QApplication already exists
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)

        window = Camera()
        window.show()
        if app is None:
            sys.exit(app.exec_())

    def openLivecam(self):
        import live_testing
        live_testing.run()

    # Function that runs when the label is clicked
    def labelClicked(self, object_path):
        print("clicked")
        print(object_path)

        # Check if model file is not empty
        if self.model_file is not None:
            results = resnet_app_test.predict_image(object_path, self.model_file)

            self.window = QtWidgets.QMainWindow()
            self.ui = Ui_Results()
            self.ui.setupUi(self.window, object_path, results)
            self.window.show()
        else:
            # Show error message
            error_msg = QMessageBox()
            error_msg.setIcon(QMessageBox.Critical)
            error_msg.setText("No model file selected")
            error_msg.setInformativeText("Please load a model file before testing.")
            error_msg.setWindowTitle("Error")
            error_msg.exec_()


# =================================|| UI ||=================================

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "ASL TRAINER"))
        self.data_file_button.setText(_translate("MainWindow", "Upload Dataset"))
        self.data_image_title.setText(_translate("MainWindow", "Data Set:"))
        self.data_button_title.setText(_translate("MainWindow", "Select Images For Data Set"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.load), _translate("MainWindow", "  Data  "))
        self.train_batch_label.setText(_translate("MainWindow", "0                                                250                                            500"))
        self.train_start_button.setText(_translate("MainWindow", "Start Training"))
        self.epoch_label.setText(_translate("MainWindow", "0                                                30                                               60"))
        self.train_batch_title.setText(_translate("MainWindow", "Batch Size"))
        self.train_epoch_title.setText(_translate("MainWindow", "Epochs"))
        self.train_ratio_title.setText(_translate("MainWindow", "Train/Test Ratio"))
        self.train_ratio_label.setText(_translate("MainWindow", "0                                                 50                                              99"))
        self.train_model_title.setText(_translate("MainWindow", "                                           Select Model"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.train), _translate("MainWindow", "  Train  "))
        self.test_file_button.setText(_translate("MainWindow", "Select Test Dataset"))
        self.test_webcam_button.setText(_translate("MainWindow", "Open Webcam"))
        self.test_live_button.setText(_translate("MainWindow", "Live Testing"))
        self.test_button_title.setText(_translate("MainWindow", "Select Method to Test"))
        self.test_image_title.setText(_translate("MainWindow", "Data Set:"))
        self.refresh_label.setText(_translate("MainWindow", "Refresh Images"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.test), _translate("MainWindow", "  Test  "))

if __name__ == "__main__":
    import sys
    #clear the images folder
    if os.path.exists('images'):
        for file in os.listdir('images'):
            os.remove(os.path.join('images', file))
    #clear the test_images folder
    if os.path.exists('test_images'):
        for file in os.listdir('test_images'):
            os.remove(os.path.join('test_images', file))
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
