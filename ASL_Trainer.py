import os
import math
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
from PyQt5.QtCore import Qt, QSize
from ASL_Training import Ui_training_session
from ASL_CAM import Camera
from csv_to_images import csv_to_images
from ClickableQLabel import ClickableLabel


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(600, 800)
        MainWindow.setMinimumSize(QtCore.QSize(600, 800))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 600, 800))
        self.tabWidget.setMinimumSize(QtCore.QSize(600, 800))
        self.tabWidget.setMaximumSize(QtCore.QSize(600, 800))
        self.tabWidget.setAutoFillBackground(False)
        self.tabWidget.setStyleSheet("QTabWidget::pane { border: 1px solid #1F1F1F; background: #1F1F1F;}\n"
                                     "QTabBar::tab { background: #333333; color: #ffffff; padding: 10px;}\n"
                                     "QTabBar::tab:selected { background: #1F1F1F; color: #FFFFFF;}")
        
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
        self.data_image_title = QtWidgets.QLabel(self.load)
        self.data_image_title.setGeometry(QtCore.QRect(30, 140, 91, 16))
        self.data_image_title.setStyleSheet("QLabel {color: white; font-weight: bold;}\n")

        # Create Scroll Area
        self.scroll = QtWidgets.QScrollArea(self.load)
        self.scroll.setGeometry(QtCore.QRect(30, 160, 541, 551))
        self.scroll.setWidgetResizable(True)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Create a widget to hold the grid layout 
        self.grid_widget = QtWidgets.QWidget()
        # Create a GridLayout
        self.data_grid = QtWidgets.QGridLayout()
        self.data_grid.setSpacing(10)
        # Put the GridLayout in the Frame
        self.scroll.setWidget(self.grid_widget)
    
# =================================|| Train Tab ||=================================

        self.tabWidget.addTab(self.load, "")
        self.train = QtWidgets.QWidget()
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
        self.train_slider = QtWidgets.QSlider(self.train_options_frame)
        self.train_slider.setStyleSheet("QSlider::groove:horizontal {height: 4px; background: #F0F0F0; margin: 2px 0; border-radius: 4px;}\n"
                                        "QSlider::handle:horizontal {background: #345CC1; width: 24px; height: 18px; margin: -5px 0;}\n"
                                        "QSlider::handle:horizontal:hover {background: #2A4BA0;}\n"
                                        "QSlider::handle:horizontal:pressed {background: #1E3C8C;}")
        self.train_slider.setMaximum(99)
        self.train_slider.setOrientation(QtCore.Qt.Horizontal)
        self.gridLayout_2.addWidget(self.train_slider, 1, 1, 1, 1)
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
        self.train_combobox = QtWidgets.QComboBox(self.train_model_frame)
        self.train_combobox.setMinimumSize(QtCore.QSize(60, 30))
        self.gridLayout_3.addWidget(self.train_combobox, 2, 1, 1, 1)
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
        self.test_button_frame.setMinimumSize(QtCore.QSize(540, 0))
        self.test_button_frame.setStyleSheet("QFrame {background-color: #333333;}")
        self.test_button_frame.setFrameShape(QtWidgets.QFrame.Panel)
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.test_button_frame)
        # Create Button to access files for testing
        self.test_file_button = QtWidgets.QPushButton(self.test_button_frame, clicked= lambda: self.selectToTest())
        self.test_file_button.setMinimumSize(QtCore.QSize(120, 40))
        self.test_file_button.setBaseSize(QtCore.QSize(120, 40))
        self.test_file_button.setStyleSheet("QPushButton {background-color: #345CC1; color: white; border: none; padding: 8px 16px;}\n"
                                              "QPushButton:hover {background-color: #2A4BA0;}\n"
                                              "QPushButton:pressed {background-color: #1E3C8C;}")
        self.horizontalLayout.addWidget(self.test_file_button)
        # Create Button to access webcam for testing
        self.test_webcam_button = QtWidgets.QPushButton(self.test_button_frame, clicked= lambda: self.openWebcam())
        self.test_webcam_button.setMinimumSize(QtCore.QSize(120, 40))
        self.test_webcam_button.setBaseSize(QtCore.QSize(120, 40))
        self.test_webcam_button.setStyleSheet("QPushButton {background-color: #345CC1; color: white; border: none; padding: 8px 16px;}\n"
                                              "QPushButton:hover {background-color: #2A4BA0;}\n"
                                              "QPushButton:pressed {background-color: #1E3C8C;}")
        self.horizontalLayout.addWidget(self.test_webcam_button)
       
        self.test_button_title = QtWidgets.QLabel(self.test)
        self.test_button_title.setGeometry(QtCore.QRect(30, 30, 141, 16))
        self.test_button_title.setStyleSheet("QLabel {color: white; font-weight: bold;}")

        # Create Scroll Area 
        self.test_scroll = QtWidgets.QScrollArea(self.test)
        self.test_scroll.setGeometry(QtCore.QRect(30, 160, 541, 551))
        self.test_scroll.setWidgetResizable(True)
        self.test_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.test_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Create a widget to hold the grid layout 
        self.test_grid_widget = QtWidgets.QWidget()
        # Create a GridLayout
        self.test_grid = QtWidgets.QGridLayout()
        self.test_grid.setSpacing(10)
        # Put the GridLayout in the Frame
        self.test_scroll.setWidget(self.test_grid_widget)

        # Create a Title for the Scroll area
        self.test_image_title = QtWidgets.QLabel(self.test)
        self.test_image_title.setGeometry(QtCore.QRect(30, 140, 91, 16))
        self.test_image_title.setStyleSheet("QLabel {color: white; font-weight: bold;}\n")
        
        self.tabWidget.addTab(self.test, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
# =================================|| Functions ||=================================

    # Function to upload files for data set
    def uploadFiles(self):
        #accessing the file path to pictures
        file = QtWidgets.QFileDialog.getOpenFileNames(None, 'Open file', 'c:\\',"Image files (*.csv)")

        if file and file[0]:
            file_path = file[0][0]
            print(file[0][0])
            csv_to_images(file_path, 'images', 100, (28, 28))

            # Clean the grid layout
            while self.data_grid.count():
                child = self.data_grid.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()

            positions = [(i, j) for i in range(25) for j in range(4)]
            for position, image in zip(positions, os.listdir('images')):
                image_path = os.path.join('images', image)
                image_data = open(image_path, 'rb').read()  # Read image data as bytes
                
                # Create a QLabel to hold images  
                QLabel = QtWidgets.QLabel()
                QLabel.setFixedSize(90, 90)
                # Create a QPixmap to display images                 
                self.qp = QPixmap()
                self.qp.loadFromData(image_data)
                self.qp = self.qp.scaled(90, 90, QtCore.Qt.KeepAspectRatio)
                QLabel.setPixmap(self.qp)

                 # Add the QLabel to the GridLayout
                self.data_grid.addWidget(QLabel, *position)  

            self.grid_widget.setLayout(self.data_grid)       
                 
	# Funtion to upload files for test set 
    def selectToTest(self):
         #accessing the file path to pictures
        file = QtWidgets.QFileDialog.getOpenFileNames(None, 'Open file', 'c:\\',"Image files (*.csv)")

        if file and file[0]:
            file_path = file[0][0]
            print(file[0][0])
            csv_to_images(file_path, 'test_images', 100, (28, 28))

            # Clean the grid layout
            while self.test_grid.count():
                child = self.test_grid.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()

            positions = [(i, j) for i in range(25) for j in range(4)]
            for position, image in zip(positions, os.listdir('test_images')):
                image_path = os.path.join('test_images', image)
                image_data = open(image_path, 'rb').read()  # Read image data as bytes
                
                # Create a ClickLabel to hold images  
                ClickLabel = ClickableLabel()
                ClickLabel.setFixedSize(90, 90)
                # Create a QPixmap to display images                 
                self.qp = QPixmap()
                self.qp.loadFromData(image_data)
                self.qp = self.qp.scaled(90, 90, QtCore.Qt.KeepAspectRatio)
                ClickLabel.setPixmap(self.qp)

                # Connect the click signal
                ClickLabel.clicked.connect(self.labelClicked)

                 # Add the ClickLabel to the GridLayout
                self.test_grid.addWidget(ClickLabel, *position)  

            self.test_grid_widget.setLayout(self.test_grid)

    # Function that opens another window 
    def startTraining(self):
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_training_session()
        self.ui.setupUi(self.window)
        self.window.show()

    def openWebcam(self):
        # Check if an instance of QApplication already exists
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)

        window = Camera()
        window.show()
        if app is None:
            sys.exit(app.exec_())

    def labelClicked(self):
        

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "ASL TRAINER"))
        self.data_file_button.setText(_translate("MainWindow", "Uplaod Dataset"))
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
        self.test_file_button.setText(_translate("MainWindow", "Select Dataset"))
        self.test_webcam_button.setText(_translate("MainWindow", "Open Webcam"))
        self.test_button_title.setText(_translate("MainWindow", "Select Method to Test"))
        self.test_image_title.setText(_translate("MainWindow", "Data Set:"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.test), _translate("MainWindow", "  Test  "))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
    