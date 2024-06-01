import os
import math
from PyQt5 import QtCore, QtGui, QtWidgets
from ASL_Training import Ui_training_session
from csv_to_images import csv_to_images

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(600, 800)
        MainWindow.setMinimumSize(QtCore.QSize(600, 800))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(-10, 0, 600, 800))
        self.tabWidget.setMinimumSize(QtCore.QSize(600, 800))
        self.tabWidget.setMaximumSize(QtCore.QSize(600, 800))
        self.tabWidget.setAutoFillBackground(False)
        self.tabWidget.setStyleSheet("QTabWidget::pane { border: 1px solid #1F1F1F; background: #1F1F1F;}\n"
                                     "QTabBar::tab { background: #333333; color: #ffffff; padding: 10px;}\n"
                                     "QTabBar::tab:selected { background: #1F1F1F; color: #FFFFFF;}")
        
# =================================|| Data Tab ||=================================
        self.tabWidget.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.tabWidget.setDocumentMode(False)
        self.load = QtWidgets.QWidget()
        self.Data_button_frame = QtWidgets.QFrame(self.load)
        self.Data_button_frame.setGeometry(QtCore.QRect(30, 50, 540, 60))
        self.Data_button_frame.setMinimumSize(QtCore.QSize(540, 0))
        self.Data_button_frame.setStyleSheet("QFrame {background-color: #333333;}")
        self.Data_button_frame.setFrameShape(QtWidgets.QFrame.Panel)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.Data_button_frame)
        spacerItem = QtWidgets.QSpacerItem(191, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem)
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
        # Create a Frame
        self.display_frame = QtWidgets.QFrame(self.load)
        self.display_frame.setFrameShape(QtWidgets.QFrame.Box) 
        self.display_frame.setGeometry(QtCore.QRect(30, 160, 541, 551))
        self.display_frame.setMinimumSize(QtCore.QSize(540, 0))
        self.display_frame.setStyleSheet("QFrame {background-color: #333333;}")

        #insert the code into here

        # # Create Scroll Area
        # self.scroll = QtWidgets.QScrollArea()
        # self.scroll.setWidget(self.display_frame)
        # self.scroll.setWidgetResizable(False)
        # # Create Scroll Bar
        # self.data_scroll = self.scroll.verticalScrollBar()
        # self.data_scroll.setStyleSheet("QScrollBar:vertical {background: #333333; width: 15px; margin: 22px 0 22px 0;}\n"
        #                                "QScrollBar::handle:vertical {background: #545454; min-height: 20px; border-radius: 5px;}\n"
        #                                "QScrollBar::add-line:vertical {background: #333333; height: 20px; subcontrol-position: bottom; subcontrol-origin: margin;}\n"
        #                                "QScrollBar::sub-line:vertical {background: #333333; height: 20px; subcontrol-position: top; subcontrol-origin: margin;}\n"
        #                                "QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {width: 3px; height: 3px; background: white;}\n"
        #                                "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {background: none;}")
        # # Create a QVBoxLayout
        # self.layout = QtWidgets.QVBoxLayout()
        # # Add the Scroll Area to the Layout
        # self.layout.addWidget(self.scroll)
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
        # Create Button
        self.data_file_button_1 = QtWidgets.QPushButton(self.test_button_frame, clicked= lambda: self.selectToTest())
        self.data_file_button_1.setMinimumSize(QtCore.QSize(120, 40))
        self.data_file_button_1.setBaseSize(QtCore.QSize(120, 40))
        self.data_file_button_1.setStyleSheet("QPushButton {background-color: #345CC1; color: white; border: none; padding: 8px 16px;}\n"
                                              "QPushButton:hover {background-color: #2A4BA0;}\n"
                                              "QPushButton:pressed {background-color: #1E3C8C;}")
        self.horizontalLayout.addWidget(self.data_file_button_1)
        self.data_file_button_2 = QtWidgets.QPushButton(self.test_button_frame, clicked= lambda: self.openWebcam())
        self.data_file_button_2.setMinimumSize(QtCore.QSize(120, 40))
        self.data_file_button_2.setBaseSize(QtCore.QSize(120, 40))
        self.data_file_button_2.setStyleSheet("QPushButton {background-color: #345CC1; color: white; border: none; padding: 8px 16px;}\n"
                                              "QPushButton:hover {background-color: #2A4BA0;}\n"
                                              "QPushButton:pressed {background-color: #1E3C8C;}")
        self.horizontalLayout.addWidget(self.data_file_button_2)
        self.test_line = QtWidgets.QFrame(self.test)
        self.test_line.setGeometry(QtCore.QRect(30, 120, 540, 20))
        self.test_line.setMinimumSize(QtCore.QSize(540, 0))
        self.test_line.setFrameShape(QtWidgets.QFrame.HLine)        
        self.test_button_title = QtWidgets.QLabel(self.test)
        self.test_button_title.setGeometry(QtCore.QRect(30, 30, 141, 16))
        self.test_button_title.setStyleSheet("QLabel {color: white; font-weight: bold;}")

        self.frame_4 = QtWidgets.QFrame(self.test)
        self.frame_4.setGeometry(QtCore.QRect(30, 160, 541, 551))
        self.frame_4.setMinimumSize(QtCore.QSize(540, 0))
        self.frame_4.setStyleSheet("QFrame {background-color: #333333;}")
        self.frame_4.setFrameShape(QtWidgets.QFrame.Box)

        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.frame_4)
        self.gridLayout = QtWidgets.QGridLayout()
        self.horizontalLayout_4.addLayout(self.gridLayout)
        self.data_scroll_2 = QtWidgets.QScrollBar(self.frame_4)
        self.data_scroll_2.setStyleSheet("QScrollBar:vertical {background: #333333; width: 15px; margin: 22px 0 22px 0;}\n"
                                         "QScrollBar::handle:vertical {background: #545454; min-height: 20px; border-radius: 5px;}\n"
                                         "QScrollBar::add-line:vertical {background: #333333; height: 20px; subcontrol-position: bottom; subcontrol-origin: margin;}\n"
                                         "QScrollBar::sub-line:vertical {background: #333333; height: 20px; subcontrol-position: top; subcontrol-origin: margin;}\n"
                                         "QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {width: 3px; height: 3px; background: white;}\n"
                                         "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {background: none;}")
        self.data_scroll_2.setOrientation(QtCore.Qt.Vertical)
        self.horizontalLayout_4.addWidget(self.data_scroll_2)
        # Create a Frame
        self.data_image_title_2 = QtWidgets.QLabel(self.test)
        self.data_image_title_2.setGeometry(QtCore.QRect(30, 140, 91, 16))
        self.data_image_title_2.setStyleSheet("QLabel {color: white; font-weight: bold;}\n")
        self.tabWidget.addTab(self.test, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.statusbar)
        self.actionsave = QtWidgets.QAction(MainWindow)
        self.actionopen = QtWidgets.QAction(MainWindow)
        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
# =================================|| Functions ||=================================

    # Function to upload files for data set
    def uploadFiles(self):
        # Create a GridLayout
        self.data_display = QtWidgets.QGridLayout()
        # Put the GridLayout in the Frame
        self.display_frame.setLayout(self.data_display)

        #accessing the file path to pictures
        file = QtWidgets.QFileDialog.getOpenFileNames(None, 'Open file', 'c:\\',"Image files (*.csv)")

        if file and file[0]:
            file_path = file[0][0]
            print(file[0][0])
            csv_to_images(file_path, 'images', 30, (28, 28))
            positions = [(i, j) for i in range(8) for j in range(4)]
            for position, image in zip(positions, os.listdir('images')):
                print("monke")
                vBox = QtWidgets.QVBoxLayout()
                QLabel = QtWidgets.QLabel()
                QLabel.setFixedHeight(150) 
                QLabel.setFixedWidth(150)
                # Create a QPixmap to hold images                 
                self.qp = QPixmap()
                self.qp.loadFromData(image)
                self.qp = self.qp.scaled(150, 150, QtCore.Qt.KeepAspectRatio)
                # Setting the image to the QLabel
                QLabel.setPixmap(self.qp)
                # Adding the QLabel to the QVBoxLayout
                vBox.addWidget(QLabel)
                # Adding the QVBoxLayout to the GridLayout
                self.data_display.addLayout(vBox, *position)   
            
            # Put the GridLayout in the Frame
            self.display_frame.setLayout(self.data_display)           
                
        
	# Funtion to upload files for data set 
    def selectToTest(self):
        #accessing the file path to pictures
        files,_ = QtWidgets.QFileDialog.getOpenFileNames(None, 'Open file', 'c:\\',"Image files (*.jpg *.png)")

        if files: 
            # Complete this shit later 
            self.selected_files = files
            for file in self.selected_files:
                print(file)

    # Function that opens another window 
    def startTraining(self):
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_training_session()
        self.ui.setupUi(self.window)
        self.window.show()

    def openWebcam(self):
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_training_session()
        self.ui.setupUi(self.window)
        self.window.show()

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
        self.data_file_button_1.setText(_translate("MainWindow", "Select Dataset"))
        self.data_file_button_2.setText(_translate("MainWindow", "Open Webcam"))
        self.test_button_title.setText(_translate("MainWindow", "Select Method to Test"))
        self.data_image_title_2.setText(_translate("MainWindow", "Data Set:"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.test), _translate("MainWindow", "  Test  "))
        self.actionsave.setText(_translate("MainWindow", "save"))
        self.actionsave.setStatusTip(_translate("MainWindow", "Save the current file"))
        self.actionsave.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.actionopen.setText(_translate("MainWindow", "open"))
        self.actionopen.setStatusTip(_translate("MainWindow", "Open a file"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
    
