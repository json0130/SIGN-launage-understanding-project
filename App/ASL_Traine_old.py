# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ASL_Trainer.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from ASL_Training import Ui_training_session

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(600, 800)
        MainWindow.setMinimumSize(QtCore.QSize(600, 800))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 600, 800))
        self.tabWidget.setMinimumSize(QtCore.QSize(600, 800))
        self.tabWidget.setMaximumSize(QtCore.QSize(600, 800))
        self.tabWidget.setAutoFillBackground(False)
        self.tabWidget.setStyleSheet("QTabWidget::pane {\n"
"    border: 1px solid #1F1F1F;\n"
"    background: #1F1F1F; /* Background color of the tab widget */\n"
"}\n"
"\n"
"QTabBar::tab {\n"
"    background: #333333; /* Background color of each tab */\n"
"    color: #ffffff; /* Text color of each tab */\n"
"    padding: 10px;\n"
"}\n"
"\n"
"QTabBar::tab:selected {\n"
"    background: #1F1F1F; /* Background color of the selected tab */\n"
"    color: #FFFFFF; /* Text color of the selected tab */\n"
"}")
        self.tabWidget.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.tabWidget.setDocumentMode(False)
        self.tabWidget.setObjectName("tabWidget")
        self.load = QtWidgets.QWidget()
        self.load.setObjectName("load")
        self.Data_butto_frame = QtWidgets.QFrame(self.load)
        self.Data_butto_frame.setGeometry(QtCore.QRect(30, 50, 540, 60))
        self.Data_butto_frame.setMinimumSize(QtCore.QSize(540, 0))
        self.Data_butto_frame.setStyleSheet("QFrame {\n"
"    background-color: #333333;\n"
"}")
        self.Data_butto_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.Data_butto_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.Data_butto_frame.setObjectName("Data_butto_frame")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.Data_butto_frame)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem = QtWidgets.QSpacerItem(191, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem)
        self.data_file_button = QtWidgets.QPushButton(self.Data_butto_frame, clicked= lambda: self.uploadFiles())
        self.data_file_button.setMinimumSize(QtCore.QSize(120, 40))
        self.data_file_button.setBaseSize(QtCore.QSize(120, 40))
        self.data_file_button.setStyleSheet("QPushButton {\n"
"    background-color: #345CC1;\n"
"    color: white; /* Set text color to white for better contrast */\n"
"    border: none;\n"
"    padding: 8px 16px; /* Optional: Add some padding for better appearance */\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: #2A4BA0; /* Optional: Slightly darker color on hover */\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    background-color: #1E3C8C; /* Optional: Even darker color when pressed */\n"
"}")
        self.data_file_button.setObjectName("data_file_button")
        self.horizontalLayout_3.addWidget(self.data_file_button)
        spacerItem1 = QtWidgets.QSpacerItem(191, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem1)
        self.data_image_title = QtWidgets.QLabel(self.load)
        self.data_image_title.setGeometry(QtCore.QRect(30, 140, 91, 16))
        self.data_image_title.setStyleSheet("QLabel {\n"
"    color: white;\n"
"    font-weight: bold;\n"
"}\n"
"")
        self.data_image_title.setObjectName("data_image_title")
        self.frame_3 = QtWidgets.QFrame(self.load)
        self.frame_3.setGeometry(QtCore.QRect(30, 160, 541, 551))
        self.frame_3.setMinimumSize(QtCore.QSize(540, 0))
        self.frame_3.setStyleSheet("QFrame {\n"
"    background-color: #333333;\n"
"}")
        self.frame_3.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_3.setObjectName("frame_3")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_3)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.widget_2 = QtWidgets.QWidget(self.frame_3)
        self.widget_2.setObjectName("widget_2")
        self.horizontalLayout_2.addWidget(self.widget_2)
        self.data_scroll = QtWidgets.QScrollBar(self.frame_3)
        self.data_scroll.setStyleSheet("QScrollBar:vertical {\n"
"    background: #333333;\n"
"    width: 15px;\n"
"    margin: 22px 0 22px 0;\n"
"}\n"
"\n"
"QScrollBar::handle:vertical {\n"
"    background: #545454; \n"
"    min-height: 20px;\n"
"    border-radius: 5px;\n"
"}\n"
"\n"
"QScrollBar::add-line:vertical {\n"
"    background: #333333;\n"
"    height: 20px;\n"
"    subcontrol-position: bottom;\n"
"    subcontrol-origin: margin;\n"
"}\n"
"\n"
"QScrollBar::sub-line:vertical {\n"
"    background: #333333;\n"
"    height: 20px;\n"
"    subcontrol-position: top;\n"
"    subcontrol-origin: margin;\n"
"}\n"
"\n"
"QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {\n"
"    width: 3px;\n"
"    height: 3px;\n"
"    background: white;\n"
"}\n"
"\n"
"QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {\n"
"    background: none;\n"
"}")
        self.data_scroll.setOrientation(QtCore.Qt.Vertical)
        self.data_scroll.setObjectName("data_scroll")
        self.horizontalLayout_2.addWidget(self.data_scroll)
        self.data_button_title = QtWidgets.QLabel(self.load)
        self.data_button_title.setGeometry(QtCore.QRect(30, 30, 181, 16))
        self.data_button_title.setStyleSheet("QLabel {\n"
"    color: white;\n"
"    font-weight: bold;\n"
"}\n"
"")
        self.data_button_title.setObjectName("data_button_title")
        self.data_line = QtWidgets.QFrame(self.load)
        self.data_line.setGeometry(QtCore.QRect(30, 120, 540, 20))
        self.data_line.setMinimumSize(QtCore.QSize(540, 0))
        self.data_line.setFrameShape(QtWidgets.QFrame.HLine)
        self.data_line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.data_line.setObjectName("data_line")
        self.tabWidget.addTab(self.load, "")
        self.train = QtWidgets.QWidget()
        self.train.setObjectName("train")
        self.train_options_frame = QtWidgets.QFrame(self.train)
        self.train_options_frame.setGeometry(QtCore.QRect(30, 260, 531, 451))
        self.train_options_frame.setStyleSheet("QFrame {\n"
"    background-color: #333333;\n"
"}")
        self.train_options_frame.setFrameShape(QtWidgets.QFrame.Panel)
        self.train_options_frame.setFrameShadow(QtWidgets.QFrame.Plain)
        self.train_options_frame.setObjectName("train_options_frame")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.train_options_frame)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.train_batch = QtWidgets.QSpinBox(self.train_options_frame)
        self.train_batch.setMaximum(500)
        self.train_batch.setSingleStep(10)
        self.train_batch.setObjectName("train_batch")
        self.gridLayout_2.addWidget(self.train_batch, 7, 1, 1, 1)
        self.train_batch_label = QtWidgets.QLabel(self.train_options_frame)
        self.train_batch_label.setStyleSheet("QLabel {\n"
"    color: white;\n"
"}")
        self.train_batch_label.setObjectName("train_batch_label")
        self.gridLayout_2.addWidget(self.train_batch_label, 8, 1, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem2, 11, 1, 1, 1)
        self.train_slider = QtWidgets.QSlider(self.train_options_frame)
        self.train_slider.setStyleSheet("QSlider::groove:horizontal {\n"
"    height: 4px;\n"
"    background: #F0F0F0;\n"
"    margin: 2px 0;\n"
"    border-radius: 4px;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"    background: #345CC1;\n"
"    width: 24px;\n"
"    height: 18px;\n"
"    margin: -5px 0; /* handle is placed by default on the contents rect of the groove, margin is for collision */\n"
"}\n"
"\n"
"QSlider::handle:horizontal:hover {\n"
"    background: #2A4BA0;\n"
"}\n"
"\n"
"QSlider::handle:horizontal:pressed {\n"
"    background: #1E3C8C;\n"
"}")
        self.train_slider.setMaximum(99)
        self.train_slider.setOrientation(QtCore.Qt.Horizontal)
        self.train_slider.setObjectName("train_slider")
        self.gridLayout_2.addWidget(self.train_slider, 1, 1, 1, 1)
        self.train_epoch = QtWidgets.QSpinBox(self.train_options_frame)
        self.train_epoch.setMaximum(60)
        self.train_epoch.setSingleStep(5)
        self.train_epoch.setObjectName("train_epoch")
        self.gridLayout_2.addWidget(self.train_epoch, 4, 1, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem3, 1, 0, 1, 1)
        self.train_start_button = QtWidgets.QPushButton(self.train_options_frame, clicked= lambda: self.startTraining())
        self.train_start_button.setMinimumSize(QtCore.QSize(0, 60))
        self.train_start_button.setMaximumSize(QtCore.QSize(16777215, 60))
        self.train_start_button.setStyleSheet("QPushButton {\n"
"    background-color: #345CC1;\n"
"    color: white; /* Set text color to white for better contrast */\n"
"    border: none;\n"
"    padding: 8px 16px; /* Optional: Add some padding for better appearance */\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: #2A4BA0; /* Optional: Slightly darker color on hover */\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    background-color: #1E3C8C; /* Optional: Even darker color when pressed */\n"
"}\n"
"")
        self.train_start_button.setObjectName("train_start_button")
        self.gridLayout_2.addWidget(self.train_start_button, 10, 1, 1, 1)
        self.epoch_label = QtWidgets.QLabel(self.train_options_frame)
        self.epoch_label.setStyleSheet("QLabel {\n"
"    color: white;\n"
"}")
        self.epoch_label.setObjectName("epoch_label")
        self.gridLayout_2.addWidget(self.epoch_label, 5, 1, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem4, 1, 2, 1, 1)
        spacerItem5 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem5, 9, 1, 1, 1)
        self.train_batch_title = QtWidgets.QLabel(self.train_options_frame)
        self.train_batch_title.setMinimumSize(QtCore.QSize(0, 50))
        self.train_batch_title.setMaximumSize(QtCore.QSize(16777215, 30))
        self.train_batch_title.setStyleSheet("QLabel {\n"
"    color: white;\n"
"    font-weight: bold;\n"
"}")
        self.train_batch_title.setObjectName("train_batch_title")
        self.gridLayout_2.addWidget(self.train_batch_title, 6, 1, 1, 1)
        self.train_epoch_title = QtWidgets.QLabel(self.train_options_frame)
        self.train_epoch_title.setMinimumSize(QtCore.QSize(0, 50))
        self.train_epoch_title.setMaximumSize(QtCore.QSize(16777215, 30))
        self.train_epoch_title.setStyleSheet("QLabel {\n"
"    color: white;\n"
"    font-weight: bold;\n"
"}")
        self.train_epoch_title.setObjectName("train_epoch_title")
        self.gridLayout_2.addWidget(self.train_epoch_title, 3, 1, 1, 1)
        self.train_ratio_title = QtWidgets.QLabel(self.train_options_frame)
        self.train_ratio_title.setMinimumSize(QtCore.QSize(0, 40))
        self.train_ratio_title.setMaximumSize(QtCore.QSize(16777215, 30))
        self.train_ratio_title.setStyleSheet("QLabel {\n"
"    color: white;\n"
"    font-weight: bold;\n"
"}\n"
"")
        self.train_ratio_title.setObjectName("train_ratio_title")
        self.gridLayout_2.addWidget(self.train_ratio_title, 0, 1, 1, 1)
        self.train_ratio_label = QtWidgets.QLabel(self.train_options_frame)
        self.train_ratio_label.setStyleSheet("QLabel {\n"
"    color: white;\n"
"}")
        self.train_ratio_label.setObjectName("train_ratio_label")
        self.gridLayout_2.addWidget(self.train_ratio_label, 2, 1, 1, 1)
        self.train_model_frame = QtWidgets.QFrame(self.train)
        self.train_model_frame.setGeometry(QtCore.QRect(30, 50, 528, 101))
        self.train_model_frame.setStyleSheet("QFrame {\n"
"    background-color: #333333;\n"
"}")
        self.train_model_frame.setFrameShape(QtWidgets.QFrame.Panel)
        self.train_model_frame.setFrameShadow(QtWidgets.QFrame.Plain)
        self.train_model_frame.setObjectName("train_model_frame")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.train_model_frame)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.train_combobox = QtWidgets.QComboBox(self.train_model_frame)
        self.train_combobox.setMinimumSize(QtCore.QSize(60, 30))
        self.train_combobox.setObjectName("train_combobox")
        self.gridLayout_3.addWidget(self.train_combobox, 2, 1, 1, 1)
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem6, 2, 7, 1, 1)
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem7, 2, 0, 1, 1)
        self.train_model_title = QtWidgets.QLabel(self.train_model_frame)
        self.train_model_title.setMinimumSize(QtCore.QSize(0, 30))
        self.train_model_title.setStyleSheet("QLabel {\n"
"    color: white;\n"
"    font-weight: bold;\n"
"}")
        self.train_model_title.setObjectName("train_model_title")
        self.gridLayout_3.addWidget(self.train_model_title, 0, 1, 1, 1)
        spacerItem8 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_3.addItem(spacerItem8, 3, 1, 1, 1)
        self.train_line_bottom = QtWidgets.QFrame(self.train)
        self.train_line_bottom.setGeometry(QtCore.QRect(30, 230, 531, 16))
        self.train_line_bottom.setFrameShape(QtWidgets.QFrame.HLine)
        self.train_line_bottom.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.train_line_bottom.setObjectName("train_line_bottom")
        self.train_line_top = QtWidgets.QFrame(self.train)
        self.train_line_top.setGeometry(QtCore.QRect(30, 160, 531, 16))
        self.train_line_top.setFrameShape(QtWidgets.QFrame.HLine)
        self.train_line_top.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.train_line_top.setObjectName("train_line_top")
        self.tabWidget.addTab(self.train, "")
        self.test = QtWidgets.QWidget()
        self.test.setObjectName("test")
        self.result_widget = QtWidgets.QWidget(self.test)
        self.result_widget.setGeometry(QtCore.QRect(30, 16, 540, 551))
        self.result_widget.setMinimumSize(QtCore.QSize(540, 0))
        self.result_widget.setObjectName("result_widget")
        self.test_button_frame = QtWidgets.QFrame(self.test)
        self.test_button_frame.setGeometry(QtCore.QRect(30, 50, 540, 60))
        self.test_button_frame.setMinimumSize(QtCore.QSize(540, 0))
        self.test_button_frame.setStyleSheet("QFrame {\n"
"    background-color: #333333;\n"
"}")
        self.test_button_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.test_button_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.test_button_frame.setObjectName("test_button_frame")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.test_button_frame)
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem9 = QtWidgets.QSpacerItem(187, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem9)
        self.data_file_button_2 = QtWidgets.QPushButton(self.test_button_frame, clicked= lambda: self.selectToTest())
        self.data_file_button_2.setMinimumSize(QtCore.QSize(120, 40))
        self.data_file_button_2.setBaseSize(QtCore.QSize(120, 40))
        self.data_file_button_2.setStyleSheet("QPushButton {\n"
"    background-color: #345CC1;\n"
"    color: white; /* Set text color to white for better contrast */\n"
"    border: none;\n"
"    padding: 8px 16px; /* Optional: Add some padding for better appearance */\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: #2A4BA0; /* Optional: Slightly darker color on hover */\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    background-color: #1E3C8C; /* Optional: Even darker color when pressed */\n"
"}")
        self.data_file_button_2.setObjectName("data_file_button_2")
        self.horizontalLayout.addWidget(self.data_file_button_2)
        spacerItem10 = QtWidgets.QSpacerItem(191, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem10)
        self.test_result_title = QtWidgets.QLabel(self.test)
        self.test_result_title.setGeometry(QtCore.QRect(30, 140, 91, 16))
        self.test_result_title.setStyleSheet("QLabel {\n"
"    color: white;\n"
"    font-weight: bold;\n"
"}\n"
"")
        self.test_result_title.setObjectName("test_result_title")
        self.test_line = QtWidgets.QFrame(self.test)
        self.test_line.setGeometry(QtCore.QRect(30, 120, 540, 20))
        self.test_line.setMinimumSize(QtCore.QSize(540, 0))
        self.test_line.setFrameShape(QtWidgets.QFrame.HLine)
        self.test_line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.test_line.setObjectName("test_line")
        self.test_button_title = QtWidgets.QLabel(self.test)
        self.test_button_title.setGeometry(QtCore.QRect(30, 30, 141, 16))
        self.test_button_title.setStyleSheet("QLabel {\n"
"    color: white;\n"
"    font-weight: bold;\n"
"}\n"
"")
        self.test_button_title.setObjectName("test_button_title")
        self.tabWidget.addTab(self.test, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionsave = QtWidgets.QAction(MainWindow)
        self.actionsave.setObjectName("actionsave")
        self.actionopen = QtWidgets.QAction(MainWindow)
        self.actionopen.setObjectName("actionopen")

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    # Function to upload files for data set
    def uploadFiles(self):
        #accessing the file path to pictures
        files,_ = QtWidgets.QFileDialog.getOpenFileNames(None, 'Open file', 'c:\\',"Image files (*.csv)")

        if files:
            pass
        
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

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "ASL TRAINER"))
        self.data_file_button.setText(_translate("MainWindow", "Upload Dataset"))
        self.data_image_title.setText(_translate("MainWindow", "Data Set:"))
        self.data_button_title.setText(_translate("MainWindow", "Select Images For Data Set"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.load), _translate("MainWindow", "  Data  "))
        self.train_batch_label.setText(_translate("MainWindow", "0                                                250                                            500"))
        self.train_start_button.setText(_translate("MainWindow", "Start Training"))
        self.epoch_label.setText(_translate("MainWindow", "0                                                 30                                              60"))
        self.train_batch_title.setText(_translate("MainWindow", "Batch Size"))
        self.train_epoch_title.setText(_translate("MainWindow", "Epochs"))
        self.train_ratio_title.setText(_translate("MainWindow", "Train/Test Ratio"))
        self.train_ratio_label.setText(_translate("MainWindow", "0                                                 50                                              99"))
        self.train_model_title.setText(_translate("MainWindow", "                                           Select Model"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.train), _translate("MainWindow", "  Train  "))
        self.data_file_button_2.setText(_translate("MainWindow", "Upload Images"))
        self.test_result_title.setText(_translate("MainWindow", "Test Results:"))
        self.test_button_title.setText(_translate("MainWindow", "Select Method To Test"))
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
