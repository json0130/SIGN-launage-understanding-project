# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ASL_Training.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_training_session(object):
    def setupUi(self, training_session):
        training_session.setObjectName("training_session")
        training_session.resize(600, 800)
        training_session.setMinimumSize(QtCore.QSize(600, 800))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(77, 77, 77))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(77, 77, 77))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(77, 77, 77))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(77, 77, 77))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(77, 77, 77))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(77, 77, 77))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        training_session.setPalette(palette)
        self.centralwidget = QtWidgets.QWidget(training_session)
        self.centralwidget.setObjectName("centralwidget")
        self.view_graph_1 = QtWidgets.QWidget(self.centralwidget)
        self.view_graph_1.setGeometry(QtCore.QRect(30, 20, 531, 250))
        self.view_graph_1.setMinimumSize(QtCore.QSize(0, 250))
        self.view_graph_1.setObjectName("view_graph_1")
        self.view_graph_2 = QtWidgets.QWidget(self.centralwidget)
        self.view_graph_2.setGeometry(QtCore.QRect(30, 280, 531, 250))
        self.view_graph_2.setMinimumSize(QtCore.QSize(0, 250))
        self.view_graph_2.setObjectName("view_graph_2")
        self.options_frame_4 = QtWidgets.QFrame(self.centralwidget)
        self.options_frame_4.setGeometry(QtCore.QRect(30, 560, 531, 161))
        self.options_frame_4.setStyleSheet("QFrame {\n"
"    background-color: #333333;\n"
"}")
        self.options_frame_4.setFrameShape(QtWidgets.QFrame.Panel)
        self.options_frame_4.setFrameShadow(QtWidgets.QFrame.Plain)
        self.options_frame_4.setObjectName("options_frame_4")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.options_frame_4)
        self.gridLayout_5.setObjectName("gridLayout_5")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_5.addItem(spacerItem, 1, 1, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_5.addItem(spacerItem1, 1, 4, 1, 1)
        self.progressBar_2 = QtWidgets.QProgressBar(self.options_frame_4)
        self.progressBar_2.setStyleSheet("QProgressBar {\n"
"    color: white;\n"
"}")
        self.progressBar_2.setProperty("value", 24)
        self.progressBar_2.setObjectName("progressBar_2")
        self.gridLayout_5.addWidget(self.progressBar_2, 1, 3, 1, 1)
        self.pushButton_4 = QtWidgets.QPushButton(self.options_frame_4)
        self.pushButton_4.setMinimumSize(QtCore.QSize(0, 40))
        self.pushButton_4.setMaximumSize(QtCore.QSize(16777215, 60))
        self.pushButton_4.setStyleSheet("QPushButton {\n"
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
        self.pushButton_4.setObjectName("pushButton_4")
        self.gridLayout_5.addWidget(self.pushButton_4, 3, 3, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_5.addItem(spacerItem2, 4, 3, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.options_frame_4)
        self.label_6.setMinimumSize(QtCore.QSize(0, 30))
        self.label_6.setStyleSheet("QLabel {\n"
"    color: white;\n"
"}")
        self.label_6.setObjectName("label_6")
        self.gridLayout_5.addWidget(self.label_6, 0, 3, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_5.addItem(spacerItem3, 2, 3, 1, 1)
        training_session.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(training_session)
        self.statusbar.setObjectName("statusbar")
        training_session.setStatusBar(self.statusbar)

        self.retranslateUi(training_session)
        QtCore.QMetaObject.connectSlotsByName(training_session)

    def retranslateUi(self, training_session):
        _translate = QtCore.QCoreApplication.translate
        training_session.setWindowTitle(_translate("training_session", "TrainingSession"))
        self.pushButton_4.setText(_translate("training_session", "Stop Training"))
        self.label_6.setText(_translate("training_session", "                                         Training Progress"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    training_session = QtWidgets.QMainWindow()
    ui = Ui_training_session()
    ui.setupUi(training_session)
    training_session.show()
    sys.exit(app.exec_())
