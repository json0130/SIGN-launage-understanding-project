from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

class Ui_training_session(object):
    def setupUi(self, training_session, data):
        training_session.resize(600, 800)
        training_session.setMinimumSize(QtCore.QSize(600, 840))
        training_session.setStyleSheet("QMainWindow {background: #4d4d4d;}\n")
        self.centralwidget = QtWidgets.QWidget(training_session)
        training_session.setCentralWidget(self.centralwidget)

        # Add a graph view
        self.view_graph = QVBoxLayout(self.centralwidget)
        
        # Create a figure and canvas
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        
        # Create a toolbar and add it to the layout
        self.toolbar = NavigationToolbar(self.canvas, training_session)
        self.view_graph.addWidget(self.toolbar)
        self.view_graph.addWidget(self.canvas)

        self.ax1 = self.figure.add_subplot(211)
        self.ax2 = self.figure.add_subplot(212)
        
        # Create a frame for options
        self.options_frame = QtWidgets.QFrame(self.centralwidget)
        self.options_frame.setGeometry(QtCore.QRect(30, 570, 531, 161))
        self.options_frame.setStyleSheet("QFrame {background-color: #333333;}")
        self.options_frame.setFrameShape(QtWidgets.QFrame.Panel)
        self.options_frame.setFrameShadow(QtWidgets.QFrame.Plain)
        # Create a grid layout for the options frame
        self.gridLayout = QtWidgets.QGridLayout(self.options_frame)

        # Create a progress bar
        self.progressBar = QtWidgets.QProgressBar(self.options_frame)
        self.progressBar.setStyleSheet("QProgressBar {color: white;}")
        self.progressBar.setProperty("value", 24)
        # Add the progress bar to the grid layout
        self.gridLayout.addWidget(self.progressBar, 1, 3, 1, 1)

        # Create a Push Button
        self.pushButton_4 = QtWidgets.QPushButton(self.options_frame)
        self.pushButton_4.setMinimumSize(QtCore.QSize(0, 40))
        self.pushButton_4.setMaximumSize(QtCore.QSize(16777215, 60))
        self.pushButton_4.setStyleSheet("QPushButton {background-color: #345CC1;color: white; border: none; padding: 8px 16px;}\n"
                                        "QPushButton:hover {background-color: #2A4BA0;}\n"
                                        "QPushButton:pressed {background-color: #1E3C8C;}")
        # Add the push button to the grid layout
        self.gridLayout.addWidget(self.pushButton_4, 3, 3, 1, 1)
        
        # Add a spacer items
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 1, 1, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 1, 4, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout.addItem(spacerItem2, 4, 3, 1, 1)

        # Create a label
        self.label = QtWidgets.QLabel(self.options_frame)
        self.label.setMinimumSize(QtCore.QSize(0, 30))
        self.label.setStyleSheet("QLabel {color: white;}")
        self.gridLayout.addWidget(self.label, 0, 3, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout.addItem(spacerItem3, 2, 3, 1, 1)
        
        training_session.setStatusBar(QtWidgets.QStatusBar(training_session))

        self.retranslateUi(training_session)
        QtCore.QMetaObject.connectSlotsByName(training_session)

    def retranslateUi(self, training_session):
        _translate = QtCore.QCoreApplication.translate
        training_session.setWindowTitle(_translate("training_session", "TrainingSession"))
        self.pushButton_4.setText(_translate("training_session", "Stop Training"))
        self.label.setText(_translate("training_session", "                                         Training Progress"))

    def plot(self, data1, data2):
        # add code here to plot the data
        self.canvas.draw()

    def update_plot(self, data1, data2):
        # add code here to update the plot with new data
        self.plot(data1, data2)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    training_session = QtWidgets.QMainWindow()
    ui = Ui_training_session()
    ui.setupUi(training_session, None)
    training_session.show()
    sys.exit(app.exec_())
