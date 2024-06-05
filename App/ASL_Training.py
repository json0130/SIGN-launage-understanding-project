from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QGridLayout, QSpacerItem, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

class Ui_training_session(object):
    def setupUi(self, training_session, data):
        training_session.resize(800, 600)
        training_session.setMinimumSize(QtCore.QSize(800, 600))
        training_session.setStyleSheet("QMainWindow {background: #4d4d4d;}\n")
        self.centralwidget = QtWidgets.QWidget(training_session)
        training_session.setCentralWidget(self.centralwidget)

        # Create the main layout
        self.main_layout = QVBoxLayout(self.centralwidget)

        # Add a graph view layout
        self.view_graph = QVBoxLayout()
        self.main_layout.addLayout(self.view_graph)

        # Create a figure with two subplots side by side
        self.figure, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Create a canvas and toolbar
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, training_session)

        # Add toolbar and canvas to the layout
        self.view_graph.addWidget(self.toolbar)
        self.view_graph.addWidget(self.canvas)

        # Set titles for the subplots
        self.ax1.set_title("Accuracy")
        self.ax2.set_title("Loss")

        # Create a frame for options
        self.options_frame = QtWidgets.QFrame(self.centralwidget)
        self.options_frame.setStyleSheet("QFrame {background-color: #333333;}")
        self.options_frame.setFrameShape(QtWidgets.QFrame.Panel)
        self.options_frame.setFrameShadow(QtWidgets.QFrame.Plain)
        self.main_layout.addWidget(self.options_frame)

        # Create a grid layout for the options frame
        self.gridLayout = QGridLayout(self.options_frame)

        # Create a progress bar
        self.progressBar = QtWidgets.QProgressBar(self.options_frame)
        self.progressBar.setStyleSheet("QProgressBar {color: white;}")
        self.progressBar.setProperty("value", 24)
        self.gridLayout.addWidget(self.progressBar, 1, 3, 1, 1)

        # Create a push button
        self.pushButton_4 = QtWidgets.QPushButton(self.options_frame)
        self.pushButton_4.setMinimumSize(QtCore.QSize(0, 40))
        self.pushButton_4.setMaximumSize(QtCore.QSize(16777215, 60))
        self.pushButton_4.setStyleSheet("QPushButton {background-color: #345CC1;color: white; border: none; padding: 8px 16px;}\n"
                                        "QPushButton:hover {background-color: #2A4BA0;}\n"
                                        "QPushButton:pressed {background-color: #1E3C8C;}")
        self.gridLayout.addWidget(self.pushButton_4, 3, 3, 1, 1)
        
        # Add spacer items
        spacerItem = QSpacerItem(40, 20, QSizePolicy.Fixed, QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 1, 1, 1, 1)
        spacerItem1 = QSpacerItem(40, 20, QSizePolicy.Fixed, QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 1, 4, 1, 1)
        spacerItem2 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.gridLayout.addItem(spacerItem2, 4, 3, 1, 1)

        # Create a label
        self.label = QtWidgets.QLabel(self.options_frame)
        self.label.setMinimumSize(QtCore.QSize(0, 30))
        self.label.setStyleSheet("QLabel {color: white;}")
        self.gridLayout.addWidget(self.label, 0, 3, 1, 1)
        spacerItem3 = QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.gridLayout.addItem(spacerItem3, 2, 3, 1, 1)
        
        training_session.setStatusBar(QtWidgets.QStatusBar(training_session))

        self.retranslateUi(training_session)
        QtCore.QMetaObject.connectSlotsByName(training_session)

    def retranslateUi(self, training_session):
        _translate = QtCore.QCoreApplication.translate
        training_session.setWindowTitle(_translate("training_session", "TrainingSession"))
        self.pushButton_4.setText(_translate("training_session", "Stop Training"))
        self.label.setText(_translate("training_session", "Training Progress"))

    def plot(self, data1, data2):
        # Add code here to plot the data on ax1 and ax2
        self.ax1.plot(data1)
        self.ax2.plot(data2)
        self.canvas.draw()

    def update_plot(self, train_losses, val_accuracies, epoch):
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()

        # Create x-axis values starting from 1
        epochs = list(range(1, epoch + 2))

        # Update the plots with new data
        self.ax1.plot(epochs, train_losses, label='Training Loss')
        self.ax1.set_title(f'Training Loss - Epoch {epoch + 1}')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.legend()

        self.ax2.plot(epochs, val_accuracies, label='Validation Accuracy')
        self.ax2.set_title(f'Validation Accuracy - Epoch {epoch + 1}')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Accuracy')
        self.ax2.legend()

        # Redraw the canvas
        self.canvas.draw()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    training_session = QtWidgets.QMainWindow()
    ui = Ui_training_session()
    ui.setupUi(training_session, None)
    training_session.show()
    sys.exit(app.exec_())
