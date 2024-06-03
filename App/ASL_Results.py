from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import numpy as np

class Ui_Results(object):

    def setupUi(self, MainWindow, data):
        MainWindow.resize(600, 800)
        MainWindow.setMinimumSize(QtCore.QSize(600, 800))
        MainWindow.setStyleSheet("QMainWindow {background: #4d4d4d;}\n")
        
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        MainWindow.setCentralWidget(self.centralwidget)
        
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(30, 20, 100, 16))
        self.label.setStyleSheet("QLabel {color: white; font-weight: bold;}")
        self.label.setText("Results:")
        
        self.layout = QVBoxLayout(self.centralwidget)
        self.layout.setGeometry(QtCore.QRect(30, 50, 540, 700))
        
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, MainWindow)

        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)

        self.plot(data)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "ASL Results"))

    def plot(self, data):
        self.figure.clear()
        image_data, bar_data = data

        ax1 = self.figure.add_subplot(211)
        ax1.imshow(image_data, cmap='viridis')
        ax1.set_title('Image')

        ax2 = self.figure.add_subplot(212)
        categories, values = bar_data
        ax2.bar(categories, values, color='blue')
        ax2.set_title('Bar Graph')
        ax2.set_ylabel('Values')

        self.canvas.draw()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, data, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.ui = Ui_Results()
        self.ui.setupUi(self, data)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    
    # Sample data for the plots
    image_data = np.random.random((10, 10))
    bar_data = (['Category 1', 'Category 2', 'Category 3'], [10, 20, 15])
    data = (image_data, bar_data)
    
    MainWindow = MainWindow(data)
    MainWindow.show()
    
    sys.exit(app.exec_())
