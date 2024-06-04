from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QLabel

class ClickableLabel(QLabel):
    clicked = pyqtSignal(str)  # Custom signal with one string argument

    def __init__(self, image_path, parent=None):
        super(ClickableLabel, self).__init__(parent)
        self.image_path = image_path  # Store the image path

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.image_path)  # Emit the image path
        super(ClickableLabel, self).mousePressEvent(event)

class ClickLabel(QLabel):
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super(ClickLabel, self).__init__(parent)

    def mousePressEvent(self, event):
        self.clicked.emit()
        super(ClickLabel, self).mousePressEvent(event)