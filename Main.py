#This is the main function that opens the application
import sys

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QToolBar, QAction, QStatusBar, QCheckBox, qApp

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Handwritten Digit Recognizer")
        label = QLabel("Welcome!")
        label.setAlignment(Qt.AlignCenter)

        self.setFixedSize(QSize(600, 500))

        # Set the central widget of the Window.
        self.setCentralWidget(label)
        
        view_action = QAction("View training", self)
        view_action.setStatusTip("This will show you training images for the AI")
        view_action.triggered.connect(self.onMyToolBarButtonClick)

        quit_action = QAction("Quit", self)
        quit_action.setShortcut('ESC')
        quit_action.setStatusTip("Quit Application")
        quit_action.triggered.connect(qApp.quit)

        self.setStatusBar(QStatusBar(self))

        menu = self.menuBar()

        file_menu = menu.addMenu("&File")
        file_menu.addAction(view_action)
        file_menu.addSeparator()
        file_menu.addAction(quit_action)

    def onMyToolBarButtonClick(self, s):
        print("click", s)


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()