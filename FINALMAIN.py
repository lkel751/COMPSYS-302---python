# Necessary imports to complete the project

import sys
import time
import zipfile
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLineEdit, QHBoxLayout, QVBoxLayout, QMainWindow, QSlider, QLabel, QStatusBar,qApp
from PyQt5.QtCore import QBasicTimer, QSize
from PIL import Image
import cv2
import numpy as np
from numpy.typing import _128Bit
import pandas as pd
from keras.utils import np_utils
from keras.datasets import mnist
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import visualkeras as vk # pip install visualkeras
import pandas as pd
import seaborn as sn
from keras.models import Sequential
from matplotlib import pyplot as plt
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from PyQt5.QtWidgets import QMainWindow, QApplication, QAction, QFileDialog, QPushButton
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QImage, QPainter, QPen, QPixmap
import cv2 as cv
import tensorflow as tf
import keras
import random

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.preprocessing.image import ImageDataGenerator

# Global variables (default values to initialize them)
test_data = 1
test_labels = ""
train_data = 1
train_labels = ""
d = 0
BATCH = 5
model = ""
letter_x = 1
letter_y = ""
digit_train_x = 1
digit_train_y = ""
isDigit = 2
#--------------------------------------------------------------------------------------------------------------TRAIN WINDOW---------------------------------------------------------------------------------------------------------------------------------------
#Train window, here you download EMNIST, and train a model, also can cancel
class TrainWindow(QWidget):
    def __init__(self):
        super(TrainWindow, self).__init__()
        self.setFixedSize(QSize(660, 483))
        self.setWindowTitle("Training Window!")
        self.initUI()

#Initialise widgets, including slider for epoch size and download bar
    def initUI(self):
        global d
        self.label = QtWidgets.QLabel(self)
        self.label.setText("Here you can download the dataset and train a model!")
        self.label.setGeometry(QtCore.QRect(50, 10, 661, 91))
      
#Frame which contains most widgets
        self.frame = QtWidgets.QFrame(self)
        self.frame.setGeometry(QtCore.QRect(0, 100, 661, 301))
        self.frame.setFrameShape(QtWidgets.QFrame.Box)
        self.frame.setFrameShadow(QtWidgets.QFrame.Plain)

        self.progressLabel = QtWidgets.QLabel(self.frame)
        self.progressLabel.setGeometry(QtCore.QRect(100, 0, 661, 81))

#Download button brings you to the download method
        self.Download = QtWidgets.QPushButton(self)
        self.Download.setText("Download")
        self.Download.setGeometry(QtCore.QRect(40, 430, 141, 41))
        self.Download.clicked.connect(self.download)

#Train button brings you to the train method
        self.Train = QtWidgets.QPushButton(self)
        self.Train.setText("Train Model")
        self.Train.setGeometry(QtCore.QRect(250, 430, 141, 41))
        self.Train.clicked.connect(self.train)

#Cancel button cancels and exits
        self.Cancel = QtWidgets.QPushButton(self)
        self.Cancel.setText("Cancel")
        self.Cancel.setGeometry(QtCore.QRect(450, 430, 141, 41))
        self.Cancel.clicked.connect(self.close)

        self.Slider = QtWidgets.QSlider(self.frame)
        self.Slider.setGeometry(QtCore.QRect(10, 240, 641, 51))
        self.Slider.setProperty("value", 0)
        self.Slider.setOrientation(QtCore.Qt.Horizontal)
        self.Slider.setMinimum(0)
        self.Slider.setMaximum(100)
        self.Slider.setValue(0)
        self.Slider.setTickInterval(10)
        self.Slider.setTickPosition(QSlider.TicksBelow)
        self.Slider.valueChanged.connect(self.slider)
        
        self.slideLabel = QtWidgets.QLabel(self)
        self.slideLabel.setGeometry(QtCore.QRect(310, 320, 35, 41))        

        step = 500
        
        self.progressBar = QtWidgets.QProgressBar(self.frame) 
        self.progressBar.setMinimum(0)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setMaximum(step)
        self.progressBar.setGeometry(QtCore.QRect(7, 142, 651, 51))  

#Slider value determines epoch size for download
    def slider(self, value):
        global BATCH
        self.slideLabel.setText(str(value))
        BATCH = int(value/10 + 1)

#This is the progress bar, calculated to take as long as the download takes
    def run(self, step):      
        for i in range(step):
            time.sleep(0.1)
            self.progressBar.setValue(i+1)   
        self.progressLabel.setText("Download complete! Move slider \n to set sample size for training")

#Here the data from EMNIST is loaded, missing lowercase as too big a file. Used https://www.youtube.com/watch?v=kOF2Lp_GbkQ to assist with this part
    def download(self):
        global test_data, test_labels, train_data, train_labels

        self.progressLabel.setText("Downloading please wait...")

        self.run(2000)
       
        #Get data from online (must have a kaggle account)
        zipFileURL = "https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format/download"
        path = tf.keras.utils.get_file('archive.zip', zipFileURL)
        with zipfile.ZipFile('archive.zip' 'r') as my_zip:
               my_zip.extractall('handwrittendataset.csv')
        data_root = 'handwrittendataset.csv'
        
#        data_root = "C:/Users/liamp/Source/Repos/lkel751/COMPSYS-302---python/dataset/A_Z Handwritten Data.csv"
        dataset = pd.read_csv(data_root).astype("float32")
        dataset.rename(columns={'0': "label"}, inplace=True)

        #Extract data (x) and labels(y) from letter and digits
        letter_x = dataset.drop("label", axis=1)
        letter_y = dataset["label"]
        (digit_train_x, digit_train_y), (digit_test_x, digit_test_y) = mnist.load_data()
        letter_x = letter_x.values

        digit_data = np.concatenate((digit_train_x, digit_test_x))
        digit_target = np.concatenate((digit_train_y, digit_test_y))

        print(digit_data.shape, digit_target.shape)

        digit_target += 26
        data = []

        for flatten in letter_x:
            image = np.reshape(flatten, (28, 28, 1))
            data.append(image)

        letter_data = np.array(data, dtype=np.float32)
        letter_target = letter_y
        digit_data = np.reshape(digit_data, (digit_data.shape[0], digit_data.shape[1], digit_data.shape[2], 1))

        plt.figure(figsize=(20, 20))

        data = np.concatenate((digit_data, letter_data))
        target = np.concatenate((digit_target, letter_target))

        train_data, test_data, train_labels, test_labels = train_test_split(data, target, test_size=0.2)

        print(train_data.shape, train_labels.shape)
        print(test_data.shape, test_labels.shape)

        train_data = train_data / 255.0
        test_data = test_data / 255.0

        train_labels = np_utils.to_categorical(train_labels)
        test_labels = np_utils.to_categorical(test_labels)

        train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1], train_data.shape[2], 1))
        test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1], test_data.shape[2], 1))

        print("download complete!")

#Here model is trained, also used https://www.youtube.com/watch?v=kOF2Lp_GbkQ to assist with this part
    def train(self):
        global model, BATCH

        self.progressLabel.setText("Training model please wait...")
        model = Sequential()
        
        #Create model
        model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation="relu"))
        model.add(BatchNormalization())

        model.add(Conv2D(32, (5, 5), activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.25))

        model.add(BatchNormalization())
        model.add(Flatten())

        model.add(Dense(256, activation="relu"))
        model.add(Dense(36, activation="softmax"))

        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.summary()

        best_loss_checkpoint = ModelCheckpoint(
            filepath="Models/best_loss_model.h5",
                monitor="loss",
                save_best_only=True,
                save_weights_only=True,
                mode="min"
            )

        best_val_loss_checkpoint = ModelCheckpoint(
                filepath="Models/best_val_loss_model.h5",
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
                mode="min"
            )
          
        history = model.fit(
                train_data,
                train_labels,
                validation_data=(test_data, test_labels),  
                epochs = BATCH,
                batch_size=200, 
                callbacks=[best_loss_checkpoint, best_val_loss_checkpoint]
            )



        model.load_weights("Models/best_val_loss_model.h5")

#--------------------------------------------------------------------------------------------------------------VIEW WINDOW---------------------------------------------------------------------------------------------------------------------------------------
#Here you enter a character and press enter or ok to get example pics of said character (0-9, A-Z)
class ViewWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.InitUI()

#initialize components of the viewWindow
    def InitUI(self):
        # GUI 
        self.setWindowTitle('View Window!')
        self.setFixedSize(QSize(560, 462))

        self.scrollArea = QtWidgets.QScrollArea(self)
        self.scrollArea.setGeometry(QtCore.QRect(0, 0, 361, 461))
        self.scrollArea.setWidgetResizable(True)

        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 359, 459))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

#Enter char widget, enter the character and press enter to bring you to the view method.
        self.enterchar = QtWidgets.QLineEdit(self)
        self.enterchar.setGeometry(QtCore.QRect(510, 150, 31, 31))
        self.enterchar.setMaxLength(1)
        self.enterchar.editingFinished.connect(self.view)
        self.charlabel = QtWidgets.QLabel(self)
        self.charlabel.setGeometry(QtCore.QRect(380, 150, 111, 31))
        self.charlabel.setText("Enter Character:")
        font = QtGui.QFont()
        font.setPointSize(8)
        self.charlabel.setFont(font)

        self.errorlabel = QtWidgets.QLabel(self)
        self.errorlabel.setGeometry(QtCore.QRect(380, 20, 200, 111))

        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(QtCore.QRect(390, 350, 161, 101))
        self.label.setText("Welcome! Enter the \ncharacter and select \nmodel to view some \nexample images of it!")

        self.OK = QtWidgets.QPushButton(self)
        self.OK.setGeometry(QtCore.QRect(430, 230, 91, 41))
        self.OK.setText("OK")

        self.toptext = QtWidgets.QLabel(self)
        self.toptext.setGeometry(QtCore.QRect(384, 9, 191, 111))

#When clicking the view button, findchar and display image, otherwise display error message.
    def view(self):
        global isDigit
        if self.enterchar.text().isdigit() or self.enterchar.text().isupper():
            self.char = self.enterchar.text()
            self.findChar()
            self.viewImage = QPixmap('Images/viewimages.png')
            self.labelImage = QLabel()
            self.labelImage.setPixmap(self.viewImage)
            self.update()
        else:
            self.errorlabel.setText("Error: \nPlease enter a valid\n character (0-9, A-Z)")

#This creates an array that has desired character
    def findChar(self):
        global train_data, train_labels
        # show 100 images of the number chosen
        cols = 5
        rows = 10
        
        fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(cols, rows))
        fig.tight_layout()

        if model == "":
                  self.errorlabel.setText("Error: \nPlease Train Model")
                  return
        else:
            for i in range(cols):
                for j in range(rows):
                            x_selected = train_data[train_labels == self.char]    
                            axs[j][i].imshow(x_selected) 
                            axs[j][i].axis("off")

            plt.savefig('Images/viewimages.png')



#--------------------------------------------------------------------------------------------------------------TEST WINDOW---------------------------------------------------------------------------------------------------------------------------------------
#Here you test images byu drawing in the canvas

class TestWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.InitUI()

    def InitUI(self):
        # GUI 
        self.setWindowTitle('TestWindow')
        self.setFixedSize(QSize(697, 471))
#Frame to draw canvas

        self.frame = QtWidgets.QFrame(self)
        self.frame.setGeometry(QtCore.QRect(0, 0, 471, 471))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Plain)

        self.commandWidget = QtWidgets.QWidget(self)
        self.commandWidget.setGeometry(QtCore.QRect(470, 0, 231, 471))

   #     self.Brushframe = QtWidgets.QFrame(self.commandWidget)
   #     self.Brushframe.setGeometry(QtCore.QRect(0, 135, 231, 51))
   #     self.Brushframe.setFrameShape(QtWidgets.QFrame.StyledPanel)
   #     self.Brushframe.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.SizeBox = QtWidgets.QComboBox(self.commandWidget)
        self.SizeBox.setGeometry(QtCore.QRect(130, 140, 91, 41))
        self.SizeBox.addItem("Size 5")
        self.SizeBox.addItem("Size 10")
        self.SizeBox.addItem("Size 15")
        self.SizeBox.addItem("Size 20")
        self.SizeText = QtWidgets.QLabel(self.commandWidget)
        self.SizeText.setGeometry(QtCore.QRect(20, 150, 91, 16))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.SizeText.setFont(font)
        self.SizeText.setText("Brush Size:")

#Clears the canvas
        self.Clearbutton = QtWidgets.QPushButton(self.commandWidget)
        self.Clearbutton.setGeometry(QtCore.QRect(10, 240, 91, 41))
        self.Clearbutton.setStatusTip("Clear the canvas")
        self.Clearbutton.setText("Clear")
        self.Clearbutton.clicked.connect(self.clear)

#Linked to predict method, which loads model and attempts to predict canvas
        self.Predictbutton = QtWidgets.QPushButton(self.commandWidget)
        self.Predictbutton.setGeometry(QtCore.QRect(130, 240, 91, 41))
        self.Predictbutton.setStatusTip("This will predict character with chosen model")
        self.Predictbutton.setText("Predict")
        self.Predictbutton.clicked.connect(self.test)

#This displays the probability
        self.textEdit = QtWidgets.QFrame(self.commandWidget)
        self.textEdit.setGeometry(QtCore.QRect(0, 280, 231, 191))
        self.textEdit.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.textEdit.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.probDisplay = QtWidgets.QLabel(self.commandWidget)
        self.probDisplay.setGeometry(QtCore.QRect(0, 280, 231, 191))

        self.label = QtWidgets.QLabel(self.commandWidget)
        self.label.setGeometry(QtCore.QRect(0, 0, 230, 81))
        self.label.setText("Welcome! draw on the canvas \non the left and click recognise \nto predict using chosen model")

        self.image = QImage(self.frame.frameSize(), QImage.Format_RGB32)
        self.image.fill(Qt.white)

        self.drawing = False
        self.brushSize = 5
        self.brushColor = Qt.black

        self.lastPoint = QPoint()

#Commands to create canvas, mainly derived from here: https://www.youtube.com/watch?v=qEgyGyVA1ZQ
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.brush()
            self.drawing = True
            self.lastPoint = event.pos()
        if self.commandWidget.underMouse():
            self.drawing = False
                    
    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) & self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(self.brushColor, self.brushSize, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.lastPoint, event.pos())
            self.lastPoint = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button == Qt.LeftButton:
            self.drawing = False

    def paintEvent(self, event):
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(self.frame.frameRect(), self.image)

    def clear(self):
        self.image.fill(Qt.white)
        self.update()

#After button is pressed, model is used to predict the canvas.
    def test(self):
        global model
        model = Sequential()
        
        #Create model
        model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation="relu"))
        model.add(BatchNormalization())

        model.add(Conv2D(32, (5, 5), activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.25))

        model.add(BatchNormalization())
        model.add(Flatten())

        model.add(Dense(256, activation="relu"))
        model.add(Dense(36, activation="softmax"))

        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        if model != "":
            model.load_weights("Models/best_val_loss_model.h5")
            self.image.save('Images/Image.png')
            # resizing the image to 28x28
            image = cv.imread('Images/Image.png')
        
            labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (28, 28))
            image = image / 255.0
            image = np.reshape(image, (1, image.shape[0], image.shape[1], 1))
            prediction = model.predict(image)
            best_predictions = dict()
    
            for i in range(3):
                max_i = np.argmax(prediction[0])
                acc = round(prediction[0][max_i], 1)
                if acc > 0:
                    label = labels[max_i]
                    best_predictions[label] = acc
                    prediction[0][max_i] = 0
                else:
                    break
           
            self.probDisplay.clear()
            self.probDisplay.setText(f'The result is probably: \n{best_predictions} \nThe probabilities are: \n{prediction}')                   
        else:
            self.label.setText("Error: Please download data")

#Different brush sizes
    def brush(self):
        if self.SizeBox.currentText() == "Size 5":
           self.brushSize = 5
        elif self.SizeBox.currentText() == "Size 10":
           self.brushSize = 10
        elif self.SizeBox.currentText() == "Size 15":
           self.brushSize = 15
        elif self.SizeBox.currentText() == "Size 20":
           self.brushSize = 20

#--------------------------------------------------------------------------------------------------------------MAIN WINDOW---------------------------------------------------------------------------------------------------------------------------------------
#Mainwindow, first window that opens, when closed everything closes.
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Handwritten Digit Recognizer")
        label = QLabel("Welcome! This App will attempt to recognize your hand drawn characters! \n\
        You can also train it using various images")
        label.setAlignment(Qt.AlignCenter)

        self.setFixedSize(QSize(600, 500))

        # Set the central widget of the Window.
        self.setCentralWidget(label)

        view_action = QAction("View training", self)                                                                                                                                                                                                
        view_action.triggered.connect(self.view)

        quit_action = QAction("Quit", self)
        quit_action.setShortcut('ESC')
        quit_action.setStatusTip("Quit Application")
        quit_action.triggered.connect(qApp.quit)

        button1 = QPushButton('Train AI!', self)
        button1.setStatusTip('Train the AI using various handwritten images of characters/numbers')
        button1.move(100,70)
        button1.clicked.connect(self.train)

        button2 = QPushButton('Test AI!', self)
        button2.setStatusTip('Test the AI by hand drawing characters/numbers')
        button2.move(400,70)
        button2.clicked.connect(self.test)

        self.setStatusBar(QStatusBar(self))

        menu = self.menuBar()

        file_menu = menu.addMenu("&File")
        file_menu.addAction(view_action)
        file_menu.addSeparator()
        file_menu.addAction(quit_action)

        self.show()

#Brings you to the view window
    def view(self):
        self.next = ViewWindow()
        self.next.show()

#Brings you to the test window
    def test(self):
        self.next = TestWindow()
        self.next.show()

#Brings you to the train window
    def train(self):
        self.next = TrainWindow()
        self.next.show()

#opens and closes application.
def window():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

window()