from tensorflow.python.keras.models import load_model
import numpy as np
import pickle
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QAction, QLineEdit, QMessageBox
from PyQt5.QtCore import pyqtSlot
import sys

model = load_model('nextword1.h5')
tokenizer = pickle.load(open('tokenizer1.pkl', 'rb'))

def Predict_Next_Words(model, tokenizer, text):

    for i in range(3):
        sequence = tokenizer.texts_to_sequences([text])[0]
        sequence = np.array(sequence)
        
        preds = np.argmax(model.predict(sequence), axis=-1)
        predicted_word = ""
        
        for key, value in tokenizer.word_index.items():
            if value == preds:
                predicted_word = key
                break
        print(predicted_word)
        return predicted_word


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'Next Word Predicter'
        self.left = 20
        self.top = 20
        self.width = 350
        self.height = 150
        self.setStyleSheet("background-color: green;")
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.textbox = QLineEdit(self)
        self.textbox.setStyleSheet("background-color: white;color: black;")
        self.textbox.move(20, 20)
        self.textbox.resize(280, 40)

        self.button = QPushButton('PREDİCT', self)
        self.button.setStyleSheet("background-color: beige;")
        self.button.move(20, 80)

        self.button.clicked.connect(self.on_click)
        self.show()

    @pyqtSlot()
    def on_click(self):
       try:
            textboxValue = self.textbox.text()
            textboxValue = textboxValue.split(" ")
            textboxValue = textboxValue[-1]

            textboxValue = ''.join(textboxValue)
            result=Predict_Next_Words(model, tokenizer, textboxValue)
       except:
           result='    '
       QMessageBox.question(self, 'Next word prediction', "Next word? :" + result, QMessageBox.Ok)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
