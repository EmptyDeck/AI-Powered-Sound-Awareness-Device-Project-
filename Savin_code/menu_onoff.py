selected_labels = ["door_nock","glass_shatter","car_horn","dog_bark","drilling","nothing","siren","nothing2"]

import os
current_path = os.getcwd()
print("현재 경로:", current_path)

# Relative path to the model file
relative_path_to_model = "../@AI/Sound Classification/ResNet18_02.pth"

# Combine the current path and the relative path to create the absolute path to the model
path_to_model = os.path.join(current_path, relative_path_to_model)

print("Path to the model:", path_to_model)

import torch
import torch.nn as nn
from torchvision.models import resnet18

#One GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(512, len(selected_labels))
try:
    state_dict = torch.load(path_to_model, map_location=device)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model = model.eval()
    print("Model successfully loaded deivce : ",device)
except:
    print("Failed to load the model. Please check the model file.")
    

import torchaudio
from torchvision import transforms  # Import the transforms module


#Transform

SAMPLE_RATE = 22050

class MonoToColor(nn.Module):
    def __init__(self, num_channels=3):
        super(MonoToColor, self).__init__()
        self.num_channels = num_channels

    def forward(self, tensor):
        return tensor.repeat(self.num_channels, 1, 1)

# Apply the same transformation as used during training
transformation = transforms.Compose([
    torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=128),
    torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80),
    MonoToColor()
])

import time
import torch.nn.functional as F
import sounddevice as sd

print("device : ",device)
## print every labels
def continuous_sound_prediction(model, device, transformation, sample_rate, target_sample_rate):
    # Define class labels

    # Record a 2 seconds mono audio at the specified sample rate
    duration = 2.0  # seconds
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1) 
    sd.wait()

    # Convert to PyTorch tensor and switch channels and frames
    recording = torch.from_numpy(recording).float()
    recording = torch.transpose(recording, 0, 1)

    # Resample if necessary
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        recording = resampler(recording)

    # Mix down if necessary
    if recording.shape[0] > 1:
        recording = torch.mean(recording, dim=0, keepdim=True)

    # Cut or pad if necessary
    if recording.shape[1] > target_sample_rate:
        recording = recording[:, :target_sample_rate]
    elif recording.shape[1] < target_sample_rate:
        num_missing_samples = target_sample_rate - recording.shape[1]
        last_dim_padding = (0, num_missing_samples)
        recording = nn.functional.pad(recording, last_dim_padding)

    # Apply transformation
    recording = transformation(recording)

    # Make the prediction
    model.eval()  # set model to evaluation mode
    with torch.no_grad():  # deactivate autograd engine to reduce memory usage and speed up computations
        recording = recording.to(device)
        outputs = model(recording[None, ...])
        #probabilities = F.softmax(outputs, dim=1)  # apply softmax to output (for 100%)
        #_, predicted = torch.max(outputs, 1)
        probabilities = torch.sigmoid(outputs)  # apply sigmoid to output (for indivisual points)
        _, predicted = torch.max(outputs, 1)

    # Get predicted label and its corresponding probability
    predicted_label = selected_labels[predicted.item()]
    predicted_confidence = probabilities[0, predicted.item()].item()  # get the probability of the predicted class

    ######## Adjust 'x' probability   #########
    change_label = "na"
    change_probability = 0.5
    try:# if you have something to reduce
        x_index = selected_labels.index(change_label)
        probabilities[0, x_index] = max(0.0, probabilities[0, x_index].item() - change_probability)
    except:#if you dont
        pass
    # Print the probabilities of all labels in one line
    #prob_strs = [f"{label} {probabilities[0, idx].item():.2%}" for idx, label in enumerate(selected_labels)]
    #print(f"/ ".join(prob_strs))


    return predicted_label, predicted_confidence, probabilities

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
import torch
import torch.nn as nn
import torchaudio
import sounddevice as sd
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.autograd import Variable


sample_rate = SAMPLE_RATE
target_sample_rate = SAMPLE_RATE

print(" 결과 값 : ")

# Sound Analysis class running on a separate thread
class SoundAnalysis(QThread):
    # Define a pyqtSignal with str type, which will be used to send the analysis results to the main thread
    result_signal = pyqtSignal(str, float)  # Add a float type for the probability

    def __init__(self, model, device, transformation, sample_rate):
        QThread.__init__(self)
        self.model = model
        self.device = device
        self.transformation = transformation
        self.sample_rate = sample_rate

    def run(self):
        count = 0
        while True:
            predicted_label, predicted_confidence, probabilities = continuous_sound_prediction(model, device, transformation, SAMPLE_RATE, SAMPLE_RATE)                
            self.result_signal.emit(predicted_label, predicted_confidence)
            prob_strs = [f"{label} {probabilities[0, idx].item():.2%}" for idx, label in enumerate(selected_labels)]
            print(f"\r{count} / " + " / ".join(prob_strs), end="")
            count = count + 1


import os
from PyQt5 import QtWidgets, QtCore, QtGui
import webbrowser

class SoundDetectionWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

class EnglishTranscriptionWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.button_open_notebook = QtWidgets.QPushButton("영어 받아쓰기")
        self.button_open_notebook.clicked.connect(self.open_notebook)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.button_open_notebook)
        self.setLayout(layout)

    def open_notebook(self):
        notebook_filename = "SPT_eng_spinx.ipynb"
        webbrowser.open(notebook_filename)


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1036, 702)
        
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 799, 21))
        self.menubar.setObjectName("menubar")
        
        # 메뉴
        self.Menu = QtWidgets.QMenu(self.menubar)
        self.Menu.setObjectName("Menu")
        
        # 다른 언어
        self.Other_Language = QtWidgets.QMenu(self.menubar)
        self.Other_Language.setObjectName("Other_Language")
        
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        
        self.sound_of_things = QtWidgets.QAction(MainWindow)
        self.sound_of_things.setObjectName("sound_of_things")
        
        self.English = QtWidgets.QAction(MainWindow)
        self.English.setObjectName("English")
        
        self.Other_Language.addAction(self.sound_of_things)
        self.Other_Language.addSeparator() # 구분자
        self.Other_Language.addAction(self.English)
        
        self.menubar.addAction(self.Menu.menuAction())
        self.menubar.addAction(self.Other_Language.menuAction())
        # 텍스트
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(30, 430, 971, 211))
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        
        # 이미지
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(30, 60, 971, 351))
        self.label_2.setText("")
        self.label_2.setObjectName("image")
        
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # 주변소리 감지와 영어 받아쓰기 위젯 초기화
        self.sound_detection_widget = SoundDetectionWidget()
        self.english_transcription_widget = EnglishTranscriptionWidget()

        # 액션과 함수를 연결
        self.sound_of_things.triggered.connect(self.start_sound_detection)
        self.English.triggered.connect(self.start_english_transcription)

        # 시작은 주변소리 감지 위젯으로 설정
        self.sound_detection_widget.show()

    def start_sound_detection(self):
        self.english_transcription_widget.hide()
        self.sound_detection_widget.show()

    def start_english_transcription(self):
        self.sound_detection_widget.hide()
        self.english_transcription_widget.show()

    def updateLabel2(self, predicted_label):
        relative_image_folder_path = "../image_file" # \는 윈도우에서만 작동합니다.
        image_folder_path = os.path.join(current_path, relative_image_folder_path)
        full_file_name = os.path.join(image_folder_path, f"{predicted_label}.png")
        self.label_2.setPixmap(QtGui.QPixmap(full_file_name))  
        #print(predicted_label)

    def updateLabel(self, predicted_label, predicted_confidence):
        #print("Received signal")  # 신호가 받아졌을 때 메시지 출력
        self.label.setText(f"{predicted_label}  {predicted_confidence*100:.2f}%")
        #print(predicted_label)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setFont(QtGui.QFont("AppleSystemUIFont",20))
        self.label.setStyleSheet("Color : black")
        self.Menu.setTitle(_translate("MainWindow", "메뉴"))
        self.Other_Language.setTitle(_translate("MainWindow", "다른 언어"))
        self.sound_of_things.setText(_translate("MainWindow", "주변소리 감지"))
        self.English.setText(_translate("MainWindow", "영어"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    widget = EnglishTranscriptionWidget()
    widget.show()
    sys.exit(app.exec_())