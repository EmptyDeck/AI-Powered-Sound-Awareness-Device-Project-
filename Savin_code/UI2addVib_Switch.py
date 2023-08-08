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
    print("a")#this is here just to cause an error

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
    change_label = "glass_shatter"
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
from PyQt5.QtGui import QMovie
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


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1036, 702)
        self.received_text = ""
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(30, 430, 971, 211))
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(30, 60, 971, 351))
        self.label_2.setText("")
        self.label_2.setObjectName("image")
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 29))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # Initialize SoundAnalysis and connect the result_signal with the updateLabel function
        self.sound_analysis = SoundAnalysis(model, device, transformation, SAMPLE_RATE)
        self.sound_analysis.result_signal.connect(self.updateLabel)
        self.sound_analysis.result_signal.connect(self.updateLabel2)
        self.sound_analysis.start()  # Start the sound analysis thread
        
    def updateLabel2(self, predicted_label):
        relative_image_folder_path = "../image_file" #\ only works for windows
        image_folder_path= os.path.join(current_path, relative_image_folder_path)
        full_file_name = os.path.join(image_folder_path, f"{predicted_label}.gif")
        self.movie = QMovie(full_file_name)
        self.label.setMovie(self.movie)
        self.movie.start()
        #self.label_2.setPixmap(QtGui.QPixmap(full_file_name))  
        #print(predicted_label)

    def updateLabel(self, predicted_label, predicted_confidence):
        #print("Received signal")  # Print message when signal is received
        self.label.setText(f"{predicted_label}  {predicted_confidence*100:.2f}%")
        #print(predicted_label)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setFont(QtGui.QFont("AppleSystemUIFont",20))
        self.label.setStyleSheet("Color : black")

import RPi.GPIO as GPIO
import threading

class VibrationController:
    def __init__(self):
        # 핀 번호 설정
        self.red_button_pin = 17
        self.yellow_button_pin = 22
        self.green_button_pin = 27
        self.vibration_motor_pin = 18
        
        # 진동 세기 초기화
        self.vibration_intensity_temp = 100
        self.vibration_intensity = 0
        
        # 이전 스위치 상태 초기화
        self.prev_red_button_state = GPIO.HIGH
        self.prev_yellow_button_state = GPIO.HIGH
        self.prev_green_button_state = GPIO.HIGH
        
        self.debounce_time = 0.2
        
        # 진동 상태를 저장하는 변수
        self.vibration_on = False
        
        # GPIO 초기화
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.red_button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(self.yellow_button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(self.green_button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(self.vibration_motor_pin, GPIO.OUT)
        
        # PWM 설정
        self.pwm_frequency = 1000
        self.pwm = GPIO.PWM(self.vibration_motor_pin, self.pwm_frequency)
        self.pwm.start(self.vibration_intensity)
    
    def display_vibration_intensity(self):
        print(f"진동 세기: {self.vibration_intensity}")
    
    def adjust_vibration_intensity(self, button_pin):
        if button_pin == self.red_button_pin:
            self.vibration_intensity = self.vibration_intensity_temp
        elif button_pin == self.yellow_button_pin:
            self.vibration_intensity = max(self.vibration_intensity - 10, 0)
        elif button_pin == self.green_button_pin:
            self.vibration_intensity = min(self.vibration_intensity + 10, 100)
        
        self.pwm.ChangeDutyCycle(self.vibration_intensity)
        self.display_vibration_intensity()
    
    def run(self):
        try:
            while True:
                red_button_state = GPIO.input(self.red_button_pin)
                yellow_button_state = GPIO.input(self.yellow_button_pin)
                green_button_state = GPIO.input(self.green_button_pin)
        
                if red_button_state != self.prev_red_button_state:
                    time.sleep(self.debounce_time)
                    if red_button_state != GPIO.input(self.red_button_pin):
                        
                        if not self.vibration_on:
                            self.vibration_on = True
                            self.adjust_vibration_intensity(self.red_button_pin)
                        else:
                            self.vibration_intensity_temp = self.vibration_intensity
                            self.vibration_intensity = 0
                            self.pwm.ChangeDutyCycle(self.vibration_intensity)
                            self.display_vibration_intensity()
                            self.vibration_on = False
        
                    self.prev_red_button_state = red_button_state
        
                if yellow_button_state != self.prev_yellow_button_state:
                    time.sleep(self.debounce_time)
                    if yellow_button_state != GPIO.input(self.yellow_button_pin):
                        self.adjust_vibration_intensity(self.yellow_button_pin)
                    self.prev_yellow_button_state = yellow_button_state
        
                if green_button_state != self.prev_green_button_state:
                    time.sleep(self.debounce_time)
                    if green_button_state != GPIO.input(self.green_button_pin):
                        self.adjust_vibration_intensity(self.green_button_pin)
                    self.prev_green_button_state = green_button_state
        
                time.sleep(0.01)  
        
        except KeyboardInterrupt:
            self.pwm.stop()
            GPIO.cleanup()
        



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    # VibrationController 객체 생성 및 실행
    vibration_controller = VibrationController()
    vibration_thread = threading.Thread(target=vibration_controller.run)
    vibration_thread.start()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())