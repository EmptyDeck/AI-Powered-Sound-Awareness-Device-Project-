import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import sounddevice as sd
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.autograd import Variable
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QFontDatabase

# Define the labels for sound classification
selected_labels = ["door_nock", "glass_shatter", "car_horn", "dog_bark", "drilling", "nothing", "siren", "nothing2"]

# Relative path to the model file
relative_path_to_model = "../@AI/Sound Classification/ResNet18_02.pth"
# Combine the current path and the relative path to create the absolute path to the model
current_path = os.getcwd()
path_to_model = os.path.join(current_path, relative_path_to_model)

# One GPU or CPU
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
    print("Model successfully loaded device:", device)
except:
    print("Failed to load the model. Please check the model file.")

# Constants for audio processing
SAMPLE_RATE = 22050
target_sample_rate = SAMPLE_RATE

# Transformation for audio input
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

# Function for continuous sound prediction
def continuous_sound_prediction(model, device, transformation, sample_rate, target_sample_rate):
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
        probabilities = torch.sigmoid(outputs)  # apply sigmoid to output (for individual points)
        _, predicted = torch.max(outputs, 1)

    # Get predicted label and its corresponding probability
    predicted_label = selected_labels[predicted.item()]
    predicted_confidence = probabilities[0, predicted.item()].item()  # get the probability of the predicted class

    # Adjust 'x' probability if needed
    change_label = "na"
    change_probability = 0.5
    try:
        x_index = selected_labels.index(change_label)
        probabilities[0, x_index] = max(0.0, probabilities[0, x_index].item() - change_probability)
    except:
        pass

    return predicted_label, predicted_confidence, probabilities

# Sound Analysis class running on a separate thread
class SoundAnalysis(QThread):
    result_signal = pyqtSignal(str, float)  # Signal for passing analysis results

    def __init__(self, model, device, transformation, sample_rate):
        QThread.__init__(self)
        self.model = model
        self.device = device
        self.transformation = transformation
        self.sample_rate = sample_rate

    def run(self):
        count = 0
        while True:
            predicted_label, predicted_confidence, probabilities = continuous_sound_prediction(
                self.model, self.device, self.transformation, SAMPLE_RATE, SAMPLE_RATE)
            self.result_signal.emit(predicted_label, predicted_confidence)
            prob_strs = [f"{label} {probabilities[0, idx].item():.2%}" for idx, label in enumerate(selected_labels)]
            print(f"\r{count} / " + " / ".join(prob_strs), end="")
            count = count + 1

# UI Class
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

        self.Menu = QtWidgets.QMenu(self.menubar)
        self.Menu.setObjectName("Menu")

        self.Other_Language = QtWidgets.QMenu(self.menubar)
        self.Other_Language.setObjectName("Other_Language")

        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.actionSound_of_things = QtWidgets.QAction(MainWindow)
        self.actionSound_of_things.setObjectName("actionSound_of_things")

        self.actionEnglish = QtWidgets.QAction(MainWindow)
        self.actionEnglish.setObjectName("actionEnglish")

        self.Menu.addAction(self.actionSound_of_things)
        self.Other_Language.addAction(self.actionEnglish)

        self.menubar.addAction(self.Menu.menuAction())
        self.menubar.addAction(self.Other_Language.menuAction())

        self.stackedWidget = QtWidgets.QStackedWidget(self.centralwidget)
        self.stackedWidget.setGeometry(QtCore.QRect(0, 20, 1041, 671))
        self.stackedWidget.setObjectName("stackedWidget")

        # Add the "주변소리 감지" widget to the stackedWidget
        self.page_sound_detection = QtWidgets.QWidget()
        self.page_sound_detection.setObjectName("page_sound_detection")
        self.stackedWidget.addWidget(self.page_sound_detection)

        # Add the "영어 받아쓰기" widget to the stackedWidget
        self.page_english_transcription = QtWidgets.QWidget()
        self.page_english_transcription.setObjectName("page_english_transcription")
        self.stackedWidget.addWidget(self.page_english_transcription)

        self.label = QtWidgets.QLabel(self.page_sound_detection)
        self.label.setGeometry(QtCore.QRect(10, 10, 801, 381))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("sound_recording.jpg"))
        self.label.setObjectName("label")

        self.label_2 = QtWidgets.QLabel(self.page_sound_detection)
        self.label_2.setGeometry(QtCore.QRect(20, 460, 181, 41))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")

        self.label_3 = QtWidgets.QLabel(self.page_sound_detection)
        self.label_3.setGeometry(QtCore.QRect(50, 520, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")

        self.label_result = QtWidgets.QLabel(self.page_sound_detection)
        self.label_result.setGeometry(QtCore.QRect(200, 520, 591, 31))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_result.setFont(font)
        self.label_result.setObjectName("label_result")

        self.label_confidence = QtWidgets.QLabel(self.page_sound_detection)
        self.label_confidence.setGeometry(QtCore.QRect(200, 460, 591, 41))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_confidence.setFont(font)
        self.label_confidence.setObjectName("label_confidence")

        # Define a signal-slot connection to switch to the "영어 받아쓰기" widget when the "actionEnglish" is triggered
        self.actionEnglish.triggered.connect(lambda: self.stackedWidget.setCurrentWidget(self.page_english_transcription))
        # Define a signal-slot connection to switch to the "주변소리 감지" widget when the "actionSound_of_things" is triggered
        self.actionSound_of_things.triggered.connect(lambda: self.stackedWidget.setCurrentWidget(self.page_sound_detection))

        self.retranslateUi(MainWindow)
        self.stackedWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Menu.setTitle(_translate("MainWindow", "Menu"))
        self.Other_Language.setTitle(_translate("MainWindow", "Other Language"))
        self.actionSound_of_things.setText(_translate("MainWindow", "sound of things"))
        self.actionEnglish.setText(_translate("MainWindow", "English"))
        self.label_2.setText(_translate("MainWindow", "Predicted Label:"))
        self.label_3.setText(_translate("MainWindow", "Confidence:"))

        # Connect the SoundAnalysis result_signal to update_result_label slot to update the labels on the UI
        self.sound_analysis = SoundAnalysis(model, device, transformation, SAMPLE_RATE)
        self.sound_analysis.result_signal.connect(self.update_result_label)
        self.sound_analysis.start()

    # Slot to update the labels on the UI
    def update_result_label(self, predicted_label, predicted_confidence):
        self.label_result.setText(predicted_label)
        self.label_confidence.setText(f"{predicted_confidence:.2%}")


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
