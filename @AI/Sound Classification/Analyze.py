import torch
import torch.nn as nn
import torchaudio
import sounddevice as sd
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.autograd import Variable
from IPython.display import Audio

import torch
import torch.nn as nn
import torchaudio
import sounddevice as sd
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.autograd import Variable
from IPython.display import Audio


try:
    # MULTI GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(512, 10)
    model = nn.DataParallel(model)  # Add this line
    model.load_state_dict(torch.load('ResNet18_Best.pth', map_location=device))
    model = model.to(device)
    model = model.eval()

    state_dict = torch.load('ResNet18_Best.pth', map_location=device)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)

    print("Model successfully loaded. + GPU")
except:
    #One GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(512, 10)
    try:
        state_dict = torch.load('ResNet18_Best.pth', map_location=device)
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        model = model.to(device)
        model = model.eval()
        print("Model successfully loaded.+CPU")
    except:
        print("Failed to load the model. Please check the model file.")


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

def continuous_sound_prediction(model, device, transformation, sample_rate, target_sample_rate):
    # Define class labels
    class_labels = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 
                    'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']
    
    while True:
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
            probabilities = F.softmax(outputs, dim=1)  # apply softmax to output
            _, predicted = torch.max(outputs, 1)

        # Get predicted label and its corresponding probability
        predicted_label = class_labels[predicted.item()]
        predicted_confidence = probabilities[0, predicted.item()].item()  # get the probability of the predicted class

        # Only print the output if the confidence is greater than 80% and the label is not in the specified list
        if predicted_confidence >= 0.0 and predicted_label not in ['air_conditioner', 'children_playing', 'street_music']:#THE EXCLUDED LABLES
            print(f"The predicted class is: {predicted_label}, with confidence: {predicted_confidence:.2%}")
        # Sleep for 2 seconds before the next prediction
        #time.sleep(2.0)

# Call the continuous sound prediction function
continuous_sound_prediction(model, device, transformation, SAMPLE_RATE, SAMPLE_RATE)
