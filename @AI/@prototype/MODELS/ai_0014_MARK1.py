import torch.nn as nn


class SoundClassifier_MARK1(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SoundClassifier_MARK1, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # IMPORTANDT (this model dont use batch norm so this could make error)
        x = x.view(x.size(0), -1).squeeze(1)
        return self.network(x)
