import torch.nn as nn


# Define dropout rate
DROPOUT = 0.2


class SoundClassifier_MARK2(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SoundClassifier_MARK2, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten the input
        return self.network(x)
