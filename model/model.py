import torch
import torch.nn as nn


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        # Convolution Layers
        self.conv_layers = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)

        )

        # Fully Connected Layers
        self.fc_layers = nn.Sequential(

            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(256, 10)

        )

    def forward(self, x):

        x = self.conv_layers(x)

        # flattening
        x = x.view(x.size(0), -1)

        x = self.fc_layers(x)

        return x