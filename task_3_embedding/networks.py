import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'


class WordEmbedder(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolution for the one-hot-encoded word
        self.conv1 = nn.Conv1d(in_channels=35, out_channels=64,
                               kernel_size=3, stride=1, padding=1)  # Len = 16

        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # Len = 8

        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128,
                               kernel_size=3, stride=1, padding=1)  # Len = 8

        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # Len = 4

        self.fc1 = nn.Linear(in_features=512, out_features=256)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = self.fc1(x.view(-1, 512))
        return x

class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM()

    def forward(self, x):
        pass