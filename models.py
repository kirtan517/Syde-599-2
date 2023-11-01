import torch
import torch.nn as nn
import json


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.convolutions = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.linears = nn.Sequential(
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self,x):
        x = self.convolutions(x)
        x = self.flatten(x)
        x = self.linears(x)
        return x

if __name__ == "__main__":
    model = CustomModel()
    input = torch.rand(12,1,28,28) # Input dims should be N * Channels * height * width
    output = model(input)
    print(output.shape)