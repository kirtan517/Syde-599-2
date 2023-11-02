import torch
import torch.nn as nn
import json
import optuna

class CustomModel(nn.Module):
    def __init__(self,params = None,trail = None):
        super().__init__()
        self.flatten = nn.Flatten()

        if trail == None and params == None:
            self.withoutOptuna()
        else:
            self.withOptuna(params,trail)

    def withOptuna(self,params,trail):
        in_channels = 1
        convolutional_layers = []

        for i in range(params["Number of Convolutional layers"]):
            out_channels = trail.suggest_int("n_units_c{}".format(i), 20, 32)
            kernel_size = trail.suggest_int("n_kernel_size{}".format(i), 3, 5)
            convolutional_layers.append(nn.Conv2d(in_channels, out_channels,kernel_size))
            convolutional_layers.append(nn.ReLU())
            p = trail.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
            convolutional_layers.append(nn.Dropout(p))

            in_channels = out_channels

        self.convolutions = nn.Sequential(*convolutional_layers)

        input = torch.rand(12, 1, 28, 28)
        output = self.convolutions.forward(input)
        in_features = self.flatten(output).shape[1]

        linear_layers = []
        for i in range(params["Number of Linear layers"]):
            out_features = trail.suggest_int("n_units_l{}".format(i), 4, 128)
            linear_layers.append(nn.Linear(in_features, out_features))
            linear_layers.append(nn.ReLU())
            p = trail.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
            linear_layers.append(nn.Dropout(p))
            in_features = out_features

        linear_layers.append(nn.Linear(in_features, 10))
        linear_layers.append(nn.LogSoftmax(dim=1))

        self.linears = nn.Sequential(*linear_layers)


    def withoutOptuna(self):
        self.convolutions = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),  # 32 * 26 * 26
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),  # 64 * 24 * 24
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 * 24 * 24
            nn.Dropout(0.25)
        )
        input = torch.rand(12, 1, 28, 28)
        output = self.convolutions.forward(input)
        size = self.flatten(output).shape[1]

        self.linears = nn.Sequential(
            nn.Linear(size, 128),
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
