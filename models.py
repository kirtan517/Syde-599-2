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
        channels_size_1 = trail.suggest_int("channelSize_1",20,128)
        channels_size_2 = trail.suggest_int("channelSize_2",20,256)

        kernel_size = trail.suggest_int("kernelSize",3,7)

        self.convolutions = nn.Sequential(
            nn.Conv2d(1, channels_size_1, kernel_size, 1),
            nn.ReLU(),
            nn.BatchNorm2d(channels_size_1),
            nn.Conv2d(channels_size_1, channels_size_2, kernel_size, 1),
            nn.ReLU(),
            nn.BatchNorm2d(channels_size_2),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )

        input = torch.rand(12, 1, 28, 28)
        output = self.convolutions.forward(input)
        in_features = self.flatten(output).shape[1]

        linear_layers = []
        for i in range(params["Number of Linear layers"]):
            out_features = trail.suggest_int("n_units_l{}".format(i), 32, 256)
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