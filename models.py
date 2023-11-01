import torch
import torch.nn as nn
import json


class CustomModel(nn.Module):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.read_config_file()
        self.convolutions = []
        self.linears = []
        self.flatten = nn.Flatten()
        self.read_data("Convolution",torch.nn.Conv2d,self.convolutions)
        self.read_data("Linears",torch.nn.Linear,self.linears)
        self.convolutions = nn.Sequential(*self.convolutions)
        self.linears = nn.Sequential(*self.linears)

    def read_config_file(self):
        with open(self.file_path) as f:
            data = json.load(f)
        print(data)
        self.data = data

    def read_data(self,type,function,store):
        for i in self.data[type]:
            store.append(function(**i))

    def forward(self,x):
        x = self.convolutions(x)
        x = self.flatten(x)
        x = self.linears(x)
        return x

if __name__ == "__main__":
    model = CustomModel("config.json")
    input = torch.rand(12,1,28,28) # Input dims should be N * Channels * height * width
    output = model(input)
    print(output.shape)