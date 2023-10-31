import json
import torch.nn as nn

with open('config.json') as f:
   data = json.load(f)

print(nn.Conv2d(**data["Convolution"][0]))