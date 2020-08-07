import numpy as np
import matplotlib.pyplot as plt
import math

import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary


class Affine_Network(nn.Module):
    def __init__(self, device):
        super(Affine_Network, self).__init__()
        self.device = device

        self.feature_extractor = Feature_Extractor(self.device)
        self.regression_network = Regression_Network()

    def forward(self, source, target):
        x = self.feature_extractor(torch.cat((source, target), dim=1))
        x = x.view(1, -1)
        x = self.regression_network(x)
        return x

class Regression_Network(nn.Module):
    def __init__(self):
        super(Regression_Network, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(256, 6),
        )

    def forward(self, x):
        x = self.fc(x)
        return x.view(-1, 2, 3)

class Forward_Layer(nn.Module):
    def __init__(self, channels, pool=False):
        super(Forward_Layer, self).__init__()
        self.pool = pool
        if self.pool:
            self.pool_layer = nn.Sequential(
                nn.Conv2d(channels, 2*channels, 3, stride=2, padding=3)
            )
            self.layer = nn.Sequential(
                nn.Conv2d(channels, 2*channels, 3, stride=2, padding=3),
                nn.GroupNorm(2*channels, 2*channels),
                nn.PReLU(),
                nn.Conv2d(2*channels, 2*channels, 3, stride=1, padding=1),
                nn.GroupNorm(2*channels, 2*channels),
                nn.PReLU(),
            )
        else:
            self.layer = nn.Sequential(
                nn.Conv2d(channels, channels, 3, stride=1, padding=1),
                nn.GroupNorm(channels, channels),
                nn.PReLU(),
                nn.Conv2d(channels, channels, 3, stride=1, padding=1),
                nn.GroupNorm(channels, channels),
                nn.PReLU(),
            )

    def forward(self, x):
        if self.pool:
            return self.pool_layer(x) + self.layer(x)
        else:
            return x + self.layer(x)

class Feature_Extractor(nn.Module):
    def __init__(self, device):
        super(Feature_Extractor, self).__init__()
        self.device = device
        self.input_layer = nn.Sequential(
            nn.Conv2d(2, 64, 7, stride=2, padding=3),
        )
        self.layer_1 = Forward_Layer(64, pool=True)
        self.layer_2 = Forward_Layer(128, pool=False)
        self.layer_3 = Forward_Layer(128, pool=True)
        self.layer_4 = Forward_Layer(256, pool=False)
        self.layer_5 = Forward_Layer(256, pool=True)
        self.layer_6 = Forward_Layer(512, pool=True)

        self.last_layer = nn.Sequential(
            nn.Conv2d(1024, 512, 3, stride=2, padding=1),
            nn.GroupNorm(512, 512),
            nn.PReLU(),
            nn.Conv2d(512, 256, 3, stride=2, padding=1),
            nn.GroupNorm(256, 256),
            nn.PReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.layer_6(x)
        x = self.last_layer(x)
        return x

def load_network(device, path=None):
    model = Affine_Network(device)
    model = model.to(device)
    if path is not None:
        model.load_state_dict(torch.load(path))
        model.eval()
    return model

def test_forward_pass():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_network(device)
    y_size = 760
    x_size = 512
    no_channels = 1
    summary(model, [(no_channels, y_size, x_size), (no_channels, y_size, x_size)])

    batch_size = 1
    example_source = torch.rand((batch_size, no_channels, y_size, x_size)).to(device)
    example_target = torch.rand((batch_size, no_channels, y_size, x_size)).to(device)

    example_source[:, :, 200:500, 50:450] = 1
    example_target[:, :, 100:600, 200:400] = 1

    result = model(example_source, example_target)
    print(result.size())


def run():
    test_forward_pass()

if __name__ == "__main__":
    run()

