import numpy as np
import matplotlib.pyplot as plt
import math

import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary



class Nonrigid_Registration_Network(nn.Module):
    def __init__(self, device):
        super(Nonrigid_Registration_Network, self).__init__()
        self.device = device

        self.encoder_1 = nn.Sequential(
            nn.Conv2d(2, 32, 3, stride=2, padding=0),
            nn.GroupNorm(32, 32),
            nn.PReLU(),
        )
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=0),
            nn.GroupNorm(64, 64),
            nn.PReLU(),
        )
        self.encoder_3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=0),
            nn.GroupNorm(128, 128),
            nn.PReLU(),
        )
        self.encoder_4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=0),
            nn.GroupNorm(256, 256),
            nn.PReLU(),
        )
        self.encoder_5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=0),
            nn.GroupNorm(512, 512),
            nn.PReLU(),
        )

        self.decoder_5 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=0, output_padding=0),
            nn.GroupNorm(256, 256),
            nn.PReLU(),
        )
        self.decoder_4 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 3, stride=2, padding=0, output_padding=0),
            nn.GroupNorm(128, 128),
            nn.PReLU(),
        )
        self.decoder_3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 3, stride=2, padding=0, output_padding=0),
            nn.GroupNorm(64, 64),
            nn.PReLU(),
        )
        self.decoder_2 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, 3, stride=2, padding=0, output_padding=0),
            nn.GroupNorm(32, 32),
            nn.PReLU(),
        )
        self.decoder_1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=0, output_padding=1),
            nn.GroupNorm(32, 32),
            nn.PReLU(),
        )
        self.layer_1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.GroupNorm(32, 32),
            nn.PReLU(),
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.GroupNorm(16, 16),
            nn.PReLU(),
        )
        self.layer_3 = nn.Sequential(
            nn.Conv2d(16, 2, 3, stride=1, padding=1),
        )

    def forward(self, source_patches, target_patches):
        x = torch.cat((source_patches, target_patches), dim=1)
        x1 = self.encoder_1(x)
        x2 = self.encoder_2(x1)
        x3 = self.encoder_3(x2)
        x4 = self.encoder_4(x3)
        x5 = self.encoder_5(x4)
        d5 = self.decoder_5(x5)
        d4 = self.decoder_4(torch.cat((d5, x4), dim=1))
        d3 = self.decoder_3(torch.cat((d4, x3), dim=1))
        d2 = self.decoder_2(torch.cat((d3, x2), dim=1))
        d1 = self.decoder_1(torch.cat((d2, x1), dim=1))
        result = self.layer_1(d1)
        result = self.layer_2(result)
        result = self.layer_3(result)
        return result

def load_network(device, path=None):
    model = Nonrigid_Registration_Network(device)
    model = model.to(device)
    if path is not None:
        model.load_state_dict(torch.load(path))
        model.eval()
    return model

def test_forward_pass_simple():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_network(device)
    y_size = 256
    x_size = 256
    no_channels = 1
    summary(model, [(no_channels, y_size, x_size), (no_channels, y_size, x_size)])

    batch_size = 10
    example_source = torch.rand((batch_size, no_channels, y_size, x_size)).to(device)
    example_target = torch.rand((batch_size, no_channels, y_size, x_size)).to(device)

    result = model(example_source, example_target)
    print(result.size())


def run():
    test_forward_pass_simple()

if __name__ == "__main__":
    run()
