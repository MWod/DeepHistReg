import numpy as np
import matplotlib.pyplot as plt
import math

import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary



class Segmentation_Network(nn.Module):
    def __init__(self, device):
        super(Segmentation_Network, self).__init__()
        self.device = device

        self.encoder_1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=0),
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
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=0, output_padding=0),
            nn.GroupNorm(32, 32),
            nn.PReLU(),
        )
        self.last_layer = nn.Sequential(
            nn.Conv2d(32, 1, 1, stride=1, padding=0),
        )

    def pad(self, image, template):
        pad_x = math.fabs(image.size(3) - template.size(3))
        pad_y = math.fabs(image.size(2) - template.size(2))
        b_x, e_x = math.floor(pad_x / 2), math.ceil(pad_x / 2)
        b_y, e_y = math.floor(pad_y / 2), math.ceil(pad_y / 2)
        image = F.pad(image, (b_x, e_x, b_y, e_y))
        return image

    def forward(self, x):
        x1 = self.encoder_1(x)
        x2 = self.encoder_2(x1)
        x3 = self.encoder_3(x2)
        x4 = self.encoder_4(x3)
        x5 = self.encoder_5(x4)
        d5 = self.decoder_5(x5)
        d5 = self.pad(d5, x4)
        d4 = self.decoder_4(torch.cat((d5, x4), dim=1))
        d4 = self.pad(d4, x3)
        d3 = self.decoder_3(torch.cat((d4, x3), dim=1))
        d3 = self.pad(d3, x2)
        d2 = self.decoder_2(torch.cat((d3, x2), dim=1))
        d2 = self.pad(d2, x1)
        d1 = self.decoder_1(torch.cat((d2, x1), dim=1))
        result = self.last_layer(d1)
        result = self.pad(result, x)
        return torch.sigmoid(result)

def load_network(device, path=None):
    model = Segmentation_Network(device)
    model = model.to(device)
    if path is not None:
        model.load_state_dict(torch.load(path))
        model.eval()
    return model

def test_forward_pass_simple():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_network(device)
    y_size = 512
    x_size = 776
    no_channels = 1
    summary(model, [(no_channels, y_size, x_size), (no_channels, y_size, x_size)])

    batch_size = 1
    example_source = torch.rand((batch_size, no_channels, y_size, x_size)).to(device)

    result = model(example_source)
    print(result.size())


def run():
    test_forward_pass_simple()

if __name__ == "__main__":
    run()
