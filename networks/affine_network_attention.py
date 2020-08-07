import numpy as np
import matplotlib.pyplot as plt
import math

import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary


class Affine_Network(nn.Module):
    # Important - assumes the temporary mini batch size equal to 1! The real batch size must be handled by the calling function.
    def __init__(self, device):
        super(Affine_Network, self).__init__()
        self.device = device

        self.feature_extractor = Feature_Extractor(self.device)
        self.feature_combiner = nn.Sequential(
            nn.Conv2d(512, 256, 3, stride=2, padding=1),
            nn.GroupNorm(256, 256),
            nn.PReLU(),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.GroupNorm(256, 256),
            nn.PReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.attention_network = Attention_Network()
        self.regression_network = Regression_Network()

        self.patch_size = (224, 224)
        self.unfold = nn.Unfold(self.patch_size, stride=self.patch_size)

    def forward(self, source, target):
        us, ut = self.pad_and_unfold(source, target)
        grid_size = (math.ceil(source.size(2) / self.patch_size[0]), math.ceil(source.size(3) / self.patch_size[1]))
        us = self.feature_extractor(us)
        ut = self.feature_extractor(ut)
        x = torch.cat((us, ut), dim=1)
        x = self.feature_combiner(x)
        x = x.view(grid_size[0], grid_size[1], x.size(1))
        x = x.permute(2, 0, 1)
        x = x.view(1, 1, x.size(0), x.size(1), x.size(2))
        x = self.attention_network(x)
        x = self.regression_network(x)
        return x

    def pad_and_unfold(self, source, target):
        pad_x = math.ceil(source.size(3) / self.patch_size[1])*self.patch_size[1] - source.size(3)
        pad_y = math.ceil(source.size(2) / self.patch_size[0])*self.patch_size[0] - source.size(2)
        b_x, e_x = math.floor(pad_x / 2), math.ceil(pad_x / 2)
        b_y, e_y = math.floor(pad_y / 2), math.ceil(pad_y / 2)
        source = F.pad(source, (b_x, e_x, b_y, e_y))
        target = F.pad(target, (b_x, e_x, b_y, e_y))
        us = self.unfold(source)
        ut = self.unfold(target)
        us = us.view(us.size(0), 1, self.patch_size[0], self.patch_size[1], us.size(2))
        ut = ut.view(ut.size(0), 1, self.patch_size[0], self.patch_size[1], ut.size(2))
        us = us[0].permute(3, 0, 1, 2)
        ut = ut[0].permute(3, 0, 1, 2)
        return us, ut

    def show_patchs(self, image, grid_size):
        t_image = image.detach().cpu()
        plt.figure()
        for i in range(grid_size[0]*grid_size[1]):
            plt.subplot(grid_size[0], grid_size[1], i+1)
            plt.imshow(t_image[i, 0, :, :,], cmap='gray')
            plt.axis('off')

class Attention_Network(nn.Module):
    def __init__(self):
        super(Attention_Network, self).__init__()

        self.attention_module = nn.Sequential(
            nn.Conv3d(1, 256, kernel_size=(256, 3, 3), padding=(0, 1, 1)),
            nn.GroupNorm(256, 256),
            nn.PReLU(),
        )

        self.compose_module = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.GroupNorm(256, 256),
            nn.PReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.attention_module(x)
        x = x[:, :, 0, :, :]
        x = self.compose_module(x)
        x = x.view(1, -1)
        return x

class Regression_Network(nn.Module):
    def __init__(self):
        super(Regression_Network, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.PReLU(),
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
            nn.Conv2d(1, 32, 7, stride=2, padding=3),
        )
        self.layer_1 = Forward_Layer(32, pool=True)
        self.layer_2 = Forward_Layer(64, pool=True)
        self.layer_3 = Forward_Layer(128, pool=True)
        self.layer_4 = Forward_Layer(256)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
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

