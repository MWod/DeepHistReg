import os

import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
import torch

import anhir_utils as au


def ncc_loss(sources, targets, device="cpu"):
    size = sources.size(2)*sources.size(3)
    sources_mean = torch.mean(sources, dim=(1, 2, 3)).view(sources.size(0), 1, 1, 1)
    targets_mean = torch.mean(targets, dim=(1, 2, 3)).view(sources.size(0), 1, 1, 1)
    sources_std = torch.std(sources, dim=(1, 2, 3)).view(sources.size(0), 1, 1, 1)
    targets_std = torch.std(targets, dim=(1, 2, 3)).view(sources.size(0), 1, 1, 1)

    ncc = (1/size)*torch.sum((sources - sources_mean)*(targets-targets_mean) / (sources_std * targets_std), dim=(1, 2, 3))
    ncc = torch.mean(ncc)
    if ncc != ncc:
        return torch.autograd.Variable(torch.Tensor([1]), requires_grad=True).to(device)
    return -ncc

def ncc(source, target):
    source_mean = np.mean(source)
    target_mean = np.mean(target)
    source_std = np.std(source)
    target_std = np.std(target)
    ncc = np.mean((source - source_mean)*(target-target_mean) / (source_std * target_std))
    return -ncc

