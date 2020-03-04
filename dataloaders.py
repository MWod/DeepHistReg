import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import imageio
from skimage import color
import random

import torch
import torch.utils as utils

import augmentation as aug
import anhir_utils as au


def collate_to_list_unsupervised(batch):
    sources = [item[0].view(-1, item[0].size(0), item[0].size(1), item[0].size(2)) for item in batch]
    targets = [item[1].view(-1, item[1].size(0), item[1].size(1), item[1].size(2)) for item in batch]
    return sources, targets

class AffineUnsupervisedLoader(utils.data.Dataset):
    def __init__(self, data_path, transforms=None, background_delete=False, randomly_swap=False):
        self.data_path = data_path
        self.all_ids = os.listdir(self.data_path)
        self.transforms = transforms
        self.background_delete = background_delete
        self.randomly_swap = randomly_swap

    def __len__(self):
        return len(self.all_ids)

    def __getitem__(self, idx):
        case_id = self.all_ids[idx]
        source_path = os.path.join(self.data_path, str(case_id), "source.png")
        target_path = os.path.join(self.data_path, str(case_id), "target.png")
        source = imageio.imread(source_path).astype(np.float32)
        target = imageio.imread(target_path).astype(np.float32)
        if self.background_delete:
            source = au.tissue_segmentation(source)
            target = au.tissue_segmentation(target)
        source = color.rgb2gray(source)
        target = color.rgb2gray(target)
        source = au.normalize(source.reshape(source.shape + (1,)))
        target = au.normalize(target.reshape(target.shape + (1,)))
        source = 1 - source
        target = 1 - target

        if self.transforms is not None:
            source, target, _  = self.transforms(source, target)

        if self.randomly_swap:
            if random.random() > 0.5:
                pass
            else:
                t_source = source.copy()
                source = target
                target = t_source

        source = source.swapaxes(0, 2).swapaxes(2, 1)
        target = target.swapaxes(0, 2).swapaxes(2, 1)

        source_tensor, target_tensor = torch.from_numpy(source.astype(np.float32)), torch.from_numpy(target.astype(np.float32))
        return source_tensor, target_tensor
