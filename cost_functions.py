import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

def ncc_loss_global(source, target, device="cpu", **params):
    return ncc_losses_global(source.view(1, 1, source.size(0), source.size(1)), target.view(1, 1, target.size(0), target.size(1)), device=device, **params)

def ncc_losses_global(sources, targets, device="cpu", **params):
    ncc = ncc_global(sources, targets, device=device, **params)
    ncc = torch.mean(ncc)
    if ncc != ncc:
        return torch.autograd.Variable(torch.Tensor([1]), requires_grad=True).to(device)
    return -ncc

def ncc_global(sources, targets, device="cpu", **params):
    size = sources.size(2)*sources.size(3)
    sources_mean = torch.mean(sources, dim=(1, 2, 3)).view(sources.size(0), 1, 1, 1)
    targets_mean = torch.mean(targets, dim=(1, 2, 3)).view(sources.size(0), 1, 1, 1)
    sources_std = torch.std(sources, dim=(1, 2, 3)).view(sources.size(0), 1, 1, 1)
    targets_std = torch.std(targets, dim=(1, 2, 3)).view(sources.size(0), 1, 1, 1)
    ncc = (1/size)*torch.sum((sources - sources_mean)*(targets-targets_mean) / (sources_std * targets_std), dim=(1, 2, 3))
    return ncc

def curvature_regularization(displacement_fields, device="cpu"):
    u_x = displacement_fields[:, 0, :, :].view(-1, 1, displacement_fields.size(2), displacement_fields.size(3))
    u_y = displacement_fields[:, 1, :, :].view(-1, 1, displacement_fields.size(2), displacement_fields.size(3))
    x_laplacian = utils.tensor_laplacian(u_x, device)[:, :, 1:-1, 1:-1]
    y_laplacian = utils.tensor_laplacian(u_y, device)[:, :, 1:-1, 1:-1]
    x_term = x_laplacian**2
    y_term = y_laplacian**2
    curvature = torch.mean(1/2*(x_term + y_term))
    return curvature

def dice_loss(prediction, target, device="cpu"):
    smooth = 1
    prediction = prediction.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = torch.sum(prediction * target)
    return 1 - ((2 * intersection + smooth) / (prediction.sum() + target.sum() + smooth))

def mind_loss(sources, targets, device="cpu", **params):
    sources = sources.view(sources.size(0), sources.size(1), sources.size(2), sources.size(3), 1)
    targets = targets.view(targets.size(0), targets.size(1), targets.size(2), targets.size(3), 1)
    try:
        dilation = params['dilation']
        radius = params['radius']
        return torch.mean((MINDSSC(sources, device=device, dilation=dilation, radius=radius) - MINDSSC(targets, device=device, dilation=dilation, radius=radius))**2)
    except:
        return torch.mean((MINDSSC(sources, device=device) - MINDSSC(targets, device=device))**2)

def pdist_squared(x):
    # Code from: https://github.com/voxelmorph/voxelmorph/pull/145 (a bit modified)
    xx = (x**2).sum(dim=1).unsqueeze(2)
    yy = xx.permute(0, 2, 1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
    dist[dist != dist] = 0
    dist = torch.clamp(dist.float(), 0.0, np.inf)
    return dist

def MINDSSC(img, radius=2, dilation=2, device="cpu"):
    # Code from: https://github.com/voxelmorph/voxelmorph/pull/145 (a bit modified)
    kernel_size = radius * 2 + 1
    six_neighbourhood = torch.Tensor([[0,1,1],
                                      [1,1,0],
                                      [1,0,1],
                                      [1,1,2],
                                      [2,1,1],
                                      [1,2,1]]).long()
    dist = pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)
    x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
    mask = ((x > y).view(-1) & (dist == 2).view(-1))
    idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1,6,1).view(-1,3)[mask,:]
    idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6,1,1).view(-1,3)[mask,:]
    mshift1 = torch.zeros(12, 1, 3, 3, 3).to(device)
    mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:,0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
    mshift2 = torch.zeros(12, 1, 3, 3, 3).to(device)
    mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:,0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
    rpad1 = nn.ReplicationPad3d(dilation)
    rpad2 = nn.ReplicationPad3d(radius)
    ssd = F.avg_pool3d(rpad2((F.conv3d(rpad1(img), mshift1, dilation=dilation) - F.conv3d(rpad1(img), mshift2, dilation=dilation)) ** 2), kernel_size, stride=1)
    mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
    mind_var = torch.mean(mind, 1, keepdim=True)
    mind_var = torch.clamp(mind_var, mind_var.mean().item()*0.001, mind_var.mean().item()*1000)
    mind /= mind_var
    mind = torch.exp(-mind)
    mind = mind[:, torch.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]
    return mind