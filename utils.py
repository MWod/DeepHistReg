import os
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import SimpleITK as sitk

import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

def load_landmarks(landmarks_path):
    landmarks = pd.read_csv(landmarks_path)
    landmarks = landmarks.to_numpy()[:, 1:]
    return landmarks

def save_landmarks(landmarks, landmarks_path):
    df = pd.DataFrame(landmarks)
    df.to_csv(landmarks_path)

def save_landmarks_submission(landmarks, landmarks_path):
    df = pd.DataFrame(landmarks)
    df.rename(columns={0: 'X', 1: 'Y'}, inplace=True)
    index_dict = {i: i +1 for i in range(len(landmarks))}
    df.rename(index=index_dict, inplace=True)
    df.to_csv(landmarks_path)

def pad_landmarks(landmarks, old_shape, new_shape):
    new_landmarks = landmarks.copy()
    new_landmarks[:, 0] += int(np.floor((new_shape[1] - old_shape[1])/2))
    new_landmarks[:, 1] += int(np.floor((new_shape[0] - old_shape[0])/2))
    return new_landmarks

def resample_landmarks(landmarks, resample_ratio):
    new_landmarks = landmarks / resample_ratio
    return new_landmarks

def resample_image(image, resample_ratio):
    y_size, x_size = image.shape
    new_y_size, new_x_size = int(y_size / resample_ratio), int(x_size / resample_ratio)
    grid_x, grid_y = np.meshgrid(np.arange(new_x_size), np.arange(new_y_size))
    grid_x = grid_x * (x_size / new_x_size)
    grid_y = grid_y * (y_size / new_y_size)
    resampled_image = nd.map_coordinates(image, [grid_y, grid_x], cval=0, order=3)
    return resampled_image

def calculate_tre(source_landmarks, target_landmarks):
    tre = np.sqrt(np.square(source_landmarks[:, 0] - target_landmarks[:, 0]) + np.square(source_landmarks[:, 1] - target_landmarks[:, 1]))
    return tre

def calculate_rtre(source_landmarks, target_landmarks, image_diagonal):
    tre = calculate_tre(source_landmarks, target_landmarks)
    rtre = tre / image_diagonal
    return rtre

def pad_single(image, new_shape):
    y_size, x_size = image.shape
    y_pad = ((int(np.floor((new_shape[0] - y_size)/2))), int(np.ceil((new_shape[0] - y_size)/2)))
    x_pad = ((int(np.floor((new_shape[1] - x_size)/2))), int(np.ceil((new_shape[1] - x_size)/2)))
    new_image = np.pad(image, (y_pad, x_pad), constant_values=0)
    return new_image

def calculate_new_shape_max(current_shape, max_size):
    if current_shape[0] < current_shape[1]:
        divider = current_shape[1] / max_size
    else:
        divider = current_shape[0] / max_size
    new_shape = (int(current_shape[0] / divider), int(current_shape[1] / divider))
    return new_shape

def calculate_new_shape_min(current_shape, min_size):
    if current_shape[0] > current_shape[1]:
        divider = current_shape[1] / min_size
    else:
        divider = current_shape[0] / min_size
    new_shape = (int(current_shape[0] / divider), int(current_shape[1] / divider))
    return new_shape

def transform_landmarks(landmarks, displacement_field):
    u_x = displacement_field[0, :, :].detach().cpu().numpy()
    u_y = displacement_field[1, :, :].detach().cpu().numpy()
    landmarks_x = landmarks[:, 0]
    landmarks_y = landmarks[:, 1]
    ux = nd.map_coordinates(u_x, [landmarks_y, landmarks_x], mode='nearest')
    uy = nd.map_coordinates(u_y, [landmarks_y, landmarks_x], mode='nearest')
    new_landmarks = np.stack((landmarks_x + ux, landmarks_y + uy), axis=1)
    return new_landmarks

def generate_rotation_matrix(angle, x0, y0):
    angle = angle * np.pi/180
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    cm1 = np.array([
        [1, 0, x0],
        [0, 1, y0],
        [0, 0, 1]
    ])
    cm2 = np.array([
        [1, 0, -x0],
        [0, 1, -y0],
        [0, 0, 1]
    ])
    transform = cm1 @ rotation_matrix @ cm2
    return transform[0:2, :]

def center_of_mass(tensor, device='cpu'):
    y_size, x_size = tensor.size(0), tensor.size(1)
    gy, gx = torch.meshgrid(torch.arange(y_size), torch.arange(x_size))
    gy = gy.type(torch.FloatTensor).to(device)
    gx = gx.type(torch.FloatTensor).to(device)
    m00 = torch.sum(tensor).item()
    m10 = torch.sum(gx*tensor).item()
    m01 = torch.sum(gy*tensor).item()
    com_x = m10 / m00
    com_y = m01 / m00
    return com_x, com_y

def pad_images_np(source, target):
    y_size_source, x_size_source = source.shape
    y_size_target, x_size_target = target.shape
    new_y_size = max(y_size_source, y_size_target)
    new_x_size = max(x_size_source, x_size_target)
    new_shape = (new_y_size, new_x_size)

    padded_source = pad_single(source, new_shape)
    padded_target = pad_single(target, new_shape)
    return padded_source, padded_target

def numpy_warp_image(image, u_x, u_y, cval=0.0):
    y_size, x_size = image.shape
    grid_x, grid_y = np.meshgrid(np.arange(x_size), np.arange(y_size))
    return nd.map_coordinates(image, [grid_y + u_y, grid_x + u_x], order=3, cval=cval)

def numpy_affine_transform(image, transform, cval=0.0):
    shape = image.shape
    u = np.zeros((2, shape[0], shape[1]))
    grid_x, grid_y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    u[0, :, :] = transform[0, 0] * grid_x + transform[0, 1] * grid_y + transform[0, 2] - grid_x
    u[1, :, :] = transform[1, 0] * grid_x + transform[1, 1] * grid_y + transform[1, 2] - grid_y
    transformed_image = np.empty(shape)
    if len(image.shape) == 3:
        for i in range(shape[2]):
            transformed_image[:, :, i] = numpy_warp_image(image[:, :, i], u[0], u[1], cval=cval)
    else:
        transformed_image = numpy_warp_image(image, u[0], u[1], cval=cval)
    return transformed_image

def numpy_inv_transform(transform):
    total_transform = np.eye(3)
    total_transform[0:2, :] = transform
    inverted_transform = np.linalg.inv(total_transform)
    return inverted_transform[0:2, :]

def load_pair(case_id, dataset_path, load_masks=False):
    base_path = os.path.join(dataset_path, str(case_id))
    source_path = os.path.join(base_path, "source.mha")
    target_path = os.path.join(base_path, "target.mha")
    if load_masks:
        source_mask_path = os.path.join(base_path, "source_mask.mha")
        target_mask_path = os.path.join(base_path, "target_mask.mha")
    source_landmarks_path = os.path.join(base_path, "source_landmarks.csv")
    target_landmarks_path = os.path.join(base_path, "target_landmarks.csv")

    source = sitk.GetArrayFromImage(sitk.ReadImage(source_path))
    target = sitk.GetArrayFromImage(sitk.ReadImage(target_path))
    if load_masks:
        source_mask = sitk.GetArrayFromImage(sitk.ReadImage(source_mask_path))
        target_mask = sitk.GetArrayFromImage(sitk.ReadImage(target_mask_path))
    source_landmarks = pd.read_csv(source_landmarks_path).to_numpy()[:, 1:]
    try:
        status = "training"
        target_landmarks = pd.read_csv(target_landmarks_path).to_numpy()[:, 1:]
    except:
        status = "evaluation"
        target_landmarks = None
    if load_masks:
        return source, target, source_landmarks, target_landmarks, status, source_mask, target_mask
    else:
        return source, target, source_landmarks, target_landmarks, status,

def compose_displacement_field(u, v, device="cpu", delete_outliers=True, return_indexes=False):
    size = u.size()
    x_size = size[2]
    y_size = size[1]
    gy, gx = torch.meshgrid(torch.arange(y_size), torch.arange(x_size))
    gy = gy.type(torch.FloatTensor).to(device)
    gx = gx.type(torch.FloatTensor).to(device)
    grid_x = (gx / (x_size - 1) - 0.5)*2
    grid_y = (gy / (y_size - 1) - 0.5)*2
    u_x_1 = u[0, :, :].view(1, u.size(1), u.size(2))
    u_y_1 = u[1, :, :].view(1, u.size(1), u.size(2))
    u_x_2 = v[0, :, :].view(1, v.size(1), v.size(2))
    u_y_2 = v[1, :, :].view(1, v.size(1), v.size(2))
    u_x_1 = u_x_1 / (x_size - 1) * 2
    u_y_1 = u_y_1 / (y_size - 1) * 2
    u_x_2 = u_x_2 / (x_size - 1) * 2
    u_y_2 = u_y_2 / (y_size - 1) * 2
    n_grid_x = grid_x.view(grid_x.size(0), grid_x.size(1))
    n_grid_y = grid_y.view(grid_y.size(0), grid_y.size(1))
    n_grid = torch.stack((n_grid_x, n_grid_y), dim=2)
    nv = torch.stack((u_x_2, u_y_2), dim=3)[0]
    t_x = n_grid_x.view(1, n_grid_x.size(0), n_grid_x.size(1))
    t_y = n_grid_y.view(1, n_grid_y.size(0), n_grid_y.size(1))
    added_x = u_x_1 + t_x
    added_y = u_y_1 + t_y
    added_grid = n_grid + nv
    i_u_x = F.grid_sample(added_x.view(1, added_x.size(0), added_x.size(1), added_x.size(2)), added_grid.view(1, added_grid.size(0), added_grid.size(1), added_grid.size(2)), padding_mode='border')[0]
    i_u_y = F.grid_sample(added_y.view(1, added_y.size(0), added_y.size(1), added_y.size(2)), added_grid.view(1, added_grid.size(0), added_grid.size(1), added_grid.size(2)), padding_mode='border')[0]
    indexes = (added_grid[:, :, 0] >= 1.0) | (added_grid[:, :, 0] <= -1.0) | (added_grid[:, :, 1] >= 1.0) | (added_grid[:, :, 1] <= -1.0)
    indexes = indexes.view(1, indexes.size(0), indexes.size(1))
    n_x = i_u_x - grid_x
    n_y = i_u_y - grid_y
    if delete_outliers:
        n_x[indexes] = 0.0
        n_y[indexes] = 0.0
    n_x = n_x / 2 * (x_size - 1)
    n_y = n_y / 2 * (y_size - 1)
    if return_indexes:
        return torch.cat((n_x, n_y), dim=0), indexes
    else:
        return torch.cat((n_x, n_y), dim=0)

def compose_displacement_fields(u, v, device="cpu"):
    size = u.size()
    no_samples = size[0]
    x_size = size[3]
    y_size = size[2]
    gy, gx = torch.meshgrid(torch.arange(y_size), torch.arange(x_size))
    gy = gy.type(torch.FloatTensor).to(device)
    gx = gx.type(torch.FloatTensor).to(device)
    grid_x = (gx / (x_size - 1) - 0.5)*2
    grid_y = (gy / (y_size - 1) - 0.5)*2
    u_x_1 = u[:, 0, :, :].view(u.size(0), 1, u.size(2), u.size(3))
    u_y_1 = u[:, 1, :, :].view(u.size(0), 1, u.size(2), u.size(3))
    u_x_2 = v[:, 0, :, :].view(v.size(0), 1, v.size(2), v.size(3))
    u_y_2 = v[:, 1, :, :].view(v.size(0), 1, v.size(2), v.size(3))
    u_x_1 = u_x_1 / (x_size - 1) * 2
    u_y_1 = u_y_1 / (y_size - 1) * 2
    u_x_2 = u_x_2 / (x_size - 1) * 2
    u_y_2 = u_y_2 / (y_size - 1) * 2
    n_grid_x = grid_x.view(1, -1).repeat(no_samples, 1).view(-1, grid_x.size(0), grid_x.size(1))
    n_grid_y = grid_y.view(1, -1).repeat(no_samples, 1).view(-1, grid_y.size(0), grid_y.size(1))
    n_grid = torch.stack((n_grid_x, n_grid_y), dim=3)
    nv = torch.stack((u_x_2.view(u_x_2.size(0), u_x_2.size(2), u_x_2.size(3)), u_y_2.view(u_y_2.size(0), u_y_2.size(2), u_y_2.size(3))), dim=3)
    t_x = n_grid_x.view(n_grid_x.size(0), 1, n_grid_x.size(1), n_grid_x.size(2))
    t_y = n_grid_y.view(n_grid_y.size(0), 1, n_grid_y.size(1), n_grid_y.size(2))
    added_x = u_x_1 + t_x
    added_y = u_y_1 + t_y
    added_grid = n_grid + nv
    i_u_x = F.grid_sample(added_x, added_grid, padding_mode='border')
    i_u_y = F.grid_sample(added_y, added_grid, padding_mode='border')
    indexes = (added_grid[:, :, :, 0] >= 1.0) | (added_grid[:, :, :, 0] <= -1.0) | (added_grid[:, :, :, 1] >= 1.0) | (added_grid[:, :, :, 1] <= -1.0)
    indexes = indexes.view(indexes.size(0), 1, indexes.size(1), indexes.size(2))
    n_x = i_u_x - grid_x
    n_y = i_u_y - grid_y
    n_x[indexes] = 0.0
    n_y[indexes] = 0.0
    n_x = n_x / 2 * (x_size - 1)
    n_y = n_y / 2 * (y_size - 1)
    return torch.cat((n_x, n_y), dim=1)

def compose_transforms(t1, t2, shape, device="cpu"):
    tr1 = torch.zeros((3, 3)).to(device)
    tr2 = torch.zeros((3, 3)).to(device)
    tr1[0:2, :] = t1
    tr2[0:2, :] = t2
    tr1[2, 2] = 1
    tr2[2, 2] = 1
    result = torch.mm(tr1, tr2)
    return result[0:2, :]

def warp_tensor(tensor, displacement_field, device="cpu"):
    size = tensor.size()
    x_size = size[1]
    y_size = size[0]
    gy, gx = torch.meshgrid(torch.arange(y_size), torch.arange(x_size))
    gy = gy.type(torch.FloatTensor).to(device)
    gx = gx.type(torch.FloatTensor).to(device)
    grid_x = (gx / (x_size - 1) - 0.5)*2
    grid_y = (gy / (y_size - 1) - 0.5)*2
    n_grid_x = grid_x.view(grid_x.size(0), grid_x.size(1))
    n_grid_y = grid_y.view(grid_y.size(0), grid_y.size(1))
    n_grid = torch.stack((n_grid_x, n_grid_y), dim=2)
    displacement_field = displacement_field.permute(1, 2, 0)
    u_x = displacement_field[:, :, 0]
    u_y = displacement_field[:, :, 1]
    u_x = u_x / (x_size - 1) * 2
    u_y = u_y / (y_size - 1) * 2
    n_grid[:, :, 0] = n_grid[:, :, 0] + u_x
    n_grid[:, :, 1] = n_grid[:, :, 1] + u_y
    transformed_tensor = F.grid_sample(tensor.view(1, 1, y_size, x_size), n_grid.view(1, n_grid.size(0), n_grid.size(1), n_grid.size(2)), mode='bilinear', padding_mode='zeros')[0, 0, :, :]
    return transformed_tensor

def warp_tensors(tensors, displacement_fields, device="cpu"):
    size = tensors.size()
    no_samples = size[0]
    x_size = size[3]
    y_size = size[2]
    gy, gx = torch.meshgrid(torch.arange(y_size), torch.arange(x_size))
    gy = gy.type(torch.FloatTensor).to(device)
    gx = gx.type(torch.FloatTensor).to(device)
    grid_x = (gx / (x_size - 1) - 0.5)*2
    grid_y = (gy / (y_size - 1) - 0.5)*2
    n_grid_x = grid_x.view(1, -1).repeat(no_samples, 1).view(-1, grid_x.size(0), grid_x.size(1))
    n_grid_y = grid_y.view(1, -1).repeat(no_samples, 1).view(-1, grid_y.size(0), grid_y.size(1))
    n_grid = torch.stack((n_grid_x, n_grid_y), dim=3)
    displacement_fields = displacement_fields.permute(0, 2, 3, 1)
    u_x = displacement_fields[:, :, :, 0]
    u_y = displacement_fields[:, :, :, 1]
    u_x = u_x / (x_size - 1) * 2
    u_y = u_y / (y_size - 1) * 2
    n_grid[:, :, :, 0] = n_grid[:, :, :, 0] + u_x
    n_grid[:, :, :, 1] = n_grid[:, :, :, 1] + u_y
    transformed_tensors = F.grid_sample(tensors, n_grid, mode='bilinear', padding_mode='zeros')
    return transformed_tensors

def resample_tensor(tensor, new_size, device="cpu"):
    x_size = new_size[1] 
    y_size = new_size[0]
    gy, gx = torch.meshgrid(torch.arange(y_size), torch.arange(x_size))
    gy = gy.type(torch.FloatTensor).to(device)
    gx = gx.type(torch.FloatTensor).to(device) 
    grid_x = (gx / (x_size - 1) - 0.5)*2
    grid_y = (gy / (y_size - 1) - 0.5)*2
    n_grid_x = grid_x.view(-1, grid_x.size(0), grid_x.size(1))
    n_grid_y = grid_y.view(-1, grid_y.size(0), grid_y.size(1))
    n_grid = torch.stack((n_grid_x, n_grid_y), dim=3)
    resampled_tensor = F.grid_sample(tensor.view(1, 1, tensor.size(0), tensor.size(1)), n_grid, mode='bilinear', padding_mode='zeros')[0, 0, :, :]
    return resampled_tensor

def resample_tensors(tensors, new_size, device="cpu"):
    current_size = tensors.size()
    no_samples = new_size[0]
    x_size = new_size[3]
    y_size = new_size[2]
    gy, gx = torch.meshgrid(torch.arange(y_size), torch.arange(x_size))
    gy = gy.type(torch.FloatTensor).to(device)
    gx = gx.type(torch.FloatTensor).to(device) 
    grid_x = (gx / (x_size - 1) - 0.5)*2
    grid_y = (gy / (y_size - 1) - 0.5)*2
    n_grid_x = grid_x.view(1, -1).repeat(no_samples, 1).view(-1, grid_x.size(0), grid_x.size(1))
    n_grid_y = grid_y.view(1, -1).repeat(no_samples, 1).view(-1, grid_y.size(0), grid_y.size(1))
    n_grid = torch.stack((n_grid_x, n_grid_y), dim=3)
    resampled_tensors = F.grid_sample(tensors, n_grid, mode='bilinear', padding_mode='zeros')
    return resampled_tensors

def affine2theta(affine, shape):
    h, w = shape[0], shape[1]
    temp = affine
    theta = torch.zeros([2, 3])
    theta[0, 0] = temp[0, 0]
    theta[0, 1] = temp[0, 1]*h/w
    theta[0, 2] = temp[0, 2]*2/w + theta[0, 0] + theta[0, 1] - 1
    theta[1, 0] = temp[1, 0]*w/h
    theta[1, 1] = temp[1, 1]
    theta[1, 2] = temp[1, 2]*2/h + theta[1, 0] + theta[1, 1] - 1
    return theta

def theta2affine(theta, shape):
    h, w = shape[0], shape[1]
    temp = theta
    affine = np.zeros((2, 3))
    affine[1, 2] = (temp[1, 2] - temp[1, 0] - temp[1, 1] + 1)*h/2
    affine[1, 1] = temp[1, 1]
    affine[1, 0] = temp[1, 0]*h/w
    affine[0, 2] = (temp[0, 2] - temp[0, 0] - temp[0, 1] + 1)*w/2
    affine[0, 1] = temp[0, 1]*w/h
    affine[0, 0] = temp[0, 0]
    return affine

def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (kernel_size - 1)/2.
    variance = sigma**2.
    gaussian_kernel = (1./(2.*math.pi*variance)) * torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) /(2*variance))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False, padding=int((kernel_size / 2)))
    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter

def tensor_affine_transform(tensor, tensor_transform):
    affine_grid = F.affine_grid(tensor_transform, tensor.size())
    transformed_tensor = F.grid_sample(tensor, affine_grid)
    return transformed_tensor

def transform_to_displacement_field(tensor, tensor_transform, device='cpu'):
    y_size, x_size = tensor.size(2), tensor.size(3)
    deformation_field = F.affine_grid(tensor_transform, tensor.size())
    gy, gx = torch.meshgrid(torch.arange(y_size), torch.arange(x_size))
    gy = gy.type(torch.FloatTensor).to(device)
    gx = gx.type(torch.FloatTensor).to(device) 
    grid_x = (gx / (x_size - 1) - 0.5)*2
    grid_y = (gy / (y_size - 1) - 0.5)*2
    u_x = deformation_field[0, :, :, 0] - grid_x
    u_y = deformation_field[0, :, :, 1] - grid_y
    u_x = u_x / 2 * (x_size - 1)
    u_y = u_y / 2 * (y_size - 1)
    displacement_field = torch.cat((u_x.view(1, y_size, x_size), u_y.view(1, y_size, x_size)), dim=0)
    return displacement_field

def upsample_displacement_fields(displacement_fields, new_size, device="cpu"):
    no_samples = new_size[0]
    old_x_size = displacement_fields.size(3)
    old_y_size = displacement_fields.size(2)
    x_size = new_size[3]
    y_size = new_size[2]
    gy, gx = torch.meshgrid(torch.arange(y_size), torch.arange(x_size))
    gy = gy.type(torch.FloatTensor).to(device)
    gx = gx.type(torch.FloatTensor).to(device) 
    grid_x = (gx / (x_size - 1) - 0.5)*2
    grid_y = (gy / (y_size - 1) - 0.5)*2
    n_grid_x = grid_x.view(1, -1).repeat(no_samples, 1).view(-1, grid_x.size(0), grid_x.size(1))
    n_grid_y = grid_y.view(1, -1).repeat(no_samples, 1).view(-1, grid_y.size(0), grid_y.size(1))
    n_grid = torch.stack((n_grid_x, n_grid_y), dim=3)
    resampled_displacement_fields = F.grid_sample(displacement_fields, n_grid, mode='bilinear', padding_mode='zeros')
    resampled_displacement_fields[:, 0, :, :] *= x_size / old_x_size
    resampled_displacement_fields[:, 1, :, :] *= y_size / old_y_size
    return resampled_displacement_fields

def upsample_displacement_field(displacement_field, new_size, device="cpu"):
    old_x_size = displacement_field.size(2)
    old_y_size = displacement_field.size(1)
    x_size = new_size[2]
    y_size = new_size[1]
    gy, gx = torch.meshgrid(torch.arange(y_size), torch.arange(x_size))
    gy = gy.type(torch.FloatTensor).to(device)
    gx = gx.type(torch.FloatTensor).to(device) 
    grid_x = (gx / (x_size - 1) - 0.5)*2
    grid_y = (gy / (y_size - 1) - 0.5)*2
    n_grid_x = grid_x.view(-1, grid_x.size(0), grid_x.size(1))
    n_grid_y = grid_y.view(-1, grid_y.size(0), grid_y.size(1))
    n_grid = torch.stack((n_grid_x, n_grid_y), dim=3)
    resampled_displacement_field = F.grid_sample(displacement_field.view(1, 2, displacement_field.size(1), displacement_field.size(2)), n_grid, mode='bilinear', padding_mode='zeros')[0, :, :, :]
    resampled_displacement_field[0, :, :] *= x_size / old_x_size
    resampled_displacement_field[1, :, :] *= y_size / old_y_size
    return resampled_displacement_field

def tensor_laplacian(tensor, device="cpu"):
    laplacian_filter = torch.Tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]).to(device)
    laplacian = F.conv2d(tensor, laplacian_filter.view(1, 1, 3, 3), padding=1) / 9
    return laplacian

def tensor_laplacian_2(tensor, device="cpu"):
    laplacian_filter = torch.Tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).to(device)
    laplacian = F.conv2d(tensor, laplacian_filter.view(1, 1, 3, 3), padding=1)
    return laplacian

def fold(unfolded_tensor, padded_output_size, padding_tuple, patch_size, stride, device="cpu"):
    new_tensor = torch.zeros((1, unfolded_tensor.size(1),) + padded_output_size).to(device)
    col_y, col_x = int(padded_output_size[0] / stride - 1), int(padded_output_size[1] / stride - 1)
    for j in range(col_y):
        for i in range(col_x):
            current_patch = unfolded_tensor[j*col_x + i, :, int(stride/2):-int(stride/2), int(stride/2):-int(stride/2)]
            b_x = i*stride + int(stride/2)
            e_x = (i+1)*stride + int(stride/2)
            b_y = j*stride + int(stride/2)
            e_y = (j+1)*stride + int(stride/2)
            new_tensor[0, :, b_y:e_y, b_x:e_x] = current_patch
    if padding_tuple[2] == 0 and padding_tuple[3] == 0:
        new_tensor = new_tensor[:, :, padding_tuple[1]:, padding_tuple[0]:]
    elif padding_tuple[2] == 0:
        new_tensor = new_tensor[:, :, padding_tuple[1]:-padding_tuple[3], padding_tuple[0]:]
    elif padding_tuple[3] == 0:
        new_tensor = new_tensor[:, :, padding_tuple[1]:, padding_tuple[0]:-padding_tuple[2]]
    else:
        new_tensor = new_tensor[:, :, padding_tuple[1]:-padding_tuple[3], padding_tuple[0]:-padding_tuple[2]]
    return new_tensor

def unfold(tensor, patch_size, stride, device="cpu"):
    unfolder = nn.Unfold(patch_size, stride=stride)
    pad_x = math.ceil(tensor.size(3) / patch_size[1])*patch_size[1] - tensor.size(3)
    pad_y = math.ceil(tensor.size(2) / patch_size[0])*patch_size[0] - tensor.size(2)
    b_x, e_x = math.floor(pad_x / 2) + patch_size[0], math.ceil(pad_x / 2) + patch_size[0]
    b_y, e_y = math.floor(pad_y / 2) + patch_size[1], math.ceil(pad_y / 2) + patch_size[1]
    new_tensor = F.pad(tensor, (b_x, e_x, b_y, e_y))
    padding_tuple = (b_x, b_y, e_x, e_y)
    padded_output_size = (new_tensor.size(2), new_tensor.size(3))
    new_tensor = unfolder(new_tensor)
    new_tensor = new_tensor.view(new_tensor.size(0), tensor.size(1), patch_size[0], patch_size[1], new_tensor.size(2))
    new_tensor = new_tensor[0].permute(3, 0, 1, 2)
    return new_tensor, padded_output_size, padding_tuple

def build_pyramid(tensor, num_levels, device="cpu"):
    pyramid = []
    for i in range(num_levels):
        if i == num_levels - 1:
            pyramid.append(tensor)
        else:
            current_size = tensor.size()
            new_size = torch.Size((current_size[0], current_size[1], int(current_size[2]/(2**(num_levels-i-1))), int(current_size[3]/(2**(num_levels-i-1)))))
            new_tensor = resample_tensors(tensor, new_size, device=device)
            pyramid.append(new_tensor)
    return pyramid





