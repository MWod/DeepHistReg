import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage as nd
import SimpleITK as sitk
from skimage import color
from skimage import measure
from skimage import segmentation
import csv
import cv2 as cv
from scipy import signal
import torch
import torch.nn as nn
import torch.nn.functional as F


def resample(image, output_x_size, output_y_size):
    y_size, x_size = np.shape(image)
    out_grid_x, out_grid_y = np.meshgrid(np.arange(output_x_size), np.arange(output_y_size))
    out_grid_x = out_grid_x * x_size / output_x_size
    out_grid_y = out_grid_y * y_size / output_y_size
    image = nd.map_coordinates(image, [out_grid_y, out_grid_x], order=3, cval=0.0)
    return image

def resample_both(source, target, resample_ratio):
    s_y_size, s_x_size = source.shape
    t_y_size, t_x_size = target.shape
    source = resample(source, int(s_x_size/resample_ratio), int(s_y_size/resample_ratio))
    target = resample(target, int(t_x_size/resample_ratio), int(t_y_size/resample_ratio))
    return source, target

def resample_displacement_field(u_x, u_y, output_x_size, output_y_size):
    y_size, x_size = np.shape(u_x)
    u_x = resample(u_x, output_x_size, output_y_size)
    u_y = resample(u_y, output_x_size, output_y_size)
    u_x = u_x * output_x_size/x_size
    u_y = u_y * output_y_size/y_size
    return u_x, u_y

def warp_image(image, u_x, u_y, cval=0.0):
    y_size, x_size = image.shape
    grid_x, grid_y = np.meshgrid(np.arange(x_size), np.arange(y_size))
    return nd.map_coordinates(image, [grid_y + u_y, grid_x + u_x], order=3, cval=cval)

def rigid_dot(image, matrix):
    y_size, x_size = np.shape(image)
    x_grid, y_grid = np.meshgrid(np.arange(x_size), np.arange(y_size))
    points = np.vstack((x_grid.ravel(), y_grid.ravel(), np.ones(np.shape(image)).ravel()))
    transformed_points = matrix @ points
    u_x = np.reshape(transformed_points[0, :], (y_size, x_size)) - x_grid
    u_y = np.reshape(transformed_points[1, :], (y_size, x_size)) - y_grid
    return u_x, u_y

def load_image(path, to_gray=True):
    image = sitk.ReadImage(path)
    image = sitk.GetArrayFromImage(image)
    if to_gray:
        image = color.rgb2gray(image)
    return image

def load_landmarks(path):
    landmarks = pd.read_csv(path).iloc[:, 1:].values.astype(np.float)
    return landmarks

def save_landmarks(path, landmarks):
    df = pd.DataFrame(landmarks, columns=['X', 'Y'])
    df.index = np.arange(1, len(df) + 1)
    df.to_csv(path)

def pad_landmarks(landmarks, x, y):
    landmarks[:, 0] += x
    landmarks[:, 1] += y
    return landmarks

def plot_landmarks(landmarks, marker_type, colors=None):
    landmarks_length = len(landmarks)
    if colors is None:
        colors = np.random.uniform(0, 1, (3, landmarks_length))
    for i in range(landmarks_length):
        plt.plot(landmarks[i, 0], landmarks[i, 1], marker_type, color=colors[:, i])
    return colors

def normalize(image):
    if len(image.shape) == 3:
        output_image = np.zeros(image.shape, dtype=image.dtype)
        for i in range(image.shape[2]):
            output_image[:, :, i] = (image[:, :, i] - np.min(image[:, :, i])) / (np.max(image[:, :, i]) - np.min(image[:, :, i]))
        return output_image
    else:
        return (image - np.min(image)) / (np.max(image) - np.min(image))

def z_norm(image):
    if len(image.shape) == 3:
        output_image = np.empty(image.shape, dtype=image.dtype)
        for i in range(image.shape[2]):
            output_image[:, :, i] = (image[:, :, i] - np.mean(image[:, :, i])) / np.std(image[:, :, i])
        return output_image
    else:
        return (image - np.mean(image)) / np.std(image)  

def normalize_by_channel(image):
    for i in range(image.shape[2]):
        image[:, :, i] = (image[:, :, i] - np.min(image[:, :, i])) / (np.max(image[:, :, i] - np.min(image[:, :, i])))
    return image

def to_image(array):
    return sitk.GetImageFromArray((255*array).astype(np.uint8))

def calculate_resample_size(source, target, output_max_size):
    target_y_size, target_x_size = np.shape(target)[0:2]
    source_y_size, source_x_size = np.shape(source)[0:2]

    max_y_size = max(source_y_size, target_y_size)
    max_x_size = max(source_x_size, target_x_size)

    max_dim = max(max_y_size, max_x_size)
    rescale_ratio = max_dim/output_max_size
    return rescale_ratio

def compose_vector_fields(u_x, u_y, v_x, v_y):
    y_size, x_size = np.shape(u_x)
    grid_x, grid_y = np.meshgrid(np.arange(x_size), np.arange(y_size))
    added_y = grid_y + v_y
    added_x = grid_x + v_x
    t_x = nd.map_coordinates(grid_x + u_x, [added_y, added_x], mode='constant', cval=0.0)
    t_y = nd.map_coordinates(grid_y + u_y, [added_y, added_x], mode='constant', cval=0.0)
    n_x, n_y = t_x - grid_x, t_y - grid_y
    indexes_x = np.logical_or(added_x >= x_size - 1, added_x <= 0)
    indexes_y = np.logical_or(added_y >= y_size - 1, added_y <= 0)
    indexes = np.logical_or(indexes_x, indexes_y)
    n_x[indexes] = 0.0
    n_y[indexes] = 0.0
    return n_x, n_y

def gaussian_filter(image, sigma):
    return nd.gaussian_filter(image, sigma)

def round_up_to_odd(value):
    return int(np.ceil(value) // 2 * 2 + 1)

def dice(image_1, image_2):
    image_1 = image_1.astype(np.bool)
    image_2 = image_2.astype(np.bool)
    return 2 * np.logical_and(image_1, image_2).sum() / (image_1.sum() + image_2.sum())

def transform_landmarks(landmarks, u_x, u_y):
    landmarks_x = landmarks[:, 0]
    landmarks_y = landmarks[:, 1]
    ux = nd.map_coordinates(u_x, [landmarks_y, landmarks_x], mode='nearest')
    uy = nd.map_coordinates(u_y, [landmarks_y, landmarks_x], mode='nearest')
    new_landmarks = np.stack((landmarks_x + ux, landmarks_y + uy), axis=1)
    return new_landmarks

def tre(landmarks_1, landmarks_2):
    tre = np.sqrt(np.square(landmarks_1[:, 0] - landmarks_2[:, 0]) + np.square(landmarks_1[:, 1] - landmarks_2[:, 1]))
    return tre

def rtre(landmarks_1, landmarks_2, x_size, y_size):
    return tre(landmarks_1, landmarks_2) / np.sqrt(x_size*x_size + y_size*y_size)

def print_rtre(source_landmarks, target_landmarks, x_size, y_size):
    calculated_tre = rtre(source_landmarks, target_landmarks, x_size, y_size)
    mean = np.mean(calculated_tre) * 100
    median = np.median(calculated_tre) * 100
    mmax = np.max(calculated_tre) * 100
    mmin = np.min(calculated_tre) * 100
    print("TRE mean [%]: ", mean)
    print("TRE median [%]: ", median)
    print("TRE max [%]: ", mmax)
    print("TRE min [%]: ", mmin)
    return mean, median, mmax, mmin

def tissue_segmentation(image):
    gray_image = color.rgb2gray(image)
    laplacian_image = np.abs(nd.laplace(gray_image))
    blurred_laplacian = nd.gaussian_filter(laplacian_image, 5)

    if np.mean(blurred_laplacian) > 0.1:
        mask = blurred_laplacian > 0.01
    else:
        mask = blurred_laplacian > np.mean(blurred_laplacian) / 1.2

    labels = measure.label(mask)
    for label in np.unique(labels):
        if np.sum(mask[labels == label]) / np.size(mask) < 0.02:
            mask[labels == label] = 0

    mask = nd.median_filter(mask, 21)
    mask = nd.binary_opening(mask, structure=np.ones((3, 3)), iterations=3)
    mask = nd.binary_fill_holes(mask)
    mask = nd.binary_closing(mask, structure=np.ones((5, 5)), iterations=3)
    output_image = image.copy()
    output_image[mask == 0] = 255
    return output_image

def affine_transform(image, transform, cval=0.0):
    shape = image.shape
    u = np.zeros((2, shape[0], shape[1]))
    grid_x, grid_y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    u[0, :, :] = transform[0, 0] * grid_x + transform[0, 1] * grid_y + transform[0, 2] - grid_x
    u[1, :, :] = transform[1, 0] * grid_x + transform[1, 1] * grid_y + transform[1, 2] - grid_y
    transformed_image = np.empty(shape)
    if len(image.shape) == 3:
        for i in range(shape[2]):
            transformed_image[:, :, i] = warp_image(image[:, :, i], u[0], u[1], cval=cval)
    else:
        transformed_image = warp_image(image, u[0], u[1], cval=cval)
    return transformed_image

def transform_to_deformation_field(image, transform):
    shape = image.shape
    u = np.zeros((2, shape[0], shape[1]))
    grid_x, grid_y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    u[0, :, :] = transform[0, 0] * grid_x + transform[0, 1] * grid_y + transform[0, 2] - grid_x
    u[1, :, :] = transform[1, 0] * grid_x + transform[1, 1] * grid_y + transform[1, 2] - grid_y
    return u[0], u[1]

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

def rotation_transformation(angle, x0, y0):
    # Function made this way to maintain the gradient history
    rot = torch.Tensor().new_tensor([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=torch.float32)
    cm1 = torch.Tensor().new_tensor([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ], dtype=torch.float32)
    cm2 = torch.Tensor().new_tensor([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=torch.float32)
    rot[0, 0] = torch.cos(angle)
    rot[0, 1] = -torch.sin(angle)
    rot[1, 0] = torch.sin(angle)
    rot[1, 1] = torch.cos(angle)
    cm1[0, 2] = x0
    cm1[1, 2] = y0
    cm2[0, 2] = -x0
    cm2[1, 2] = -y0
    matrix = torch.mm(torch.mm(cm1, rot), cm2)
    final_transform = matrix[0:2, :]
    return final_transform

def tensor_affine_transform(tensor, tensor_transform):
    affine_grid = F.affine_grid(tensor_transform, tensor.size())
    transformed_tensor = F.grid_sample(tensor, affine_grid)
    return transformed_tensor

def tensor_gradient(tensor, device="cpu"):
    # Assumes single channel image (B x 1 x Y x X)
    sobel_x = torch.Tensor([[0, 0, 0], [1, 0, -1], [0, 0, 0]]).to(device)
    sobel_y = torch.transpose(sobel_x, dim0=0, dim1=1)
    x_gradient = F.conv2d(tensor, sobel_x.view(1, 1, 3, 3)) * 9
    y_gradient = F.conv2d(tensor, sobel_y.view(1, 1, 3, 3)) * 9
    return x_gradient, y_gradient

def tensor_gradient_padded(tensor, device="cpu"):
    # Assumes single channel image (B x 1 x Y x X)
    sobel_x = torch.Tensor([[0, 0, 0], [1, 0, -1], [0, 0, 0]]).to(device)
    sobel_y = torch.transpose(sobel_x, dim0=0, dim1=1)
    x_gradient = F.conv2d(tensor, sobel_x.view(1, 1, 3, 3), padding=1) * 9
    y_gradient = F.conv2d(tensor, sobel_y.view(1, 1, 3, 3), padding=1) * 9
    return x_gradient, y_gradient

def inv_transform(transform):
    total_transform = np.eye(3)
    total_transform[0:2, :] = transform
    inverted_transform = np.linalg.inv(total_transform)
    return inverted_transform[0:2, :]

def compose_transform(t1, t2):
    tr1 = np.eye(3)
    tr2 = np.eye(3)
    tr1[0:2, :] = t1
    tr2[0:2, :] = t2
    result = tr1 @ tr2
    return result[0:2, :]

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

def calculate_pad_size(source_shape, target_shape):
    def inner_pad(image_shape, x_size, y_size):
        image_y_shape, image_x_shape = image_shape[0:2]
        image_l_x, image_r_x = int(np.floor((x_size - image_x_shape)/2)), int(np.ceil((x_size - image_x_shape)/2))
        image_l_y, image_r_y = int(np.floor((y_size - image_y_shape)/2)), int(np.ceil((y_size - image_y_shape)/2))
        return image_l_x, image_r_x, image_l_y, image_r_y

    x_size = max(source_shape[1], target_shape[1])
    y_size = max(source_shape[0], target_shape[0])
    source_padding = inner_pad(source_shape, x_size, y_size)
    target_padding = inner_pad(target_shape, x_size, y_size) 
    return source_padding, target_padding

def resample_to_min_size(image, output_min_size, gaussian=True):
    for i in range(image.shape[2]):
        temp_image = image[:, :, i]
        y_size, x_size = image.shape[0], image.shape[1]
        resample_ratio = min(image.shape[0:2]) / output_min_size
        if gaussian:
            temp_image = nd.gaussian_filter(temp_image, 1)
        resampled_image = resample(temp_image, int(x_size/resample_ratio), int(y_size/resample_ratio))
        if i == 0:
            output_image = np.empty(resampled_image.shape + (image.shape[2],), dtype=np.float32)
        output_image[:, :, i] = resampled_image
    return output_image

def resample_and_pad_single(image, output_max_size):
    output_shape = (output_max_size, output_max_size, 3)
    output_image = np.empty(output_shape, dtype=np.float32)
    for i in range(image.shape[2]):
        temp_image = image[:, :, i]
        y_size, x_size = image.shape[0], image.shape[1]
        resample_ratio = max(image.shape) / output_max_size
        temp_image = nd.gaussian_filter(temp_image, 1)
        resampled_image = au.resample(temp_image, int(x_size/resample_ratio), int(y_size/resample_ratio))
        y_size, x_size = resampled_image.shape[0], resampled_image.shape[1]
        padded_image = np.pad(resampled_image, (((int(np.floor((output_max_size - y_size)/2)), (int(np.ceil((output_max_size - y_size)/2)))), ((int(np.floor((output_max_size - x_size)/2))), (int(np.ceil((output_max_size - x_size)/2)))))), mode='constant', constant_values=1)
        output_image[:, :, i] = padded_image
    return output_image

def resample_tensor(tensor, new_size, device="cpu"):
    current_size = tensor.size()
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
    resampled_tensor = F.grid_sample(tensor, n_grid, mode='bilinear', padding_mode='zeros')
    return resampled_tensor

def compose_vector_fields_torch(u, v, device="cpu"):
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

    i_u_x = nn.functional.grid_sample(added_x, added_grid, padding_mode='border')
    i_u_y = nn.functional.grid_sample(added_y, added_grid, padding_mode='border')

    indexes = (added_grid[:, :, :, 0] >= 1.0) | (added_grid[:, :, :, 0] <= -1.0) | (added_grid[:, :, :, 1] >= 1.0) | (added_grid[:, :, :, 1] <= -1.0)
    indexes = indexes.view(indexes.size(0), 1, indexes.size(1), indexes.size(2))
    n_x = i_u_x - grid_x
    n_y = i_u_y - grid_y

    n_x[indexes] = 0.0
    n_y[indexes] = 0.0
    n_x = n_x / 2 * (x_size - 1)
    n_y = n_y / 2 * (y_size - 1)
    return torch.cat((n_x, n_y), dim=1)

def upsample_displacement_field_torch(displacement_field, new_size, device="cpu"):
    no_samples = new_size[0]
    old_x_size = displacement_field.size(3)
    old_y_size = displacement_field.size(2)
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
    resampled_displacement_field = F.grid_sample(displacement_field, n_grid, mode='bilinear', padding_mode='zeros')
    resampled_displacement_field[:, 0, :, :] *= x_size / old_x_size
    resampled_displacement_field[:, 1, :, :] *= y_size / old_y_size
    return resampled_displacement_field

def transform_tensors(tensors, displacement_fields, device="cpu"):
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
    transformed_patches = F.grid_sample(tensors, n_grid, mode='bilinear', padding_mode='zeros')
    return transformed_patches
