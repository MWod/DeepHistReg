import os

import numpy as np
import matplotlib.pyplot as plt
import torch

import cost_functions as cf
import utils


def rotation_alignment(source, target, params, device="cpu"):
    displacement_field = torch.zeros(2, source.size(0), source.size(1)).to(device)
    angle_step = params['angle_step']
    loss_single = cf.ncc_loss_global

    output_min_size = 512
    new_shape = utils.calculate_new_shape_min((source.size(0), source.size(1)), output_min_size)
    gaussian_kernel = utils.get_gaussian_kernel(7, 1, 1).to(device)
    smoothed_source = gaussian_kernel(source.view(1, 1, source.size(0), source.size(1)))[0, 0, : :]
    smoothed_target = gaussian_kernel(target.view(1, 1, source.size(0), source.size(1)))[0, 0, : :]
    resampled_source = utils.resample_tensor(smoothed_source, new_shape, device=device)
    resampled_target = utils.resample_tensor(smoothed_target, new_shape, device=device)
    y_size, x_size = resampled_source.size(0), resampled_source.size(1)
    com_x_source, com_y_source = utils.center_of_mass((resampled_source > 0.00001).float(), device=device)
    com_x_target, com_y_target = utils.center_of_mass((resampled_target > 0.00001).float(), device=device)
    initial_ncc = loss_single(resampled_source, resampled_target, device=device)
    identity_transform = np.array([
            [1, 0, 0.0],
            [0, 1, 0.0],
        ])
    identity_transform = utils.affine2theta(identity_transform, (y_size, x_size)).view(1, 2, 3).to(device)
    if initial_ncc < -0.85:
        displacement_field = utils.transform_to_displacement_field(resampled_source.view(1, 1, y_size, x_size), identity_transform, device=device)
        displacement_field = utils.upsample_displacement_field(displacement_field, (2, source.size(0), source.size(1)), device=device)
        return displacement_field

    centroid_transform = np.array([
        [1, 0, com_x_source - com_x_target],
        [0, 1, com_y_source - com_y_target],
    ])
    centroid_transform = utils.affine2theta(centroid_transform, (y_size, x_size)).view(1, 2, 3).to(device)
    translated_source = utils.tensor_affine_transform(resampled_source.view(1, 1, y_size, x_size), centroid_transform)[0, 0, :, :]

    centroid_ncc = loss_single(translated_source, resampled_target, device=device)
    if centroid_ncc < -0.85:
        displacement_field = utils.transform_to_displacement_field(resampled_source.view(1, 1, y_size, x_size), centroid_transform, device=device)
        displacement_field = utils.upsample_displacement_field(displacement_field, (2, source.size(0), source.size(1)), device=device)
        return displacement_field

    best_ncc = centroid_ncc
    found = False
    for i in range(0, 360, angle_step):
        transform = utils.compose_transforms(centroid_transform, utils.affine2theta(utils.generate_rotation_matrix(i, com_x_target, com_y_target), (y_size, x_size)), (y_size, x_size), device=device).view(1, 2, 3).to(device)
        transformed_source = utils.tensor_affine_transform(resampled_source.view(1, 1, y_size, x_size), transform)[0, 0, :, :]
        current_ncc = loss_single(transformed_source, resampled_target, device=device)
        if current_ncc < best_ncc:
            found = True
            best_ncc = current_ncc
            best_transform = transform
    if found:
        displacement_field = utils.transform_to_displacement_field(resampled_source.view(1, 1, y_size, x_size), best_transform, device=device)
    else:
        if centroid_ncc < initial_ncc:
            displacement_field = utils.transform_to_displacement_field(resampled_source.view(1, 1, y_size, x_size), centroid_transform, device=device)
        else:
            displacement_field = utils.transform_to_displacement_field(resampled_source.view(1, 1, y_size, x_size), identity_transform, device=device)
    displacement_field = utils.upsample_displacement_field(displacement_field, (2, source.size(0), source.size(1)), device=device)
    return displacement_field