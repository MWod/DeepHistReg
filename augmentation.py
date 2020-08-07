import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
import random

import utils

def randrange(vmin, vmax):
    return (random.random() * (vmax - vmin)) + vmin

def generate_random_rigid_transform(shape, **params):
    min_translation = params['min_translation']
    max_translation = params['max_translation']
    min_rotation = params['min_rotation']
    max_rotation = params['max_rotation']

    min_rotation = min_rotation * np.pi / 180
    max_rotation = max_rotation * np.pi / 180
    min_translation = min_translation*min(shape[0:2]) 
    max_translation = max_translation*max(shape[0:2])

    x_translation = randrange(min_translation, max_translation)
    y_translation = randrange(min_translation, max_translation)
    rotation = randrange(min_rotation, max_rotation)

    rigid_matrix = np.array([
        [np.cos(rotation), -np.sin(rotation), x_translation],
        [np.sin(rotation), np.cos(rotation), y_translation],
        [0, 0, 1],
    ])
    cm1 = np.array([
        [1, 0, ((shape[0] - 1) / 2)],
        [0, 1, ((shape[1] - 1) / 2)],
        [0, 0, 1], 
    ])
    cm2 = np.array([
        [1, 0, -((shape[0] - 1) / 2)],
        [0, 1, -((shape[1] - 1) / 2)],
        [0, 0, 1], 
    ])
    rigid_matrix = cm1 @ rigid_matrix @ cm2
    final_transform = rigid_matrix[0:2, :]
    return final_transform

def generate_random_affine_transform(shape, **params):
    min_translation = params['min_translation']
    max_translation = params['max_translation']
    min_rotation = params['min_rotation']
    max_rotation = params['max_rotation']
    min_shear = params['min_shear']
    max_shear = params['max_shear']
    min_scale = params['min_scale']
    max_scale = params['max_scale']

    min_rotation = min_rotation * np.pi / 180
    max_rotation = max_rotation * np.pi / 180
    min_translation = min_translation*min(shape[0:2]) 
    max_translation = max_translation*max(shape[0:2])

    x_translation = randrange(min_translation, max_translation)
    y_translation = randrange(min_translation, max_translation)
    rotation = randrange(min_rotation, max_rotation)

    x_shear = randrange(min_shear, max_shear)
    y_shear = randrange(min_shear, max_shear)

    x_scale = randrange(min_scale, max_scale)
    y_scale = randrange(min_scale, max_scale)

    rigid_matrix = np.array([
        [np.cos(rotation), -np.sin(rotation), x_translation],
        [np.sin(rotation), np.cos(rotation), y_translation],
        [0, 0, 1],
    ])
    cm1 = np.array([
        [1, 0, ((shape[0] - 1) / 2)],
        [0, 1, ((shape[1] - 1) / 2)],
        [0, 0, 1], 
    ])
    cm2 = np.array([
        [1, 0, -((shape[0] - 1) / 2)],
        [0, 1, -((shape[1] - 1) / 2)],
        [0, 0, 1], 
    ])
    rigid_matrix = cm1 @ rigid_matrix @ cm2

    shear_matrix = np.array([
        [1, x_shear, 0],
        [y_shear, 1, 0],
        [0, 0, 1],
    ])
    scale_matrix = np.array([
        [x_scale, 0, 0],
        [0, y_scale, 0],
        [0, 0, 1],
    ])

    all_matrices = [rigid_matrix, shear_matrix, scale_matrix]
    random.shuffle(all_matrices)
    transform = np.eye(3)
    for i in range(len(all_matrices)):
        transform = transform @ all_matrices[i]
    final_transform = transform[0:2, :]
    return final_transform

def affine_augmentation(affine_generation_params, augment_both=True):
    def augmentation(source, target):
        transform = generate_random_affine_transform(source.shape, **affine_generation_params)
        if augment_both:
            if random.random() > 0.5:
                transformed_source = utils.numpy_affine_transform(source, transform)
                transformed_target = target
            else:
                transformed_source = source
                transformed_target = utils.numpy_affine_transform(target, transform)
                transform = utils.numpy_inv_transform(transform)
        else:
            transformed_source = utils.numpy_affine_transform(source, transform)
            transformed_target = target
        return transformed_source, transformed_target, utils.numpy_inv_transform(transform)
    return augmentation

def rigid_augmentation(rigid_generation_params, augment_both=True):
    def augmentation(source, target):
        transform = generate_random_rigid_transform(source.shape, **rigid_generation_params)
        if augment_both:
            if random.random() > 0.5:
                transformed_source = utils.numpy_affine_transform(source, transform)
                transformed_target = target
            else:
                transformed_source = source
                transformed_target = utils.numpy_affine_transform(target, transform)
                transform = utils.numpy_inv_transform(transform)
        else:
            transformed_source = utils.numpy_affine_transform(source, transform)
            transformed_target = target
        return transformed_source, transformed_target, utils.numpy_inv_transform(transform)
    return augmentation