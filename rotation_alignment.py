import os
import time

import numpy as np
import matplotlib.pyplot as plt
from skimage import color
import scipy.ndimage as nd
import cv2 as cv
from skimage import filters
from skimage import morphology
from skimage import util as skut

import anhir_utils as au
import preprocessing as pre
import cost_functions as cf


def iterative_search(source, target):
    com_source = nd.center_of_mass(source)
    com = nd.center_of_mass(target)
    source, target = pre.preprocess_for_rotation_alignment(source, target) 
    angle_step = 2

    best_transform = np.array([
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=np.float32)

    centroid_transform = np.array([
        [1, 0, com_source[1] - com[1]],
        [0, 1, com_source[0] - com[0]],
    ])
    translated_source = au.affine_transform(source, centroid_transform)
    mask = np.ones(source.shape)

    best_ncc = cf.ncc(translated_source, target)
    failed = False
    if best_ncc < -0.75:
        return best_transform, failed
    initial_ncc = best_ncc

    found = False
    failed = True
    for i in range(0, 360, angle_step):
        transform = au.generate_rotation_matrix(i, com[1], com[0])
        transformed_source = au.affine_transform(translated_source, transform)
        transformed_mask = au.affine_transform(mask, transform)
        c_size = np.sum(transformed_mask)/np.size(mask)
        current_ncc = cf.ncc(transformed_source, target)
        if current_ncc < best_ncc:
            found = True
            failed = False
            best_ncc = current_ncc
            best_transform = transform
    if found:
        return_transform = au.compose_transform(centroid_transform, best_transform)
    else:
        return_transform = best_transform
    return return_transform, failed

def rotation_alignment(source, target, output_min_size=512):
    r_source = au.normalize(source.astype(np.float32))
    r_target = au.normalize(target.astype(np.float32))
    if min(r_source.shape[0:2]) != output_min_size:
        r_source = au.resample_to_min_size(r_source, output_min_size, gaussian=False)
        r_target = au.resample_to_min_size(r_target, output_min_size, gaussian=False)
    t_source = au.tissue_segmentation((r_source*255).astype(np.uint8))
    t_target = au.tissue_segmentation((r_target*255).astype(np.uint8))
    t_source = (1 - color.rgb2gray(au.normalize(t_source.astype(np.float32))))
    t_target = (1 - color.rgb2gray(au.normalize(t_target.astype(np.float32))))
    transform, failed = iterative_search(t_source, t_target)
    if failed == True:
        transform, _ = iterative_search(1 - color.rgb2gray(r_source), 1 - color.rgb2gray(r_target))
    i_u_x, i_u_y = au.transform_to_deformation_field(t_source, transform)
    i_u_x, i_u_y = au.resample_displacement_field(i_u_x, i_u_y, source.shape[1], source.shape[0])
    return i_u_x, i_u_y

