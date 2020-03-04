import os
import numpy as np
import time
from skimage import filters
from skimage import morphology
from skimage import util

import anhir_utils as au

def image_entropy(image):
    return filters.rank.entropy(util.img_as_ubyte(image), morphology.disk(3))

def histogram_correction(source, target):
    oldshape = source.shape
    source = source.ravel()
    target = target.ravel()
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(target, return_counts=True)
    sq = np.cumsum(s_counts).astype(np.float64)
    sq /= sq[-1]
    tq = np.cumsum(t_counts).astype(np.float64)
    tq /= tq[-1]
    interp_t_values = np.interp(sq, tq, t_values)
    return interp_t_values[bin_idx].reshape(oldshape)

def pad_image(image, x_size, y_size, constant_values=0):
    image_y_shape, image_x_shape = np.shape(image)
    image_l_x, image_r_x = int(np.floor((x_size - image_x_shape)/2)), int(np.ceil((x_size - image_x_shape)/2))
    image_l_y, image_r_y = int(np.floor((y_size - image_y_shape)/2)), int(np.ceil((y_size - image_y_shape)/2))
    image = np.pad(image, [(image_l_y, image_r_y), (image_l_x, image_r_x)], mode='constant', constant_values=constant_values)
    return image, image_l_x, image_r_x, image_l_y, image_r_y

def preprocess_for_rotation_alignment(source, target):
    s_y_size, s_x_size = source.shape
    t_y_size, t_x_size = target.shape
    e_resample_ratio = np.max(np.array([s_y_size, s_x_size, t_y_size, t_x_size])) / 512
    e_source, e_target = au.resample_both(source, target, e_resample_ratio)
    e_source, e_target = au.normalize(e_source), au.normalize(e_target)
    source_entropy = image_entropy(e_source)
    target_entropy = image_entropy(e_target)
    if np.mean(target_entropy) > np.mean(source_entropy):
        source = histogram_correction(source, target)
    else:
        target = histogram_correction(target, source)
    return source, target

def preprocess_for_parsing(source, target, echo=True):    
    source = au.normalize(source)
    target = au.normalize(target)
    x_size = max(source.shape[1], target.shape[1])
    y_size = max(source.shape[0], target.shape[0])
    source, source_l_x, source_r_x, source_l_y, source_r_y = pad_image(source, x_size, y_size, constant_values=1)
    target, target_l_x, target_r_x, target_l_y, target_r_y = pad_image(target, x_size, y_size, constant_values=1)
    return source, target

def prepare_for_initial_alignment(source, target, initial_resample_size, gaussian_divider, output_min_size, gaussian=True):
    org_source_shape = source.shape
    org_target_shape = target.shape
    initial_resample_ratio = au.calculate_resample_size(source, target, initial_resample_size)
    _, _, channels = source.shape
    for i in range(channels):
        source[:, :, i] = au.gaussian_filter(source[:, :, i], initial_resample_ratio / gaussian_divider)
        target[:, :, i] = au.gaussian_filter(target[:, :, i], initial_resample_ratio / gaussian_divider)

    for i in range(channels):
        temp_source, temp_target = au.resample_both(source[:, :, i], target[:, :, i], initial_resample_ratio)
        temp_source, temp_target = preprocess_for_parsing(temp_source, temp_target)
        if i == 0:
            y_size, x_size = temp_source.shape
            result_source = np.empty((y_size, x_size, 3))
            result_target = np.empty((y_size, x_size, 3))
        result_source[:, :, i] = temp_source
        result_target[:, :, i] = temp_target 
        
    org_source = au.normalize(result_source.astype(np.float64))
    org_source = au.resample_to_min_size(result_source, output_min_size, gaussian=gaussian)
    org_target = au.normalize(result_target.astype(np.float64))
    org_target = au.resample_to_min_size(result_target, output_min_size, gaussian=gaussian)
    seg_source = au.tissue_segmentation((org_source * 255).astype(np.uint8))
    seg_target = au.tissue_segmentation((org_target * 255).astype(np.uint8))
    seg_source = au.normalize(seg_source.astype(np.float32))
    seg_target = au.normalize(seg_target.astype(np.float32))
    return org_source, org_target, seg_source, seg_target, initial_resample_ratio, org_source_shape, org_target_shape