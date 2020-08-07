import os
import time

import numpy as np
import matplotlib.pyplot as plt

import torch

import deep_segmentation as ds
import rotation_alignment as ra
import affine_registration as ar
import deformable_registration as nr
import utils

from networks import segmentation_network as sn
from networks import affine_network_attention as an
from networks import affine_network_simple as asimple 
from networks import nonrigid_registration_network as nrn

def deephistreg(source, target, device, params):
    result_dict = dict()
    b_total_time = time.time()

    segmentation_mode = params['segmentation_mode']
    if segmentation_mode == "deep_segmentation":
        segmentation_params = params['segmentation_params']
        seg_model_path = segmentation_params['model_path']
        seg_model = sn.load_network(device, path=seg_model_path)
    initial_rotation = params['initial_rotation']
    if initial_rotation:
        initial_rotation_params = params['initial_rotation_params']
    affine_registration = params['affine_registration']
    if affine_registration:
        affine_registration_params = params['affine_registration_params']
        affine_model_path = affine_registration_params['model_path']
        affine_type = affine_registration_params['affine_type']
        if affine_type == "attention":
            affine_model = an.load_network(device, path=affine_model_path)
        elif affine_type == "simple":
            affine_model = asimple.load_network(device, path=affine_model_path)
    nonrigid_registration = params['nonrigid_registration']
    if nonrigid_registration:
        nonrigid_registration_params = params['nonrigid_registration_params']
        nonrigid_model_path = nonrigid_registration_params['model_path']
        num_levels = nonrigid_registration_params['num_levels']
        nonrigid_models = list()
        for i in range(num_levels):
            current_path = nonrigid_model_path + "_level_" + str(i+1)
            nonrigid_models.append(nrn.load_network(device, path=current_path))

    displacement_field = torch.zeros(2, source.size(0), source.size(1)).to(device)
    # Tissue segmentation
    b_seg_time = time.time()

    if segmentation_mode == "deep_segmentation":
        source_mask, target_mask = ds.segmentation(source, target, seg_model, device=device)
        source[source_mask == 0] = 0
        target[target_mask == 0] = 0
    elif segmentation_mode == "manual":
        segmentation_params = params['segmentation_params']
        source_mask, target_mask = segmentation_params['source_mask'], segmentation_params['target_mask']
        source[source_mask == 0] = 0 
        target[target_mask == 0] = 0
    elif segmentation_mode is None:
        source, target = source, target
    else:
        raise ValueError("Unsupported segmentation mode.")
    e_seg_time = time.time()

    warped_source = source.clone()
    # Rotation alignment
    b_rot_time = time.time()
    if initial_rotation:
        if segmentation_mode is not None:
            if torch.sum(source_mask) >= 0.99*source.size(0)*source.size(1):
                pass
            else:
                rot_displacement_field = ra.rotation_alignment(warped_source, target, initial_rotation_params, device=device)
                displacement_field = utils.compose_displacement_field(displacement_field, rot_displacement_field, device=device, delete_outliers=False)
                warped_source = utils.warp_tensor(source, displacement_field, device=device)
        else:
            rot_displacement_field = ra.rotation_alignment(warped_source, target, initial_rotation_params, device=device)
            displacement_field = utils.compose_displacement_field(displacement_field, rot_displacement_field, device=device, delete_outliers=False)
            warped_source = utils.warp_tensor(source, displacement_field, device=device)
    else:
        pass
    e_rot_time = time.time()
    
    # Affine registration
    b_aff_time = time.time()
    if affine_registration:
        affine_displacement_field = ar.affine_registration(warped_source, target, affine_model, device=device)
        displacement_field = utils.compose_displacement_field(displacement_field, affine_displacement_field, device=device, delete_outliers=False)
        warped_source = utils.warp_tensor(source, displacement_field, device=device)
    else:
        pass
    e_aff_time = time.time()
    
    # Nonrigid registration
    b_nr_time = time.time()
    if nonrigid_registration:
        nonrigid_displacement_field = nr.nonrigid_registration(warped_source, target, nonrigid_models, nonrigid_registration_params, device=device)
        displacement_field = utils.compose_displacement_field(displacement_field, nonrigid_displacement_field, device=device, delete_outliers=False)
        warped_source = utils.warp_tensor(source, displacement_field, device=device)
    else:
        pass
    e_nr_time = time.time()

    e_total_time = time.time()
    result_dict['total_time'] = e_total_time - b_total_time
    result_dict['seg_time'] = e_seg_time - b_seg_time
    result_dict['rot_time'] = e_rot_time - b_rot_time
    result_dict['aff_time'] = e_aff_time - b_aff_time
    result_dict['nr_time'] = e_nr_time - b_nr_time
    return source, target, warped_source, displacement_field, result_dict

