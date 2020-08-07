import os
import time

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

import utils
import paths

import deephistreg as dhr 

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_path = paths.parsed_data_path

### Identity - to create results without any registration (useful for creating the baseline)
# output_path = None # TO DEFINE
# dhr_params = dict()
# dhr_params['segmentation_mode'] = "deep_segmentation"
# dhr_params['initial_rotation'] = False
# dhr_params['affine_registration'] = False
# dhr_params['nonrigid_registration'] = False
# segmentation_params = dict()
# segmentation_params['model_path'] =  None # Path to segmentation model
# dhr_params['segmentation_params'] = segmentation_params
# load_masks = False
###

### Seg + Rotation Params - to create results from the initial rotation (useful for creating the affine training dataset)
# output_path = None # TO DEFINE
# dhr_params = dict()
# dhr_params['segmentation_mode'] = "deep_segmentation"
# dhr_params['initial_rotation'] = True
# dhr_params['affine_registration'] = False
# dhr_params['nonrigid_registration'] = False
# initial_rotation_params = dict()
# initial_rotation_params['angle_step'] = 1
# dhr_params['initial_rotation_params'] = initial_rotation_params
# segmentation_params = dict()
# segmentation_params['model_path'] =  None # Path to segmentation model
# dhr_params['segmentation_params'] = segmentation_params
# load_masks = False
###

### Seg + Rotation Params + Affine - to create results from the initial rotation + affine registration (useful for creating the nonrigid training dataset)
# output_path = None # TO DEFINE
# dhr_params = dict()
# dhr_params['segmentation_mode'] = "deep_segmentation"
# dhr_params['initial_rotation'] = True
# dhr_params['affine_registration'] = True
# dhr_params['nonrigid_registration'] = False
# initial_rotation_params = dict()
# initial_rotation_params['angle_step'] = 1
# dhr_params['initial_rotation_params'] = initial_rotation_params
# affine_registration_params = dict()
# models_path = paths.models_path
# affine_registration_params['model_path'] = None # TO DEFINE
# affine_registration_params['affine_type'] = "simple"
# dhr_params['affine_registration_params'] = affine_registration_params
# segmentation_params = dict()
# segmentation_params['model_path'] =  None # Path to segmentation model
# dhr_params['segmentation_params'] = segmentation_params
# load_masks = False
###

### Seg + Rotation Params + Affine + Nonrigid - to create final nonrigid results
output_path = None # TO DEFINE
dhr_params = dict()
dhr_params['segmentation_mode'] = "deep_segmentation"
dhr_params['initial_rotation'] = True
dhr_params['affine_registration'] = True
dhr_params['nonrigid_registration'] = True
initial_rotation_params = dict()
initial_rotation_params['angle_step'] = 1
dhr_params['initial_rotation_params'] = initial_rotation_params
affine_registration_params = dict()
models_path = None # TO DEFINE
affine_registration_params['model_path'] = None # TO DEFINE
affine_registration_params['affine_type'] = "simple"
dhr_params['affine_registration_params'] = affine_registration_params
nonrigid_registration_params = dict() # Params used during training
nonrigid_registration_params['stride'] = 128
nonrigid_registration_params['patch_size'] = (256, 256)
nonrigid_registration_params['number_of_patches'] = 32
nonrigid_registration_params['num_levels'] = 3
nonrigid_registration_params['inner_iterations_per_level'] = [3, 3, 3]
nonrigid_registration_params['model_path'] = None # TO DEFINE
dhr_params['nonrigid_registration_params'] = nonrigid_registration_params
segmentation_params = dict()
segmentation_params['model_path'] =  None # TO DEFINE
dhr_params['segmentation_params'] = segmentation_params
load_masks = False
###

def run():
    show = False
    ids = range(0, 481)
    for current_id in ids:
        b_loading = time.time()
        current_pair = str(current_id)
        if load_masks:
            source, target, source_landmarks, target_landmarks, status, source_mask, target_mask = utils.load_pair(current_pair, dataset_path, load_masks=load_masks)
        else:
            source, target, source_landmarks, target_landmarks, status = utils.load_pair(current_pair, dataset_path, load_masks=load_masks)
        print("Current pair: ", current_pair)
        print("Status: ", status)
        source = torch.from_numpy(source).to(device)
        target = torch.from_numpy(target).to(device)
        if load_masks:
            source_mask = torch.from_numpy(source_mask).to(device)
            target_mask = torch.from_numpy(target_mask).to(device)
            dhr_params['segmentation_params']['source_mask'] = target_mask
            dhr_params['segmentation_params']['target_mask'] = source_mask
        e_loading = time.time()
        loading_time = e_loading - b_loading
        print("Time for loading and memory transfer: ", loading_time)
        b_registration = time.time()
        target, source, transformed_target, displacement_field, _ = dhr.deephistreg(target, source, device, dhr_params)
        e_registration = time.time()
        registration_time = e_registration - b_registration
        print("Time for registration: ", registration_time)

        if show:
            plt.figure()
            plt.subplot(2, 3, 1)
            plt.imshow(source.cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.subplot(2, 3, 2)
            plt.imshow(transformed_target.cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.subplot(2, 3, 3)
            plt.imshow(target.cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.subplot(2, 3, 4)
            plt.imshow(displacement_field[0, :, :].cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.subplot(2, 3, 5)
            plt.imshow(displacement_field[1, :, :].cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.show()

        transformed_source_landmarks = utils.transform_landmarks(source_landmarks, displacement_field)

        source_save_path = os.path.join(output_path, current_pair, "source.mha")
        target_save_path = os.path.join(output_path, current_pair, "target.mha")
        transformed_target_save_path = os.path.join(output_path, current_pair, "transformed_target.mha")
        source_landmarks_path = os.path.join(output_path, current_pair, "source_landmarks.csv")
        transformed_source_landmarks_path = os.path.join(output_path, current_pair, "transformed_source_landmarks.csv")
        if status == "training":
            target_landmarks_path = os.path.join(output_path, current_pair, "target_landmarks.csv")

        if not os.path.isdir(os.path.dirname(source_save_path)):
            os.makedirs(os.path.dirname(source_save_path))

        sitk.WriteImage(sitk.GetImageFromArray(source.cpu().numpy()), source_save_path)
        sitk.WriteImage(sitk.GetImageFromArray(target.cpu().numpy()), target_save_path)
        sitk.WriteImage(sitk.GetImageFromArray(transformed_target.cpu().numpy()), transformed_target_save_path)
        utils.save_landmarks(source_landmarks, source_landmarks_path)
        utils.save_landmarks(transformed_source_landmarks, transformed_source_landmarks_path)
        if status == "training":
            utils.save_landmarks(target_landmarks, target_landmarks_path)
            try:
                image_diagonal = np.sqrt(source.shape[0]**2 + source.shape[1]**2)
                rtre_initial = utils.calculate_rtre(source_landmarks, target_landmarks, image_diagonal)
                rtre_final = utils.calculate_rtre(transformed_source_landmarks, target_landmarks, image_diagonal)
                string_to_save = "Initial TRE: " + str(np.median(rtre_initial)) + "\n" + "Resulting TRE: " + str(np.median(rtre_final))
                txt_path = os.path.join(output_path, current_pair, "tre.txt")
                with open(txt_path, "w") as file:
                    file.write(string_to_save)
            except:
                pass
        time_to_save = str(registration_time + loading_time)
        time_txt_path = os.path.join(output_path, current_pair, "time.txt")
        with open(time_txt_path, "w") as file:
            file.write(time_to_save)


if __name__ == "__main__":
    run()