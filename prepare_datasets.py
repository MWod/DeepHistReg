import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.ndimage as nd
import SimpleITK as sitk
import skimage.color as color

import torch
import torch.nn.functional as F

import deephistreg as dhr
import utils
import paths

def parse_dataset(csv_path, dataset_path, output_path, masks_path=None):
    output_max_size = 4096
    show = False

    csv_file = pd.read_csv(csv_path)
    for current_case in csv_file.iterrows():
        current_id = current_case[1]['Unnamed: 0']
        size = current_case[1]['Image size [pixels]']
        diagonal = int(current_case[1]['Image diagonal [pixels]'])
        y_size, x_size = int(size.split(",")[0][1:]), int(size.split(",")[1][:-1])
        source_path = current_case[1]['Source image']
        target_path = current_case[1]['Target image']
        source_landmarks_path = current_case[1]['Source landmarks']
        target_landmarks_path = current_case[1]['Target landmarks']
        status = current_case[1]['status']

        extension = source_path[-4:]
        if masks_path is not None:
            source_mask_path = os.path.join(masks_path, source_path.replace(extension, ".mha"))
            target_mask_path = os.path.join(masks_path, target_path.replace(extension, ".mha"))

        source_path = os.path.join(dataset_path, source_path.replace(extension, ".mha"))
        target_path = os.path.join(dataset_path, target_path.replace(extension, ".mha"))
        source_landmarks_path = os.path.join(dataset_path, source_landmarks_path)
        target_landmarks_path = os.path.join(dataset_path, target_landmarks_path)

        source_landmarks = utils.load_landmarks(source_landmarks_path)
        if status == "training":
            target_landmarks = utils.load_landmarks(target_landmarks_path)

        source = color.rgb2gray(sitk.GetArrayFromImage(sitk.ReadImage(source_path)))
        target = color.rgb2gray(sitk.GetArrayFromImage(sitk.ReadImage(target_path)))

        if masks_path is not None:
            source_mask = sitk.GetArrayFromImage(sitk.ReadImage(source_mask_path)).astype(np.ubyte)
            target_mask = sitk.GetArrayFromImage(sitk.ReadImage(target_mask_path)).astype(np.ubyte)
            source_mask = (source_mask / np.max(source_mask))
            target_mask = (target_mask / np.max(target_mask))

        source = 1 - utils.normalize(source)
        target = 1 - utils.normalize(target)

        padded_source, padded_target = utils.pad_images_np(source, target)
        padded_source_landmarks = utils.pad_landmarks(source_landmarks, source.shape, padded_source.shape)
        if status == "training":
            padded_target_landmarks = utils.pad_landmarks(target_landmarks, target.shape, padded_target.shape)

        if masks_path is not None:
            padded_source_mask, padded_target_mask = utils.pad_images_np(source_mask, target_mask)

        resample_factor = np.max(padded_source.shape) / output_max_size
        gaussian_sigma = resample_factor / 1.25

        smoothed_source = nd.gaussian_filter(padded_source, gaussian_sigma)
        smoothed_target = nd.gaussian_filter(padded_target, gaussian_sigma)

        resampled_source = utils.resample_image(smoothed_source, resample_factor)
        resampled_target = utils.resample_image(smoothed_target, resample_factor)
        resampled_source_landmarks = utils.resample_landmarks(padded_source_landmarks, resample_factor)
        if status == "training":
            resampled_target_landmarks = utils.resample_landmarks(padded_target_landmarks, resample_factor)

        if masks_path is not None:
            resampled_source_mask = (utils.resample_image(padded_source_mask, resample_factor) > 0.5).astype(np.ubyte)
            resampled_target_mask = (utils.resample_image(padded_target_mask, resample_factor) > 0.5).astype(np.ubyte)

        print("Current ID: ", current_id)
        print("Y Size, X_size: ", y_size, x_size)
        print("Source path: ", source_path)
        print("Target path: ", target_path)
        print("Source shape: ", source.shape)
        print("Target shape: ", target.shape)
        print("Padded source shape: ", padded_source.shape)
        print("Padded target shape: ", padded_target.shape)
        print("Resampled source shape: ", resampled_source.shape)
        print("Resampled target shape: ", resampled_target.shape)
        print("Source landmarks path: ", source_landmarks_path)
        print("Target landmarks path: ", target_landmarks_path)
        print("Source landmarks shape: ", source_landmarks.shape)
        print("Status: ", status)
        print("Resample factor: ", resample_factor)
        if status == "training":
            print("Target landmarks shape: ", target_landmarks.shape)
            try:
                print("Initial Median TRE: ", np.median(utils.calculate_tre(source_landmarks, target_landmarks)))
                print("Resampled Median TRE: ", np.median(utils.calculate_tre(resampled_source_landmarks, resampled_target_landmarks)))
            except:
                print("Unequal number of landmarks.")
        print()

        to_save_source_mha = sitk.GetImageFromArray((utils.normalize(resampled_source).astype(np.float32)))
        to_save_target_mha = sitk.GetImageFromArray((utils.normalize(resampled_target).astype(np.float32)))
        to_save_source_jpg = sitk.GetImageFromArray(((utils.normalize(resampled_source).astype(np.float32))*255).astype(np.ubyte))
        to_save_target_jpg = sitk.GetImageFromArray(((utils.normalize(resampled_target).astype(np.float32))*255).astype(np.ubyte))
        to_save_source_landmarks = resampled_source_landmarks.astype(np.float32)
        if status == "training":
            to_save_target_landmarks = resampled_target_landmarks.astype(np.float32)

        if masks_path is not None:
            to_save_source_mask_mha = sitk.GetImageFromArray(resampled_source_mask.astype(np.ubyte))
            to_save_target_mask_mha = sitk.GetImageFromArray(resampled_target_mask.astype(np.ubyte))
            to_save_source_mask_jpg = sitk.GetImageFromArray((resampled_source_mask*255).astype(np.ubyte))
            to_save_target_mask_jpg = sitk.GetImageFromArray((resampled_target_mask*255).astype(np.ubyte))

        to_save_source_mha_path = os.path.join(output_path, str(current_id), "source.mha")
        to_save_target_mha_path = os.path.join(output_path, str(current_id), "target.mha")
        to_save_source_jpg_path = os.path.join(output_path, str(current_id), "source.jpg")
        to_save_target_jpg_path = os.path.join(output_path, str(current_id), "target.jpg")
        to_save_source_landmarks_path = os.path.join(output_path, str(current_id), "source_landmarks.csv")
        if status == "training":
            to_save_target_landmarks_path = os.path.join(output_path, str(current_id), "target_landmarks.csv")

        if masks_path is not None:
            to_save_source_mask_mha_path = os.path.join(output_path, str(current_id), "source_mask.mha")
            to_save_target_mask_mha_path = os.path.join(output_path, str(current_id), "target_mask.mha")
            to_save_source_mask_jpg_path = os.path.join(output_path, str(current_id), "source_mask.jpg")
            to_save_target_mask_jpg_path = os.path.join(output_path, str(current_id), "target_mask.jpg")

        if not os.path.isdir(os.path.dirname(to_save_source_mha_path)):
            os.makedirs(os.path.dirname(to_save_source_mha_path))

        sitk.WriteImage(to_save_source_mha, to_save_source_mha_path)
        sitk.WriteImage(to_save_target_mha, to_save_target_mha_path)
        sitk.WriteImage(to_save_source_jpg, to_save_source_jpg_path)
        sitk.WriteImage(to_save_target_jpg, to_save_target_jpg_path)
        utils.save_landmarks(to_save_source_landmarks, to_save_source_landmarks_path)
        if status == "training":
            utils.save_landmarks(to_save_target_landmarks, to_save_target_landmarks_path) 

        if masks_path is not None:
            sitk.WriteImage(to_save_source_mask_mha, to_save_source_mask_mha_path)
            sitk.WriteImage(to_save_target_mask_mha, to_save_target_mask_mha_path)
            sitk.WriteImage(to_save_source_mask_jpg, to_save_source_mask_jpg_path)
            sitk.WriteImage(to_save_target_mask_jpg, to_save_target_mask_jpg_path)

        if show:
            plt.figure()
            no_rows = 1 if masks_path is None else 2
            plt.subplot(no_rows, 2, 1)
            plt.imshow(source, cmap='gray')
            plt.plot(source_landmarks[:, 0], source_landmarks[:, 1], "r*")
            plt.title("Source")
            plt.subplot(no_rows, 2, 2)
            plt.imshow(target, cmap='gray')
            if status == "training":
                plt.plot(target_landmarks[:, 0], target_landmarks[:, 1], "r*")
            plt.title("Target")
            if masks_path is not None:
                plt.subplot(no_rows, 2, 3)
                plt.imshow(source_mask, cmap='gray')
                plt.title("Source Mask")
                plt.subplot(no_rows, 2, 4)
                plt.imshow(target_mask, cmap='gray')
                plt.title("Target Mask")

            plt.figure()
            plt.subplot(no_rows, 2, 1)
            plt.imshow(padded_source, cmap='gray')
            plt.plot(padded_source_landmarks[:, 0], padded_source_landmarks[:, 1], "r*")
            plt.title("Padded Source")
            plt.subplot(no_rows, 2, 2)
            plt.imshow(padded_target, cmap='gray')
            if status == "training":
                plt.plot(padded_target_landmarks[:, 0], padded_target_landmarks[:, 1], "r*")
            plt.title("Padded Target")
            if masks_path is not None:
                plt.subplot(no_rows, 2, 3)
                plt.imshow(padded_source_mask, cmap='gray')
                plt.title("Padded Source Mask")
                plt.subplot(no_rows, 2, 4)
                plt.imshow(padded_target_mask, cmap='gray')
                plt.title("Padded Target Mask")

            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(smoothed_source, cmap='gray')
            plt.plot(padded_source_landmarks[:, 0], padded_source_landmarks[:, 1], "r*")
            plt.title("Smoothed Source")
            plt.subplot(1, 2, 2)
            plt.imshow(smoothed_target, cmap='gray')
            if status == "training":
                plt.plot(padded_target_landmarks[:, 0], padded_target_landmarks[:, 1], "r*")
            plt.title("Smoothed Target")

            plt.figure()
            plt.subplot(no_rows, 2, 1)
            plt.imshow(resampled_source, cmap='gray')
            plt.plot(resampled_source_landmarks[:, 0], resampled_source_landmarks[:, 1], "r*")
            plt.title("Resampled Source")
            plt.subplot(no_rows, 2, 2)
            plt.imshow(resampled_target, cmap='gray')
            if status == "training":
                plt.plot(resampled_target_landmarks[:, 0], resampled_target_landmarks[:, 1], "r*")
            plt.title("Resampled Target")
            if masks_path is not None:
                plt.subplot(no_rows, 2, 3)
                plt.imshow(resampled_source_mask, cmap='gray')
                plt.title("Resampled Source Mask")
                plt.subplot(no_rows, 2, 4)
                plt.imshow(resampled_target_mask, cmap='gray')
                plt.title("Resampled Target Mask")
            plt.show()


def create_training_dataset(results_path, dataset_path, size, mode='min'):
    show = False
    ids = range(0, 481)
    for current_id in ids:
        current_id = str(current_id)
        case_path = os.path.join(results_path, current_id)
        transformed_target_path = os.path.join(case_path, "transformed_target.mha")
        source_path = os.path.join(case_path, "source.mha")

        transformed_target = sitk.GetArrayFromImage(sitk.ReadImage(transformed_target_path))
        source = sitk.GetArrayFromImage(sitk.ReadImage(source_path))

        t_source = torch.from_numpy(source)
        t_target = torch.from_numpy(transformed_target)
        if mode == 'min':
            new_shape = utils.calculate_new_shape_min((t_source.size(0), t_source.size(1)), size)
            if min(new_shape) == min(source.shape):
                print("Resampling not required")
                resampled_source = t_source
                resampled_target = t_target
            else:
                resampled_source = utils.resample_tensor(t_source, new_shape)
                resampled_target = utils.resample_tensor(t_target, new_shape)
        elif mode == 'max':
            new_shape = utils.calculate_new_shape_max((t_source.size(0), t_source.size(1)), size)
            if max(new_shape) == max(source.shape):
                print("Resampling not required")
                resampled_source = t_source
                resampled_target = t_target
            else:
                resampled_source = utils.resample_tensor(t_source, new_shape)
                resampled_target = utils.resample_tensor(t_target, new_shape)

        transformed_target_resampled = resampled_target.numpy()
        source_resampled = resampled_source.numpy()

        target_landmarks_path = os.path.join(case_path, "target_landmarks.csv")
        try:
            target_landmarks = utils.load_landmarks(target_landmarks_path)
            status = "training"
        except:
            status = "evaluation"

        print("Current ID: ", current_id)
        print("Transformed target shape: ", transformed_target.shape)
        print("Source shape: ", source.shape)
        print("Resampled target shape: ", transformed_target_resampled.shape)
        print("Resampled source shape: ", source_resampled.shape)

        if show:
            plt.figure()
            plt.subplot(2, 2, 1)
            plt.imshow(source, cmap='gray')
            plt.axis('off')
            plt.subplot(2, 2, 2)
            plt.imshow(transformed_target, cmap='gray')
            plt.axis('off')
            plt.subplot(2, 2, 3)
            plt.imshow(source_resampled, cmap='gray')
            plt.axis('off')
            plt.subplot(2, 2, 4)
            plt.imshow(transformed_target_resampled, cmap='gray')
            plt.axis('off')
            plt.show()

        to_save_source = sitk.GetImageFromArray(transformed_target_resampled)
        to_save_target = sitk.GetImageFromArray(source_resampled)

        output_path = os.path.join(dataset_path, status, current_id)
        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        source_output_path = os.path.join(output_path, "source.mha")
        target_output_path = os.path.join(output_path, "target.mha")
        sitk.WriteImage(to_save_source, source_output_path)
        sitk.WriteImage(to_save_target, target_output_path)






if __name__ == "__main__":
    csv_path = paths.csv_path
    dataset_path = paths.original_data_path
    output_path = paths.parsed_data_path
    parse_dataset(csv_path, dataset_path, output_path, masks_path=None)

    # The purpose of the code below is to create training dataset for the next registration step (e.g. from rotation alignment to affine or from affine to nonrigid)
    # results_path = None
    # dataset_path = None
    # size = 1024 # Max shape in the training dataset (useful for e.g. decreasing the resolution for initial rotation search or affine registration)
    # create_training_dataset(results_path, dataset_path, size, "min")







