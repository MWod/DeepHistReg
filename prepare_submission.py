import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
import paths
import utils

original_data_path = paths.original_data_path 
csv_path = paths.csv_path

results_path = None # Path to the results (from main.py)
submission_path = None # Path where submission should be saved


def prepare_submission():
    if not os.path.isdir(submission_path):
        os.makedirs(submission_path)
    output_csv_path = os.path.join(submission_path, "registration-results.csv")
    prepare_output_csv(csv_path, output_csv_path)

    ids = range(0, 481)
    for current_id in ids:
        current_id = str(current_id)
        case_path = os.path.join(submission_path, current_id)
        if not os.path.isdir(case_path):
            os.makedirs(case_path)

        source_landmarks_path = os.path.join(results_path, current_id, "source_landmarks.csv")
        transformed_source_landmarks_path = os.path.join(results_path, current_id, "transformed_source_landmarks.csv")
        target_landmarks_path = os.path.join(results_path, current_id, "target_landmarks.csv")
        time_path = os.path.join(results_path, current_id, "time.txt")

        source_landmarks = utils.load_landmarks(source_landmarks_path)
        transformed_source_landmarks = utils.load_landmarks(transformed_source_landmarks_path)
        try:
            target_landmarks = utils.load_landmarks(target_landmarks_path)
            if target_landmarks.shape == transformed_source_landmarks.shape:
                status = "training"
            else:
                status = "evaluation"
        except:
            status = "evaluation"

        with open(time_path, "r") as file:
            execution_time = file.read()

        dataframe = pd.read_csv(output_csv_path)
        del dataframe['Unnamed: 0']
        print()
        print("Current ID: ",  current_id)
        print("Execution time: ", execution_time)

        current_id = int(current_id)
        org_source_path = dataframe['Source image'][current_id]
        org_target_path = dataframe['Target image'][current_id]
        sizes = dataframe['Image size [pixels]'][current_id]
        sizes = sizes[:].split(", ")
        y_size = int(sizes[0][1:])
        x_size = int(sizes[1][:-1])
        diagonal = np.sqrt(y_size**2 + x_size**2)
        current_id = str(current_id)
        if ".jpg" in org_source_path:
            org_source_path = org_source_path.replace(".jpg", ".mha")
            org_target_path = org_target_path.replace(".jpg", ".mha")
        elif ".png" in org_source_path:
            org_source_path = org_source_path.replace(".png", ".mha")
            org_target_path = org_target_path.replace(".png", ".mha")

        source_path = os.path.join(results_path, current_id, "source.mha")
        original_source_path = os.path.join(original_data_path, org_source_path)
        original_target_path = os.path.join(original_data_path, org_target_path)

        def get_size(path):
            reader = sitk.ImageFileReader()
            reader.SetFileName(path)
            reader.LoadPrivateTagsOn()
            reader.ReadImageInformation()
            size = reader.GetSize()
            return size

        source_size = get_size(source_path)
        original_source_size = get_size(original_source_path)
        original_target_size = get_size(original_target_path)

        print("Source path: ", source_path)
        print("Original source path: ", original_source_path)
        print("Original target path: ", original_target_path)
        print("Resampled size: ", source_size)
        print("Original source size: ", original_source_size)
        print("Original target size: ", original_target_size)

        if status == "training":
            print("Resampled median initial rTRE: ", np.median(utils.calculate_rtre(source_landmarks, target_landmarks, np.sqrt(source_size[0]**2 + source_size[1]**2))))
            print("Resampled median final rTRE: ", np.median(utils.calculate_rtre(transformed_source_landmarks, target_landmarks, np.sqrt(source_size[0]**2 + source_size[1]**2))))

        _, _, _, _, new_shape = calculate_pad_size(original_source_size, original_target_size)
        resample_ratio = max(new_shape) / max(source_size)
        
        org_source_landmarks = source_landmarks.copy()
        org_source_landmarks[:, 0] =  org_source_landmarks[:, 0] * resample_ratio - (int(np.floor((new_shape[0] - original_source_size[0])/2)))
        org_source_landmarks[:, 1] =  org_source_landmarks[:, 1] * resample_ratio - (int(np.floor((new_shape[1] - original_source_size[1])/2)))

        org_transformed_source_landmarks = transformed_source_landmarks.copy()
        org_transformed_source_landmarks[:, 0] =  org_transformed_source_landmarks[:, 0] * resample_ratio - (int(np.floor((new_shape[0] - original_target_size[0])/2)))
        org_transformed_source_landmarks[:, 1] =  org_transformed_source_landmarks[:, 1] * resample_ratio - (int(np.floor((new_shape[1] - original_target_size[1])/2)))

        if status == "training":
            org_target_landmarks = target_landmarks.copy()
            org_target_landmarks[:, 0] =  org_target_landmarks[:, 0] * resample_ratio - (int(np.floor((new_shape[0] - original_target_size[0])/2)))
            org_target_landmarks[:, 1] =  org_target_landmarks[:, 1] * resample_ratio - (int(np.floor((new_shape[1] - original_target_size[1])/2)))
            i_tre = np.median(utils.calculate_rtre(org_source_landmarks, org_target_landmarks, diagonal))
            f_tre = np.median(utils.calculate_rtre(org_transformed_source_landmarks, org_target_landmarks, diagonal))
            print("Original median initial rTRE: ", i_tre)
            print("Original median final rTRE: ", f_tre)
            dataframe['Initial TRE Median'][int(current_id)] = i_tre
            dataframe['Final TRE Median'][int(current_id)] = f_tre
            try:
                string_to_save = "Initial TRE: " + str(i_tre) + "\n" + "Resulting TRE: " + str(f_tre)
                txt_path = os.path.join(submission_path, str(current_id), "tre.txt")
                with open(txt_path, "w") as file:
                    file.write(string_to_save)
            except:
                pass
        transformed_path = os.path.join(current_id, "transformed_source_landmarks.csv")
        org_src_path = os.path.join(current_id, "org_source_landmarks.csv")
        utils.save_landmarks_submission(org_transformed_source_landmarks, os.path.join(submission_path, transformed_path))
        utils.save_landmarks_submission(org_source_landmarks, os.path.join(submission_path, org_src_path))
        if status == "training":
            org_trg_path = os.path.join(current_id, "org_target_landmarks.csv")
            utils.save_landmarks_submission(org_target_landmarks, os.path.join(submission_path, org_trg_path))

        dataframe['Execution time [minutes]'][int(current_id)] = str(float(execution_time) / 60)
        dataframe['Warped source landmarks'][int(current_id)] = transformed_path
        dataframe.to_csv(output_csv_path)

def calculate_pad_size(source_size, target_size):
    y_size_source, x_size_source = source_size
    y_size_target, x_size_target = target_size
    new_y_size = max(y_size_source, y_size_target)
    new_x_size = max(x_size_source, x_size_target)
    new_shape = (new_y_size, new_x_size)
    source_y_pad = ((int(np.floor((new_shape[0] - y_size_source)/2))), int(np.ceil((new_shape[0] - y_size_source)/2)))
    source_x_pad = ((int(np.floor((new_shape[1] - x_size_source)/2))), int(np.ceil((new_shape[1] - x_size_source)/2)))
    target_y_pad = ((int(np.floor((new_shape[0] - y_size_target)/2))), int(np.ceil((new_shape[0] - y_size_target)/2)))
    target_x_pad = ((int(np.floor((new_shape[1] - x_size_target)/2))), int(np.ceil((new_shape[1] - x_size_target)/2)))
    return source_x_pad, source_y_pad, target_x_pad, target_y_pad, new_shape

def prepare_output_csv(input_csv_path, output_csv_path):
    df = pd.read_csv(input_csv_path)
    del df['Unnamed: 0']
    i_tre = np.empty(len(df))
    i_tre[:] = np.nan
    f_tre = np.empty(len(df))
    f_tre[:] = np.nan
    df['Initial TRE Median'] = i_tre
    df['Final TRE Median'] = f_tre
    df.to_csv(output_csv_path)  


if __name__ == "__main__":
    prepare_submission()