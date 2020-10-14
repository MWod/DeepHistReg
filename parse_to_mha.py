import os
import shutil
import time


import numpy as np
import matplotlib.pyplot as plt


import SimpleITK as sitk

input_path = None # ANHIR data path 
output_path = None # Output path

original = "ANHIR_Data" # assumes that the last folder is names "ANHIR_Data", otherwise replace
to_replace = "ANHIR_MHA" # assumes that the last folder of the outputdata is "ANHIR_MHA", otherwise replace

def run():
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if ".jpg" in file.lower() or ".png" in file.lower():
                input_file_path = os.path.join(root, file)
                if ".jpg" in file.lower():
                    output_file_path = os.path.join(root.replace(original, to_replace), file.replace(".jpg", ".mha"))
                elif ".png" in file.lower():
                    output_file_path = os.path.join(root.replace(original, to_replace), file.replace(".png", ".mha"))
                if not os.path.isdir(os.path.dirname(output_file_path)):
                    os.makedirs(os.path.dirname(output_file_path))
                print("Current input path:" , input_file_path)
                print("Current output path: ", output_file_path)
                b_t = time.time()
                image_jpg = sitk.ReadImage(input_file_path)
                e_t = time.time()
                print("JPG loading time: ", e_t - b_t)
                sitk.WriteImage(image_jpg, output_file_path)
                print("Done")
                print()

            elif ".csv" in file.lower():
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(root.replace(original, to_replace), file)
                print("Current input path:" , input_file_path)
                print("Current output path: ", output_file_path)
                if not os.path.isdir(os.path.dirname(output_file_path)):
                    os.makedirs(os.path.dirname(output_file_path))
                shutil.copy(input_file_path, output_file_path)
                print("Done")
                print()




if __name__ == "__main__":
    run()