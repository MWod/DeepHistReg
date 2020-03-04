import os
import time

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import torch.nn.functional as F
from torchsummary import summary

import dataloaders as dl
import augmentation as aug
import anhir_utils as au
import cost_functions as cf

from networks import affine_network as an

current_path = os.path.abspath(os.path.dirname(__file__))
training_path = None # to specify
validation_path = None # to specify
models_path = None # to specify
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") None # to specify

model_name = "test_name"

def training():
    num_epochs = 100
    batch_size = 4
    learning_rate = 0.0001

    model_save_path = os.path.join(models_path, model_name)

    model = an.load_network(device)
    model = model.to(device)
    summary(model, [(1, 512, 512),   (1, 512, 512)])
    parameters = model.parameters()
    optimizer = optim.Adam(parameters, learning_rate)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.998**epoch)

    min_translation = -0.02
    max_translation = 0.02
    min_rotation = -15
    max_rotation = 15
    min_shear = -0.005
    max_shear = 0.005
    min_scale = 0.9
    max_scale = 1.15
    params = dict()
    params['min_translation'] = min_translation
    params['max_translation'] = max_translation
    params['min_rotation'] = min_rotation
    params['max_rotation'] = max_rotation
    params['min_shear'] = min_shear
    params['max_shear'] = max_shear
    params['min_scale'] = min_scale
    params['max_scale'] = max_scale
    possible_transforms = aug.affine_augmentation(params, True)
    background_delete = False
    
    training_generator = dl.AffineUnsupervisedLoader(training_path, possible_transforms, background_delete, randomly_swap=True)
    validation_generator = dl.AffineUnsupervisedLoader(validation_path, possible_transforms, background_delete, randomly_swap=True)
    training_dataloader = utils.data.DataLoader(training_generator, batch_size = batch_size, shuffle = True, num_workers = 16, collate_fn = dl.collate_to_list_unsupervised)
    validation_dataloader = utils.data.DataLoader(validation_generator, batch_size = batch_size, shuffle = True, num_workers = 16, collate_fn = dl.collate_to_list_unsupervised)

    cost_function = cf.ncc_loss

    # Training starts here
    train_history = []
    val_history = []

    training_size = len(training_dataloader.dataset)
    validation_size = len(validation_dataloader.dataset)

    print("Training size: ", training_size)
    print("Validation size: ", validation_size)

    for epoch in range(num_epochs):
        bet = time.time()
        print("Current epoch: ", str(epoch + 1) + "/" + str(num_epochs))
        # Training
        train_running_loss = 0.0
        model.train()
        for sources, targets in training_dataloader:
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                for i in range(len(sources)):
                    source = sources[i]
                    target = targets[i]

                    source = source.to(device)
                    target = target.to(device)

                    calculated_transform = model(source, target)
                    transformed_source = au.tensor_affine_transform(source, calculated_transform)
                    loss = cost_function(transformed_source, target, device)
                    loss.backward()
                    train_running_loss += loss.item()
                optimizer.step()

        print("Train Loss: ", train_running_loss / training_size)
        train_history.append(train_running_loss / training_size)

        # Validation
        val_running_loss = 0.0
        model.eval()
        for sources, targets in validation_dataloader:
            optimizer.zero_grad()
            with torch.set_grad_enabled(False):
                for i in range(len(sources)):
                    source = sources[i]
                    target = targets[i]

                    source = source.to(device)
                    target = target.to(device)

                    calculated_transform = model(source, target)
                    transformed_source = au.tensor_affine_transform(source, calculated_transform)
                    loss = cost_function(transformed_source, target, device)
                    val_running_loss += loss.item()

        print("Val Loss: ", val_running_loss / validation_size)
        val_history.append(val_running_loss / validation_size)

        scheduler.step()

        eet = time.time()
        print("Epoch time: ", eet - bet, "seconds.")
        print("Estimated time to end: ", (eet - bet)*(num_epochs-epoch), "seconds.")

    if model_save_path is not None:
        torch.save(model.state_dict(), model_save_path)

    plt.figure()
    plt.plot(train_history, "r-")
    plt.plot(val_history, "b-")
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Train", "Validation"])
    plt.show()


def test_visually():
    model_load_path = os.path.join(models_path, model_name)
    model = an.load_network(device, path=model_load_path)
    model = model.to(device)
    model.eval()

    background_delete = False
    path = validation_path
    batch_size = 1
    validation_generator = dl.AffineUnsupervisedLoader(path, None, background_delete)
    validation_dataloader = utils.data.DataLoader(validation_generator, batch_size = batch_size, shuffle = True, num_workers = 16, collate_fn = dl.collate_to_list_unsupervised)
    cost_function = cf.ncc_loss

    tt = 0
    for sources, targets in validation_dataloader:
        plt.figure(dpi=250)
        for i in range(len(sources)):
            source = sources[i]
            target = targets[i]

            source = source.to(device)
            target = target.to(device)

            b_t = time.time()
            with torch.set_grad_enabled(False):
                calculated_transform = model(source, target)
            e_t = time.time()
            tt += e_t - b_t

            transformed_source = au.tensor_affine_transform(source, calculated_transform)
            print("Cost before: ", cost_function(source, target, device))
            print("Cost after: ", cost_function(transformed_source, target, device))

            plt.subplot(batch_size, 4, i*4 + 1)
            plt.imshow(source.cpu().permute(0, 2, 3, 1)[0, :, :, 0], cmap='gray')
            plt.axis('off')
            plt.subplot(batch_size, 4, i*4 + 2)
            plt.imshow(target.cpu().permute(0, 2, 3, 1)[0, :, :, 0], cmap='gray')
            plt.axis('off')
            plt.subplot(batch_size, 4, i*4 + 3)
            plt.imshow(transformed_source.cpu().permute(0, 2, 3, 1)[0, :, :, 0], cmap='gray')
            plt.axis('off')
            plt.subplot(batch_size, 4, i*4 + 4)
            plt.imshow(target.cpu().permute(0, 2, 3, 1)[0, :, :, 0] - transformed_source.cpu().permute(0, 2, 3, 1)[0, :, :, 0], cmap='gray')
            plt.axis('off')
        plt.show()

    print("Average forward pass time: ", tt / len(validation_dataloader.dataset))

def run():
    training()
    # test_visually()


if __name__ == "__main__":
    run()