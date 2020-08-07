import os
import time

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.nn.functional as F
from torchsummary import summary

import dataloaders as dl
import augmentation as aug
import cost_functions as cf
import utils
import paths

# from networks import affine_network_attention as an # Uncomment and use instead of the simple network for the more accurate patch-based affine registration (at the cost of longer training/inference time)
from networks import affine_network_simple as an


training_path = None # TO DEFINE
validation_path = None # TO DEFINE
models_path = paths.models_path
figures_path = paths.figures_path
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def training(training_params):
    model_name = training_params['model_name']
    num_epochs = training_params['num_epochs']
    batch_size = training_params['batch_size']
    learning_rate = training_params['learning_rate'] 
    initial_path = training_params['initial_path']
    decay_rate = training_params['decay_rate']
    model_save_path = os.path.join(models_path, model_name)

    model = an.load_network(device, path=initial_path)
    model = model.to(device)
    parameters = model.parameters()
    optimizer = optim.Adam(parameters, learning_rate)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: decay_rate**epoch)

    min_translation = -0.005
    max_translation = 0.005
    min_rotation = -5
    max_rotation = 5
    min_shear = -0.0001
    max_shear = 0.0001
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
    transforms = aug.affine_augmentation(params, True)

    training_loader = dl.UnsupervisedLoader(training_path, transforms=transforms)
    validation_loader = dl.UnsupervisedLoader(validation_path, transforms=None) 
    training_dataloader = torch.utils.data.DataLoader(training_loader, batch_size = batch_size, shuffle = True, num_workers = 4, collate_fn = dl.collate_to_list_unsupervised)
    validation_dataloader = torch.utils.data.DataLoader(validation_loader, batch_size = batch_size, shuffle = True, num_workers = 4, collate_fn = dl.collate_to_list_unsupervised)

    cost_function = cf.ncc_loss_global
    cost_function_params = dict()

    # Training starts here
    train_history = []
    val_history = []
    training_size = len(training_dataloader.dataset)
    validation_size = len(validation_dataloader.dataset)
    print("Training size: ", training_size)
    print("Validation size: ", validation_size)

    initial_training_loss = 0.0
    initial_validation_loss = 0.0
    for sources, targets in training_dataloader:
        with torch.set_grad_enabled(False):
            for i in range(len(sources)):
                source = sources[i]
                target = targets[i]
                source = source.to(device)
                target = target.to(device)
                source = source + 0.00001*torch.randn((source.size(0), source.size(1))).to(device)
                target = target + 0.00001*torch.randn((source.size(0), source.size(1))).to(device)
                loss = cost_function(source, target, device=device, **cost_function_params)
                initial_training_loss += loss.item()
    for sources, targets in validation_dataloader:
        with torch.set_grad_enabled(False):
            for i in range(len(sources)):
                source = sources[i]
                target = targets[i]
                source = source.to(device)
                target = target.to(device)
                source = source + 0.00001*torch.randn((source.size(0), source.size(1))).to(device)
                target = target + 0.00001*torch.randn((source.size(0), source.size(1))).to(device)
                loss = cost_function(source, target, device=device, **cost_function_params)
                initial_validation_loss += loss.item()
    print("Initial training loss: ", initial_training_loss / training_size)
    print("Initial validation loss: ", initial_validation_loss / validation_size)

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
                    source = source + 0.00001*torch.randn((source.size(0), source.size(1))).to(device)
                    target = target + 0.00001*torch.randn((source.size(0), source.size(1))).to(device)
                    calculated_transform = model(source.view(1, 1, source.size(0), source.size(1)), target.view(1, 1, target.size(0), target.size(1)))
                    transformed_source = utils.tensor_affine_transform(source.view(1, 1, source.size(0), source.size(1)), calculated_transform).view(source.size(0), source.size(1))
                    loss = cost_function(transformed_source, target, device=device, **cost_function_params)
                    loss_before = cost_function(source, target, device=device, **cost_function_params)
                    total_loss = loss - loss_before
                    total_loss.backward()
                    train_running_loss += loss.item()
                optimizer.step()

        print("Train Loss: ", train_running_loss / training_size)
        train_history.append(train_running_loss / training_size)

        # Validation
        val_running_loss = 0.0
        model.eval()
        for sources, targets in validation_dataloader:
            with torch.set_grad_enabled(False):
                for i in range(len(sources)):
                    source = sources[i]
                    target = targets[i]
                    source = source.to(device)
                    target = target.to(device)
                    source = source + 0.00001*torch.randn((source.size(0), source.size(1))).to(device)
                    target = target + 0.00001*torch.randn((source.size(0), source.size(1))).to(device)
                    calculated_transform = model(source.view(1, 1, source.size(0), source.size(1)), target.view(1, 1, target.size(0), target.size(1)))
                    transformed_source = utils.tensor_affine_transform(source.view(1, 1, source.size(0), source.size(1)), calculated_transform).view(source.size(0), source.size(1))
                    loss = cost_function(transformed_source, target, device=device, **cost_function_params)
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
    plt.savefig(os.path.join(figures_path, model_name + ".png"), bbox_inches = 'tight', pad_inches = 0)
    plt.show()

def visualization(model_name):
    model_path = os.path.join(models_path, model_name)
    model = an.load_network(device, path=model_path)
    model = model.to(device)

    batch_size = 4
    validation_loader = dl.UnsupervisedLoader(validation_path, transforms=None) 
    validation_dataloader = torch.utils.data.DataLoader(validation_loader, batch_size = batch_size, shuffle = True, num_workers = 4, collate_fn = dl.collate_to_list_unsupervised)

    cost_function = cf.ncc_loss_global
    cost_function_params = dict()

    validation_size = len(validation_dataloader.dataset)
    total_loss_before = 0.0
    total_loss_after = 0.0
    model.eval()
    for sources, targets in validation_dataloader:
        with torch.set_grad_enabled(False):
            for i in range(len(sources)):
                source = sources[i]
                target = targets[i]
                source = source.to(device)
                target = target.to(device)
                calculated_transform = model(source.view(1, 1, source.size(0), source.size(1)), target.view(1, 1, target.size(0), target.size(1)))
                transformed_source = utils.tensor_affine_transform(source.view(1, 1, source.size(0), source.size(1)), calculated_transform).view(source.size(0), source.size(1))
                loss_before = cost_function(source, target, device=device, **cost_function_params)
                loss_after =  cost_function(transformed_source, target, device=device, **cost_function_params)

                print("Loss before: ", loss_before.item())
                print("Loss after: ", loss_after.item())
                total_loss_before += loss_before.item()
                total_loss_after += loss_after.item()

                plt.figure()
                plt.subplot(1, 3, 1)
                plt.imshow(source.detach().cpu().numpy(), cmap='gray')
                plt.axis('off')
                plt.title("Source")
                plt.subplot(1, 3, 2)
                plt.imshow(target.detach().cpu().numpy(), cmap='gray')
                plt.axis('off')
                plt.title("Target")
                plt.subplot(1, 3, 3)
                plt.imshow(transformed_source.detach().cpu().numpy(), cmap='gray')
                plt.axis('off')
                plt.title("Transformed Source")
                plt.show()

    print("Initial validation loss: ", total_loss_before / validation_size)
    print("Final validation loss: ", total_loss_after / validation_size)

def affine_registration(source, target, model, device='cpu'):
    with torch.set_grad_enabled(False):
        output_min_size = 512
        new_shape = utils.calculate_new_shape_min((source.size(0), source.size(1)), output_min_size)
        resampled_source = utils.resample_tensor(source, new_shape, device=device)
        resampled_target = utils.resample_tensor(target, new_shape, device=device)
        calculated_transform = model(resampled_source.view(1, 1, resampled_source.size(0), resampled_source.size(1)), resampled_target.view(1, 1, resampled_target.size(0), resampled_target.size(1)))
        displacement_field = utils.transform_to_displacement_field(resampled_source.view(1, 1, resampled_source.size(0), resampled_source.size(1)), calculated_transform, device=device)
        displacement_field = utils.upsample_displacement_field(displacement_field, (2, source.size(0), source.size(1)), device=device)
        return displacement_field


def run():
    training_params = dict()
    training_params['model_name'] = None # TO DEFINE
    training_params['num_epochs'] = 500
    training_params['batch_size'] = 1
    training_params['learning_rate'] = 0.0001
    training_params['initial_path'] = None
    training_params['decay_rate'] = 0.995
    training_params['add_noise'] = True
    training(training_params)

    model_name = None # TO DEFINE
    visualization(model_name)


if __name__ == "__main__":
    run()