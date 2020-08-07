import os
import time
import random

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import math

import torch
import torch.utils
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F

import dataloaders as dl
import augmentation as aug
import cost_functions as cf
import utils
import paths

from networks import segmentation_network as sn

training_path = None
validation_path = None
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

    model = sn.load_network(device, path=initial_path)
    model = model.to(device)
    parameters = model.parameters()
    optimizer = optim.Adam(parameters, learning_rate)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: decay_rate**epoch)


    training_loader = dl.SegmentationLoader(training_path)
    validation_loader = dl.SegmentationLoader(validation_path) 
    training_dataloader = torch.utils.data.DataLoader(training_loader, batch_size = batch_size, shuffle = True, num_workers = 4, collate_fn = dl.collate_to_list_segmentation)
    validation_dataloader = torch.utils.data.DataLoader(validation_loader, batch_size = batch_size, shuffle = True, num_workers = 4, collate_fn = dl.collate_to_list_segmentation)

    cost_function = cf.dice_loss
    cost_function_params = dict()

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
        for sources, targets, source_masks, target_masks in training_dataloader:
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                for i in range(len(sources)):
                    source = sources[i].to(device)
                    target = targets[i].to(device)
                    source = source + 0.00001*torch.randn((source.size(0), source.size(1))).to(device)
                    target = target + 0.00001*torch.randn((source.size(0), source.size(1))).to(device)
                    source_mask = source_masks[i].to(device).view(1, 1, source.size(0), source.size(1))
                    target_mask = target_masks[i].to(device).view(1, 1, target.size(0), target.size(1))
                    source_mask_pred = model(source.view(1, 1, source.size(0), source.size(1)))
                    target_mask_pred = model(target.view(1, 1, target.size(0), target.size(1)))
                    loss_src = cost_function(source_mask_pred, source_mask, device=device, **cost_function_params)
                    loss_tgt = cost_function(target_mask_pred, target_mask, device=device, **cost_function_params)
                    loss = (loss_src + loss_tgt) / 2
                    train_running_loss += loss.item()
                    loss.backward()
                optimizer.step()
        print("Train Loss: ", train_running_loss / training_size)
        train_history.append(train_running_loss / training_size)

        # Validation
        val_running_loss = 0.0
        model.eval()
        for sources, targets, source_masks, target_masks in validation_dataloader:
            with torch.set_grad_enabled(False):
                for i in range(len(sources)):
                    source = sources[i].to(device)
                    target = targets[i].to(device)
                    source = source + 0.00001*torch.randn((source.size(0), source.size(1))).to(device)
                    target = target + 0.00001*torch.randn((source.size(0), source.size(1))).to(device)
                    source_mask = source_masks[i].to(device).view(1, 1, source.size(0), source.size(1))
                    target_mask = target_masks[i].to(device).view(1, 1, target.size(0), target.size(1))
                    source_mask_pred = model(source.view(1, 1, source.size(0), source.size(1)))
                    target_mask_pred = model(target.view(1, 1, target.size(0), target.size(1)))
                    loss_src = cost_function(source_mask_pred, source_mask, device=device, **cost_function_params)
                    loss_tgt = cost_function(target_mask_pred, target_mask, device=device, **cost_function_params)
                    loss = (loss_src + loss_tgt) / 2
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
    model = sn.load_network(device, path=model_path)
    model = model.to(device)

    batch_size = 4
    validation_loader = dl.SegmentationLoader(training_path) 
    validation_dataloader = torch.utils.data.DataLoader(validation_loader, batch_size = batch_size, shuffle = True, num_workers = 4, collate_fn = dl.collate_to_list_segmentation)
    cost_function = cf.dice_loss
    cost_function_params = dict()

    model.eval()
    for sources, targets, source_masks, target_masks in validation_dataloader:
        with torch.set_grad_enabled(False):
            for i in range(len(sources)):
                source = sources[i].to(device)
                target = targets[i].to(device)
                source = source + 0.00001*torch.randn((source.size(0), source.size(1))).to(device)
                target = target + 0.00001*torch.randn((source.size(0), source.size(1))).to(device)
                source_mask = source_masks[i].to(device).view(1, 1, source.size(0), source.size(1))
                target_mask = target_masks[i].to(device).view(1, 1, target.size(0), target.size(1))
                source_mask_pred = model(source.view(1, 1, source.size(0), source.size(1)))
                target_mask_pred = model(target.view(1, 1, target.size(0), target.size(1)))
                loss_src = cost_function(source_mask_pred, source_mask, device=device, **cost_function_params)
                loss_tgt = cost_function(target_mask_pred, target_mask, device=device, **cost_function_params)
                print("Loss src: ", loss_src.item())
                print("Loss tgt: ", loss_tgt.item())
                plt.figure()
                plt.subplot(2, 2, 1)
                plt.imshow(source_mask[0, 0, :, :].detach().cpu().numpy(), cmap='gray')
                plt.axis('off')
                plt.title("S")
                plt.subplot(2, 2, 2)
                plt.imshow(source_mask_pred[0, 0, :, :].detach().cpu().numpy(), cmap='gray')
                plt.axis('off')
                plt.title("SPred")
                plt.subplot(2, 2, 3)
                plt.imshow(target_mask[0, 0, :, :].detach().cpu().numpy(), cmap='gray')
                plt.axis('off')
                plt.title("R")
                plt.subplot(2, 2, 4)
                plt.imshow(target_mask_pred[0, 0, :, :].detach().cpu().numpy(), cmap='gray')
                plt.axis('off')
                plt.title("TPred")
                plt.show()

def segmentation(source, target, model, device="cpu"):
    with torch.set_grad_enabled(False):
        output_min_size = 512
        new_shape = utils.calculate_new_shape_min((source.size(0), source.size(1)), output_min_size)
        resampled_source = utils.resample_tensor(source, new_shape, device=device)
        resampled_target = utils.resample_tensor(target, new_shape, device=device)
        source_mask = model(resampled_source.view(1, 1, resampled_source.size(0), resampled_source.size(1)))[0, 0, :, :]
        target_mask = model(resampled_target.view(1, 1, resampled_target.size(0), resampled_target.size(1)))[0, 0, :, :]
        source_mask = utils.resample_tensor(source_mask, (source.size(0), source.size(1)), device=device) > 0.5
        target_mask = utils.resample_tensor(target_mask, (target.size(0), target.size(1)), device=device) > 0.5
        return source_mask, target_mask


if __name__ == "__main__":
    training_params = dict()
    training_params['model_name'] = None # TO DEFINE
    training_params['num_epochs'] = 100
    training_params['batch_size'] = 4
    training_params['learning_rate'] = 0.001
    training_params['initial_path'] = None
    training_params['decay_rate'] = 0.98
    training(training_params)

    model_name = None # TO DEFINE
    visualization(model_name)
