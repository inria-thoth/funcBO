import sys
import math
import time
import random
import numpy as np
import torch
import torch.nn as nn
import wandb
from torch import autograd
from torch.autograd import grad
from torch.func import functional_call
from torch.nn import functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd.functional import hessian
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import torchvision
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from statistics import mean

# Add main project directory path
sys.path.append('/home/clear/ipetruli/projects/bilevel-optimization/src')

from data.birds.Bird_dataset import Bird_dataset_naive
from model.InnerSolution.InnerSolution import InnerSolution
from model.BilevelProblem.BilevelProblem import BilevelProblem
from model.NeuralNetworks.BirdsResNet import ResNet
from model.NeuralNetworks.NeuralNetworkOuterModel import NeuralNetworkOuterModel
from model.utils import *

# Add AuxiLearn project directory path
sys.path.append('/home/clear/ipetruli/projects/AuxiLearn')

from auxilearn.hypernet import MonoLinearHyperNet
from auxilearn.optim import MetaOptimizer

# Set logging
wandb.init(group="Auxilearn_aux_time", job_type="eval_Auxilearn")

# Setting the random seed.
set_seed()

# Paths to data
train_image_file = '/home/clear/ipetruli/projects/bilevel-optimization/src/data/birds/CUB_200_2011/preprocess_data/rest_train_set.json'
aux_image_file = '/home/clear/ipetruli/projects/bilevel-optimization/src/data/birds/CUB_200_2011/preprocess_data/aux_set.json'
valid_image_file = '/home/clear/ipetruli/projects/bilevel-optimization/src/data/birds/CUB_200_2011/preprocess_data/valid_set.json'
test_image_file = '/home/clear/ipetruli/projects/bilevel-optimization/src/data/birds/CUB_200_2011/preprocess_data/test_set.json'
label_file = '/home/clear/ipetruli/projects/bilevel-optimization/src/data/birds/CUB_200_2011/preprocess_data/image_dictionary.json'
image_root = '/home/clear/ipetruli/projects/bilevel-optimization/src/data/birds/CUB_200_2011/CUB_200_2011/images'

# Data pre-processing
train_transform = transforms.Compose([transforms.Resize(256),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ])
test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])

# Setting the directory to save figures to.
figures_dir = "figures/"

# Setting the device to GPUs if available.
if torch.cuda.is_available():
    device = "cuda"
    print("All good, switching to GPUs.")
else:
    device = "cpu"
    print("No GPUs found, setting the device to CPU.")

# Setting hyper-parameters
nb_aux_tasks = 312
batch_size = 128
max_epochs = 800
max_outer_iters = 100
max_inner_dual_iters, max_inner_iters = 3, 15
nb_classes = 200
eval_every_n = 7
# Number of batches to accumulate for meta loss
n_meta_loss_accum = 1

# Dataloaders for inner and outer data
train_dataset = Bird_dataset_naive(train_image_file,label_file,image_root,transform=train_transform, finegrain=True)
inner_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
aux_dataset = Bird_dataset_naive(aux_image_file,label_file,image_root,transform=train_transform, finegrain=True)
aux_dataloader = DataLoader(aux_dataset, batch_size=batch_size, shuffle=True)
valid_dataset = Bird_dataset_naive(valid_image_file,label_file,image_root,transform=test_transform, finegrain=True)
outer_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
test_dataset = Bird_dataset_naive(test_image_file,label_file,image_root,transform=test_transform, finegrain=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Inner model
# Neural network to approximate the function h*
inner_model = ResNet().to(device)
# Optimizer that improves the approximation of h*
inner_optimizer = torch.optim.Adam(inner_model.parameters(), lr=1e-5)#, weight_decay=5e-5)
inner_scheduler = None#ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=20)

# Outer model
# The outer linear neural network
outer_model = NeuralNetworkOuterModel(layer_sizes=[nb_aux_tasks,1]).to(device)
# Optimizer that improves the outer linear model 
outer_optimizer = torch.optim.SGD(outer_model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-5)
outer_scheduler = None

meta_optimizer = MetaOptimizer(meta_optimizer=outer_optimizer, hpo_lr=1e-4, truncate_iter=max_inner_dual_iters, max_grad_norm=25)

# Print configuration
suffix = "AuxiLearn_baseline"
print("Run configuration:", suffix)

wandb.watch((inner_model),log_freq=100)

# Loss helper functions
binary_loss = nn.BCELoss(reduction='mean')
class_loss = nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)
relu = torch.nn.ReLU()

# Outer objective function
def fo(inner_value, labels):
    (main_pred, aux_pred) = inner_value
    (main_label, aux_label) = labels
    loss = class_loss(main_pred, main_label.to(device, dtype=torch.long))
    return loss

# Inner objective function
def fi(h_X_in, y_in):
    (main_pred, aux_pred) = h_X_in
    aux_pred = torch.sigmoid(aux_pred)
    (main_label, aux_label) = y_in
    aux_label = aux_label.T
    loss = class_loss(main_pred, main_label.to(device, dtype=torch.long))
    aux_loss_vector = torch.zeros((nb_aux_tasks,1)).to(device)
    for task in range(nb_aux_tasks):
        aux_loss_vector[task] = binary_loss(aux_pred[:,task], aux_label[:,task].to(device, dtype=torch.float))
    # Get the outer loss
    loss += outer_model(aux_loss_vector.T)[0,0]
    return loss

def hyperstep():
    """
    Compute hyper-parameter gradient.
    """
    # Get loss for outer data
    iters = 0
    total_meta_val_loss = .0
    for data in outer_dataloader:
        X_outer, main_label, aux_label, data_id = data
        aux_label = torch.stack(aux_label)
        y_outer = (main_label.to(device, dtype=torch.float), aux_label.to(device, dtype=torch.float))
        X_outer = X_outer.to(device, dtype=torch.float)
        inner_value = inner_model(X_outer)
        loss = fo(inner_value, y_outer)
        wandb.log({"out. loss": loss.item()})
        iters += 1
        total_meta_val_loss += loss
        meta_val_loss = total_meta_val_loss/iters
        if iters>=n_meta_loss_accum:
            break
    # Get loss for inner data
    iters = 0
    total_meta_train_loss = .0
    for data in inner_dataloader:
        X_inner, main_label, aux_label, data_id = data
        aux_label = torch.stack(aux_label)
        y_inner = (main_label.to(device, dtype=torch.float), aux_label.to(device, dtype=torch.float))
        X_inner = X_inner.to(device, dtype=torch.float)
        inner_value = inner_model(X_outer)
        loss = fi(inner_value, y_outer)
        iters += 1
        total_meta_train_loss += loss
        meta_train_loss = total_meta_train_loss/iters
        if iters>=n_meta_loss_accum:
            break
    # Hyperparam step
    hypergrads = meta_optimizer.step(
        val_loss=meta_val_loss,
        train_loss=total_meta_train_loss,
        aux_params=list(outer_model.parameters()),
        parameters=list(inner_model.parameters()),
        return_grads=True
    )
    return hypergrads

def evaluate(dataloader):
    """
    Evaluate the prediction quality on the test dataset.
    """
    total_loss, total_acc, iters = 0, 0, 0
    for data in dataloader:
        X_outer, main_label, aux_label, data_id = data
        aux_label = torch.stack(aux_label)
        y_outer = (main_label.to(device, dtype=torch.float), aux_label.to(device, dtype=torch.float))
        X_outer = X_outer.to(device, dtype=torch.float)
        inner_value = inner_model(X_outer)
        # Compute loss
        loss = fo(inner_value, y_outer)
        # Compute accuracy
        _, class_pred = torch.max(inner_value[0], dim=1)
        class_pred = class_pred.to(device)
        accuracy = get_accuracy(class_pred, y_outer[0])
        total_acc += accuracy.item()
        total_loss += loss.item()
        iters += 1
    return total_loss/iters, total_acc/iters

def optimize_inner():
    """
    Optimization loop for the inner-level model that approximates h*.
    """
    epoch_loss, epoch_iters = 0, 0
    for data in inner_dataloader:
        X_inner, main_label, aux_label, data_id = data
        aux_label = torch.stack(aux_label)
        y_inner = (main_label.to(device, dtype=torch.float), aux_label.to(device, dtype=torch.float))
        X_inner = X_inner.to(device, dtype=torch.float)
        h_X_i = inner_model(X_inner)
        # Compute the loss
        loss = fi(h_X_i, y_inner)
        # Backpropagation
        inner_optimizer.zero_grad()
        loss.backward()
        inner_optimizer.step()
        epoch_loss += loss.item()
        epoch_iters += 1
        if epoch_iters >= max_inner_iters:
            break
    inner_loss = epoch_loss/epoch_iters
    return inner_loss

def optimize_outer(max_epochs=10, max_iters=100, eval_every_n=10):
    """
    Find the optimal outer solution.
        param maxiter: maximum number of iterations
    """
    iters, outer_losses, inner_losses, val_accs, times, evaluated, acc_smooth = 0, [], [], [], [], False, 10
    # Making sure gradient of mu is computed.
    for epoch in range(max_epochs):
        epoch_iters = 0
        epoch_loss = 0
        for data in aux_dataloader:
            X_outer, main_label, aux_label, data_id = data
            aux_label = torch.stack(aux_label)
            y_outer = (main_label.to(device, dtype=torch.float), aux_label.to(device, dtype=torch.float))
            start = time.time()
            # Move data to GPU
            X_outer = X_outer.to(device, dtype=torch.float)
            # Inner value corresponds to h*(X_outer)
            forward_start = time.time()
            inner_loss = optimize_inner()
            inner_value = inner_model(X_outer)
            wandb.log({"duration of forward": time.time() - forward_start})
            loss = fo(inner_value, y_outer)
            wandb.log({"inn. loss": inner_loss})
            wandb.log({"out. loss": loss.item()})
            # Backpropagation
            backward_start = time.time()
            curr_hypergrads = hyperstep()
            wandb.log({"duration of backward": time.time() - backward_start})
            # Make sure all weights of the single linear layer are positive or null
            for p in outer_model.parameters():
                p.data.clamp_(0)
                wandb.log({"outer var. norm": torch.norm(p.data).item()})
            # Update loss and iteration count
            epoch_loss += loss.item()
            epoch_iters += 1
            iters += 1
            duration = time.time() - start
            wandb.log({"iter. time": duration})
            times.append(duration)
            # Inner losses
            inner_losses.append(inner_loss)
            # Outer losses
            outer_losses.append(loss.item())
            # Evaluate
            if (iters % eval_every_n == 0):
                _, val_acc = evaluate(outer_dataloader)
                val_accs.append(val_acc)
                wandb.log({"acc": val_acc})
                if (test_dataloader!=None) and (not evaluated) and (len(val_accs)>acc_smooth) and (mean(val_accs[-acc_smooth:]) <= mean(val_accs[-(acc_smooth*2):-(acc_smooth)])):
                    test_loss, test_acc = evaluate(test_dataloader)
                    wandb.log({"test loss": test_loss})
                    wandb.log({"test acc": test_acc})
                    evaluated = True
            if epoch_iters >= max_iters:
                break
        _, class_pred = torch.max(inner_value[0], dim=1)
        print("Train prediction:", class_pred[0:6])
        print("Train label:", y_outer[0][0:6])
    return iters, outer_losses, inner_losses, val_accs, times

# Optimize using classical implicit differention
iters, outer_losses, inner_losses, val_accs, times = optimize_outer(max_epochs=max_epochs, max_iters=max_outer_iters, eval_every_n=eval_every_n)

# Show results
print("\nNumber of iterations:", iters)
print("Average iteration time:", np.average(times))