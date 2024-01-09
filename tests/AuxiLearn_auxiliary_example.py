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
from torchsummary import summary
import os
from pathlib import Path
import torchvision
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# Add main project directory path
sys.path.append('/home/clear/ipetruli/projects/bilevel-optimization/src')

from funcBO.BilevelProblem.ClassicalBilevelProblem import ClassicalBilevelProblem
from networks.NeuralNetworkOuterModel import NeuralNetworkOuterModel
from networks.NeuralNetworkInnerModel import NeuralNetworkInnerModel
from funcBO.utils import *

# Add AuxiLearn project directory path
sys.path.append('/home/clear/ipetruli/projects/AuxiLearn')
from auxilearn.optim import MetaOptimizer

# Set logging
wandb.init(group="AuxiLearn_auxiliary_example", job_type="eval")

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
nb_aux_tasks = 4
batch_size = 64
max_epochs = 20
max_outer_iters = 100
max_inner_dual_iters, max_inner_iters = 100, 100
eval_every_n = 7

# Get data
n, m, m_out, m_in, X_train, X_val, y_train, y_val, coef = auxiliary_toy_data()

# Dataloaders for inner and outer data
inner_data = Data(X_train, y_train)
inner_dataloader = DataLoader(dataset=inner_data, batch_size=batch_size, shuffle=True)
outer_data = Data(X_val, y_val)
outer_dataloader = DataLoader(dataset=outer_data, batch_size=batch_size, shuffle=True)

# Inner model
# Neural network to approximate the function h*
inner_model = (NeuralNetworkInnerModel([2, 10, 20, 10, 1+nb_aux_tasks])).to(device)
# Optimizer that improves the approximation of h*
inner_optimizer = torch.optim.Adam(inner_model.parameters(), lr=1e-4)
inner_scheduler = None
# Neural network to approximate the function a*
inner_dual_model = None
# Optimizer that improves the approximation of a*
inner_dual_optimizer = None
inner_dual_scheduler = None

# Outer model
# The outer neural network parametrized by the outer variable mu
outer_model = (NeuralNetworkOuterModel([nb_aux_tasks, 1])).to(device)
# Initialize mu
#mu0 = (torch.ones((nb_aux_tasks,1))).to(device)
#mu0.normal_(mean=1, std=1)
# Optimizer that improves the outer variable mu
outer_optimizer = torch.optim.Adam(outer_model.parameters(), lr=1e-4)
outer_scheduler = None

meta_optimizer = MetaOptimizer(meta_optimizer=outer_optimizer, hpo_lr=1e-2, truncate_iter=max_inner_dual_iters, max_grad_norm=25)

# Print configuration
suffix = "Auxiliary_example"
print("Run configuration:", suffix)

wandb.watch((inner_model),log_freq=100)

# Gather all models
inner_models = (inner_model, inner_optimizer, inner_scheduler)
outer_models = (outer_model, outer_optimizer, outer_scheduler)

# Loss helper functions
relu = torch.nn.ReLU()
MSE = nn.MSELoss()

# Outer objective function
def fo(inner_value, labels):
    (main_pred, aux_pred) = inner_value
    (main_label, aux_label) = labels
    loss = MSE(main_pred, main_label)
    return loss

# Inner objective function
def fi(h_X_in, y_in):
    # Keep mu positive
    for p in outer_model.parameters():
        p.data.clamp_(0)
    (main_pred, aux_pred) = h_X_in
    aux_pred = torch.sigmoid(aux_pred)
    (main_label, aux_label) = y_in
    aux_label = aux_label.T
    loss = MSE(main_pred, main_label)
    aux_loss_vector = torch.zeros((nb_aux_tasks,1)).to(device)
    for task in range(nb_aux_tasks):
        aux_loss_vector[task] = MSE(aux_pred[:,task], aux_label[:,task])
    # Essentially dot product with mu
    loss += outer_model(aux_loss_vector.T)[0,0]
    return loss

# Optimize using classical implicit differention
bp_classical = ClassicalBilevelProblem(fo, fi, outer_dataloader, inner_dataloader, outer_models, inner_models, device, batch_size=batch_size, max_inner_iters=max_inner_iters, meta_optimizer=meta_optimizer)
iters, outer_losses, inner_losses, test_losses, times = bp_classical.optimize(max_epochs=max_epochs, max_iters=max_outer_iters, eval_every_n=eval_every_n)

# Show results
print("mu:\n", outer_model.layer_1.weight.data)
print("\nNumber of iterations:", iters)
print("Average iteration time:", np.average(times))