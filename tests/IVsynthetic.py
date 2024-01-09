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

from funcBO.BilevelProblem import BilevelProblem
from networks.NeuralNetworkOuterModel import NeuralNetworkOuterModel
from networks.NeuralNetworkInnerModel import NeuralNetworkInnerModel
from funcBO.utils import *

# Set seed
set_seed()

# Set logging
wandb.init(group="IVsynthetic_example", job_type="IVsynthetic_eval")

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
batch_size = 64
max_epochs = 200
max_outer_iters = 100
max_inner_dual_iters, max_inner_iters = 10, 10
eval_every_n = 10

# Get data
n, m, X_train, X_val, y_train, y_val = syntheticIVdata()

# Dataloaders for inner and outer data
inner_data = IVSyntheticData(X_train, y_train)
inner_dataloader = DataLoader(dataset=inner_data, batch_size=batch_size, shuffle=True, drop_last=True)
outer_data = IVSyntheticData(X_val, y_val)
outer_dataloader = DataLoader(dataset=outer_data, batch_size=batch_size, shuffle=True, drop_last=True)

# Inner model
# Neural network to approximate the function h*
inner_model = (NeuralNetworkOuterModel([n, 1])).to(device)#(NeuralNetworkInnerModel([1, 10, 20, 10, 1])).to(device)
# Optimizer that improves the approximation of h*
inner_optimizer = torch.optim.Adam(inner_model.parameters(), lr=1e-3)
inner_scheduler = None
# Neural network to approximate the function a*
inner_dual_model = (NeuralNetworkOuterModel([n, 1])).to(device)#(NeuralNetworkInnerModel([1, 10, 20, 10, 1])).to(device)
# Optimizer that improves the approximation of a*
inner_dual_optimizer = torch.optim.Adam(inner_dual_model.parameters(), lr=1e-3)
inner_dual_scheduler = None

# Outer model
# The outer neural network parametrized by the outer variable
outer_model = (NeuralNetworkOuterModel([n, 1])).to(device)
# Optimizer that improves the outer variable
outer_param = torch.clone(outer_model.layer_1.weight.data)
outer_optimizer = torch.optim.Adam([outer_param], lr=1e-3)
outer_scheduler = None

# Print configuration
suffix = "IVsynthetic_example"
print("Run configuration:", suffix)

wandb.watch((inner_model), log_freq=100)

# Gather all models
inner_models = (inner_model, inner_optimizer, inner_scheduler, inner_dual_model, inner_dual_optimizer, inner_dual_scheduler)
outer_models = (outer_model, outer_optimizer, outer_scheduler)

# Loss helper functions
relu = torch.nn.ReLU()
MSE = nn.MSELoss()

# Outer objective function
def fo(outer_param, g_z_out, Y):
    loss = MSE(g_z_out, Y)
    return loss

# Inner objective function
def fi(outer_param, g_z_in, X):
    outer_NN_dic = tensor_to_state_dict(outer_model, outer_param, device)
    f_X = torch.func.functional_call(outer_model, parameter_and_buffer_dicts=outer_NN_dic, args=X)
    loss = MSE(g_z_in, f_X)
    return loss

# Optimize using neural implicit differention
bp_neural = BilevelProblem(fo, fi, outer_dataloader, inner_dataloader, outer_models, inner_models, device, batch_size=batch_size, max_inner_iters=max_inner_iters, max_inner_dual_iters=max_inner_dual_iters)
iters, outer_losses, inner_losses, test_losses, times = bp_neural.optimize(outer_param, max_epochs=max_epochs, max_iters=max_outer_iters, eval_every_n=eval_every_n)

# Show results
print("Outer param:", outer_param)
print("Inner param:", inner_model.layer_1.weight.data)
print("Number of iterations:", iters)
print("Average iteration time:", np.average(times))