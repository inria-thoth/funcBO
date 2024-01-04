import sys
import math
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch import autograd
from torch.autograd import grad
from torch.func import functional_call
from torch.nn import functional as func
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd.functional import hessian
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

# Add main project directory path
sys.path.append('/home/clear/ipetruli/projects/bilevel-optimization/src')

from model.InnerSolution.InnerSolution import InnerSolution
from model.BilevelProblem.BilevelProblem import BilevelProblem
from model.utils import set_seed, plot_loss, tensor_to_state_dict, get_memory_info

from experiments.NYuv2.experiments.nyuv2.data import nyu_dataloaders
from experiments.NYuv2.experiments.nyuv2.model import SegNetSplit
from model.NeuralNetworks.NeuralNetworkOuterModel import NeuralNetworkOuterModel
from model.NeuralNetworks.NeuralNetworkInnerDualModel import NeuralNetworkInnerDualModel

# Setting the random seed.
set_seed()

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
batch_size = 4
max_epochs = 200

# Dataloaders for inner and outer data
inner_dataloader, outer_dataloader, test_dataloader = nyu_dataloaders(
    datapath="/home/clear/ipetruli/projects/bilevel-optimization/src/experiments/NYuv2/data",
    validation_indices="/home/clear/ipetruli/projects/bilevel-optimization/src/experiments/NYuv2/hpo_validation_indices.json",
    aux_set=False,
    batch_size=batch_size,
    val_batch_size=batch_size
)

# Inner model
# Neural network to approximate the function h*
inner_model = SegNetSplit(logsigma=False).to(device)
# Optimizer that improves the approximation of h*
inner_optimizer = torch.optim.Adam(inner_model.parameters(), lr=1e-4)
inner_scheduler = lr_scheduler.StepLR(inner_optimizer, step_size=100, gamma=0.5)

# Inner dual model
# Neural network to approximate the function a*
inner_dual_model = SegNetSplit(logsigma=False).to(device)
# Optimizer that improves the approximation of a*
inner_dual_optimizer = torch.optim.Adam(inner_model.parameters(), lr=1e-4)
inner_dual_scheduler = lr_scheduler.StepLR(inner_optimizer, step_size=100, gamma=0.5)

# Outer model
# The outer neural network parametrized by the outer variable mu
outer_model = NeuralNetworkOuterModel(layer_sizes=[2,1]).to(device)
# Initialize mu
mu0 = torch.ones((2, 1)).to(device)
# Optimizer that improves the outer variable mu
outer_optimizer = torch.optim.SGD([mu0], lr=1e-3, momentum=.9, weight_decay=1e-4)
outer_scheduler = lr_scheduler.StepLR(outer_optimizer, step_size=100, gamma=0.5)

# Gather all models
inner_models = (inner_model, inner_optimizer, inner_scheduler, inner_dual_model, inner_dual_optimizer, inner_dual_scheduler)
outer_models = (outer_model, outer_optimizer, outer_scheduler)

# Helper function for objectives
def calc_loss(seg_pred, seg, depth_pred, depth, pred_normal, normal):
    # binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(depth, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).to(depth.device)
    n_nonzeros = (binary_mask != 0).sum(dim=(1, 2, 3))  # non-zeros per image
    # semantic loss: depth-wise cross entropy
    seg_loss = func.nll_loss(seg_pred, seg, ignore_index=-1, reduction='none').mean(dim=(1, 2))
    # depth loss: l1 norm
    depth_loss = torch.sum(torch.abs(depth_pred - depth) * binary_mask, dim=1).sum(dim=(1, 2)) / n_nonzeros
    # normal loss: dot product
    normal_loss = 1 - torch.sum((pred_normal * normal) * binary_mask, dim=1).sum(dim=(1, 2)) / n_nonzeros
    return (seg_loss, depth_loss, normal_loss)

# Outer objective function
def fo(mu, h_X_out, y_out):
    seg, depth, normal = y_out
    seg_pred, depth_pred, pred_normal = h_X_out
    seg = seg.type(torch.LongTensor).to(device)
    seg_loss, depth_loss, normal_loss = calc_loss(seg_pred, seg, depth_pred, depth, pred_normal, normal)
    loss_image = torch.column_stack((depth_loss.mean(0), normal_loss.mean(0)))
    loss = seg_loss.mean(0)
    return loss

# Inner objective function
def fi(mu, h_X_in, y_in):
    seg, depth, normal = y_in
    seg_pred, depth_pred, pred_normal = h_X_in
    seg = seg.type(torch.LongTensor).to(device)
    seg_loss, depth_loss, normal_loss = calc_loss(seg_pred, seg, depth_pred, depth, pred_normal, normal)
    loss_image = torch.column_stack((depth_loss.mean(0), normal_loss.mean(0)))
    loss = seg_loss.mean(0) + functional_call(outer_model, tensor_to_state_dict(outer_model, mu, device), loss_image)
    return loss

# Optimize using neural implicit differention
bp_neural = BilevelProblem(fo, fi, outer_dataloader, inner_dataloader, outer_models, inner_models, device, batch_size=batch_size)
iters, outer_losses, inner_losses, test_losses, times = bp_neural.optimize(mu0, max_epochs=max_epochs, test_dataloader=test_dataloader)

# Show results
print("Number of iterations:", iters)
print("Average iteration time:", np.average(times))

plot_loss(figures_dir+"out_loss_NID_200epochs", outer_losses, inner_losses, test_losses, title="Losses of neur. im. diff.")

#Use gpu 23 or 28 rather than 21, more memory