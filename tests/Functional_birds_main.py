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

sys.path.insert(1, '/home/clear/ipetruli/projects/bilevel-optimization/src/data/birds')

from Bird_dataset import Bird_dataset_naive
from torchvision import transforms

# Add main project directory path
sys.path.append('/home/clear/ipetruli/projects/bilevel-optimization/src')

from funcBO.InnerSolution import InnerSolution
from funcBO.BilevelProblem import BilevelProblem
from networks.BirdsResNet import ResNet
from networks.NeuralNetworkOuterModel import NeuralNetworkOuterModel
from funcBO.utils import *

# Set logging
wandb.init(group="NID_0.01", job_type="eval")

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
# Neural network to approximate the function a*
inner_dual_model = ResNet().to(device)
# Optimizer that improves the approximation of a*
inner_dual_optimizer = torch.optim.Adam(inner_dual_model.parameters(), lr=1e-5)
inner_dual_scheduler = None

# Outer model
# The outer neural network parametrized by the outer variable mu
outer_model = NeuralNetworkOuterModel(layer_sizes=[nb_aux_tasks,1]).to(device)
# Initialize mu
mu0 = (torch.randn((nb_aux_tasks,1))).to(device)
# Optimizer that improves the outer variable mu
outer_optimizer = torch.optim.SGD([mu0], lr=1e-2, momentum=0.9, weight_decay=1e-5)
outer_scheduler = None

# Print configuration
suffix = "NID"
print("Run configuration:", suffix)

wandb.watch((inner_model),log_freq=100)

# Gather all models
inner_models = (inner_model, inner_optimizer, inner_scheduler, inner_dual_model, inner_dual_optimizer, inner_dual_scheduler)
outer_models = (outer_model, outer_optimizer, outer_scheduler)

# Loss helper functions
binary_loss = nn.BCELoss(reduction='mean')
class_loss = nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)
relu = torch.nn.ReLU()

# Outer objective function
def fo(mu, inner_value, labels):
    (main_pred, aux_pred) = inner_value
    (main_label, aux_label) = labels
    loss = class_loss(main_pred, main_label.to(device, dtype=torch.long))
    return loss

# Inner objective function
def fi(mu, h_X_in, y_in):
    # Keep mu positive
    mu = relu(mu)
    (main_pred, aux_pred) = h_X_in
    aux_pred = torch.sigmoid(aux_pred)
    (main_label, aux_label) = y_in
    aux_label = aux_label.T
    loss = class_loss(main_pred, main_label.to(device, dtype=torch.long))
    aux_loss_vector = torch.zeros((nb_aux_tasks,1)).to(device)
    for task in range(nb_aux_tasks):
        aux_loss_vector[task] = binary_loss(aux_pred[:,task], aux_label[:,task].to(device, dtype=torch.float))
    # Essentially dot product with mu
    loss += (mu.T @ aux_loss_vector)[0,0]#functional_call(outer_model, tensor_to_state_dict(outer_model, mu, device), aux_loss_vector.T)[0,0]
    return loss

# Optimize using neural implicit differention
bp_neural = BilevelProblem(fo, fi, outer_dataloader, inner_dataloader, outer_models, inner_models, device, batch_size=batch_size, max_inner_iters=max_inner_iters, max_inner_dual_iters=max_inner_dual_iters, aux_dataloader=aux_dataloader)
iters, outer_losses, inner_losses, test_losses, times = bp_neural.optimize(mu0, max_epochs=max_epochs, test_dataloader=test_dataloader, max_iters=max_outer_iters, eval_every_n=eval_every_n)

# Show results
print("mu:\n", mu0)
print("\nNumber of iterations:", iters)
print("Average iteration time:", np.average(times))