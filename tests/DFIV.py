import torch
import wandb
import numpy as np

import time

import os
os.environ['WANDB_DISABLED'] = 'true'
os.chdir('/home/ipetruli/funcBO')
from datasets.dsprite.dspriteDFIV import *
from datasets.dsprite.trainer import DFIVTrainer

# Setting the device to GPUs if available.
if torch.cuda.is_available():
    device = "cuda"
    print("All good, switching to GPUs.")
else:
    device = "cpu"
    print("No GPUs found, setting the device to CPU.")

# Setting hyper-parameters
seed = 42
max_epochs = 5000
max_inner_iters = 20
lam2 = 0.1
lam1 = 0.1

# Get data
#inner_data, outer_data, test_data = generate_dsprite_data(train_size=6, val_size=6, device=device)
test_data = generate_test_dsprite(device=device)
train_data = generate_train_dsprite(data_size=5000, rand_seed=seed, device=device)
inner_data, outer_data = split_train_data(train_data, split_ratio=0.5)

# Neural networks for dsprites data
inner_model, outer_model = build_net_for_dsprite(seed)
inner_model.to(device)
outer_model.to(device)
print("First inner layer:", list(inner_model.parameters())[0].data)
print("First outer layer:", list(outer_model.parameters())[0].data)

# Optimizer that improves the approximation of h*
inner_lr = 1e-4
inner_wd = lam1
inner_optimizer = torch.optim.Adam(inner_model.parameters(), lr=inner_lr, weight_decay=inner_wd)
inner_scheduler = None

# Optimizer that improves the outer variable
outer_lr = 1e-4
outer_wd = lam2
outer_optimizer = torch.optim.Adam(outer_model.parameters(), lr=outer_lr, weight_decay=outer_wd)
outer_scheduler = None

# Print configuration
run_name = "DFIV::="+" inner_lr:"+str(inner_lr)+", outer_lr"+str(outer_lr)+", inner_wd:"+str(inner_wd)+", outer_wd:"+str(outer_wd)+", seed:"+str(seed)+", lam_u:"+str(lam2*(outer_data[0].size()[0]))+", lam_V:"+str(lam1*(inner_data[0].size()[0]))+", max_epochs:"+str(max_epochs)
print("Run configuration:", run_name)

# Set logging
wandb.init(group="Dsprites_DFIV_test", name=run_name)

# Initialize and train the DFIVTrainer
dfiv_trainer = DFIVTrainer(
    outer_model, inner_model, outer_optimizer, inner_optimizer,
    train_params = {"stage1_iter": max_inner_iters, "stage2_iter": 1, "max_epochs": max_epochs, "lam1": lam1, "lam2": lam2}
)

# Solve the two-stage regression problem
dfiv_trainer.train(inner_data, outer_data, test_data)