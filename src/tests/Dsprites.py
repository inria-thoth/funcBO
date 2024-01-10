import sys
import torch
import wandb
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import MSELoss
#from torch.optim.lr_scheduler import StepLR

# Add main project directory path
sys.path.append('/home/ipetruli/funcBO/src')

from model.utils import *
# The experiment-specific functions
from model.BilevelProblem.BilevelProblem import BilevelProblem
from my_data.dsprite.dspriteBilevel import *
from my_data.dsprite.trainer import *

import os
#os.environ['WANDB_DISABLED'] = 'true'

# Set seed
seed = 42#set_seed()
set_seed(seed)

# Setting the device to GPUs if available.
if torch.cuda.is_available():
    device = "cuda"
    print("All good, switching to GPUs.")
else:
    device = "cpu"
    print("No GPUs found, setting the device to CPU.")

# Setting hyper-parameters
torch.set_default_dtype(torch.float64)
batch_size = 2500
max_epochs = 5000
max_inner_dual_epochs, max_inner_epochs = 20, 20
eval_every_n = 1
lam_u = 0.1
lam_V = 0.1
# Method for computing a*() : "closed_form_a", "GD", "closed_form_DFIV", "GDinH" (for this option need to change the architecture of the inner dual network)
a_star_method = "closed_form_a_tensors"

# Get data
#instrumental_train, treatment_train, outcome_train, instrumental_val, treatment_val, outcome_val, treatment_test, outcome_test = generate_dsprite_data(train_size=6, val_size=6)
test_data = generate_test_dsprite(device=device)
train_data, validation_data = generate_train_dsprite(data_size=5000, rand_seed=seed, device=device, val_size=0)
inner_data, outer_data = split_train_data(train_data, split_ratio=0.5)

# Weird scaling of lambdas done in training
lam_V *= inner_data[0].size()[0]
lam_u *= outer_data[0].size()[0]

instrumental_in, treatment_in, outcome_in, instrumental_out, treatment_out, outcome_out, treatment_test, outcome_test = inner_data.instrumental, inner_data.treatment, inner_data.outcome, outer_data.instrumental, outer_data.treatment, outer_data.outcome, test_data.treatment, test_data.structural
if not (validation_data is None):
    instrumental_val, treatment_val, outcome_val = validation_data.instrumental, validation_data.treatment, validation_data.outcome

# Dataloaders for inner and outer data
inner_data = DspritesData(instrumental_in, treatment_in, outcome_in)
inner_dataloader = DataLoader(dataset=inner_data, batch_size=batch_size, shuffle=False)#, drop_last=True)
outer_data = DspritesData(instrumental_out, treatment_out, outcome_out)
outer_dataloader = DataLoader(dataset=outer_data, batch_size=batch_size, shuffle=False)#, drop_last=True)
test_data = DspritesTestData(treatment_test, outcome_test)
if not (validation_data is None):
    validation_data = DspritesData(instrumental_val, treatment_val, outcome_val)

inner_data.instrumental = inner_data.instrumental.to(device)
inner_data.treatment = inner_data.treatment.to(device)

# Neural networks for dsprites data
inner_model, inner_dual_model, outer_model = build_net_for_dsprite(seed)
inner_model.to(device)
inner_dual_model.to(device)
outer_model.to(device)
outer_model.eval()
print("First inner layer:", inner_model.layer1.weight.data)
print("First outer layer:", outer_model.layer1.weight.data)

# Optimizer that improves the approximation of h*
inner_lr = 1e-4
inner_wd = 0.1
inner_optimizer = torch.optim.Adam(inner_model.parameters(), lr=inner_lr, weight_decay=inner_wd)
inner_scheduler = None

# Optimizer that improves the approximation of a*
inner_dual_lr = 1e-4
inner_dual_wd = 0.1
inner_dual_optimizer = torch.optim.Adam(inner_dual_model.parameters(), lr=inner_dual_lr, weight_decay=inner_dual_wd)
inner_dual_scheduler = None

# The outer neural network parametrized by the outer variable
outer_param = state_dict_to_tensor(outer_model, device)#torch.cat((torch.rand(u_dim).to(device), state_dict_to_tensor(outer_model, device)), 0)
outer_lr = 1e-4
outer_wd = 0.1
# Optimizer that improves the outer variable
outer_optimizer = torch.optim.Adam([outer_param], lr=outer_lr, weight_decay=outer_wd)
outer_scheduler = None

# Print configuration
run_name = "Dsprites bilevel::="+" inner_lr:"+str(inner_lr)+", dual_lr:"+str(inner_dual_lr)+", outer_lr:"+str(outer_lr)+", inner_wd:"+str(inner_wd)+", dual_wd:"+str(inner_dual_wd)+", outer_wd:"+str(outer_wd)+", max_inner_dual_epochs:"+str(max_inner_dual_epochs)+", max_inner_epochs:"+str(max_inner_epochs)+", seed:"+str(seed)+", lam_u:"+str(lam_u)+", lam_V:"+str(lam_V)+", batch_size:"+str(batch_size)+", max_epochs:"+str(max_epochs)
print("Run configuration:", run_name)

# Set logging
wandb.init(group="Dsprites_bilevelIV_a=Wh_test", name=run_name)

# Gather all models
inner_models = (inner_model, inner_optimizer, inner_scheduler, inner_dual_model, inner_dual_optimizer, inner_dual_scheduler)
outer_models = (outer_model, outer_optimizer, outer_scheduler)

# Loss helper functions
MSE = nn.MSELoss()

# Outer objective function
def fo(outer_param, g_z_out, Y):
    # Get the value of g(Z) inner
    instrumental_1st_feature = inner_model(inner_data.instrumental)
    instrumental_1st_feature = instrumental_1st_feature.detach()
    # Get the value of g(Z) outer
    instrumental_2nd_feature = g_z_out
    # Get the value of f(X) inner
    outer_NN_dic = tensor_to_state_dict(outer_model, outer_param, device)
    treatment_1st_feature = torch.func.functional_call(outer_model, parameter_and_buffer_dicts=outer_NN_dic, args=inner_data.treatment, strict=True)
    treatment_1st_feature = treatment_1st_feature.detach()
    #print("before call instrumental_net first layer norm:", torch.norm(inner_model.layer1.weight))
    res = fit_2sls(treatment_1st_feature, instrumental_1st_feature, instrumental_2nd_feature, Y, lam_V, lam_u)
    return res["stage2_loss"], res["stage2_weight"]

# Inner objective function
def fi(outer_param, g_z_in, X):
    # Get the value of f(X) outer
    outer_NN_dic = tensor_to_state_dict(outer_model, outer_param, device)
    treatment_feature = (torch.func.functional_call(outer_model, parameter_and_buffer_dicts=outer_NN_dic, args=X, strict=True))
    # Get the value of g(Z)
    instrumental_feature = g_z_in
    feature = augment_stage1_feature(instrumental_feature)
    loss = linear_reg_loss(treatment_feature, feature, lam_V)
    return loss

# Inner objective function
def fi_function_of_h(outer_param, h_Z, X):
    # Get the value of f(X) outer
    outer_NN_dic = tensor_to_state_dict(outer_model, outer_param, device)
    treatment_feature = (torch.func.functional_call(outer_model, parameter_and_buffer_dicts=outer_NN_dic, args=X, strict=True))
    if h_Z.size(1) != treatment_feature.size(1):
        treatment_feature = augment_stage1_feature(treatment_feature)
    loss = torch.norm((treatment_feature - h_Z)) ** 2# + lam_V * torch.norm(weight) ** 2
    return loss

# Optimize using neural implicit differention
bp_neural = BilevelProblem(fo, fi, outer_dataloader, inner_dataloader, outer_models, inner_models, device, batch_size=batch_size, max_inner_epochs=max_inner_epochs, max_inner_dual_epochs=max_inner_dual_epochs, args=[lam_u, lam_V, a_star_method, fi_function_of_h])
# Solve the bilevel problem
iters, outer_losses, inner_losses, test_losses, times = bp_neural.optimize(outer_param, max_epochs=max_epochs, eval_every_n=eval_every_n, validation_data=validation_data, test_data=test_data)