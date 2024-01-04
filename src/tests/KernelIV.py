import sys
import torch
import wandb
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import MSELoss

# Add main project directory path
sys.path.append('/home/clear/ipetruli/projects/bilevel-optimization/src')

from model.utils import *
# The experiment-specific functions
from model_previous.BilevelProblem.BilevelProblem import BilevelProblem
from my_data.dsprite.dspriteKernel import *

# Set seed
seed = 42
set_seed(seed)

# Setting the device to GPUs if available.
if torch.cuda.is_available():
    device = "cuda"
    print("All good, switching to GPUs.")
else:
    device = "cpu"
    print("No GPUs found, setting the device to CPU.")


# Setting hyper-parameters
batch_size = 2500
max_epochs = 5000
max_outer_iters = 100
max_inner_iters = 20
eval_every_n = 100
lam1 = 0.1
lam2 = 0.1
gamma = 0.04
stepsize = 1e-1

# Get data
test_data = generate_test_dsprite(device=device)
train_data, validation_data = generate_train_dsprite(data_size=5000, rand_seed=seed, device=device, val_size=0)
inner_data, outer_data = split_train_data(train_data, split_ratio=0.5)
instrumental_in, treatment_in, outcome_in, instrumental_out, treatment_out, outcome_out, treatment_test, outcome_test = inner_data.instrumental, inner_data.treatment, inner_data.outcome, outer_data.instrumental, outer_data.treatment, outer_data.outcome, test_data.treatment, test_data.structural
#instrumental_val, treatment_val, outcome_val = validation_data.instrumental, validation_data.treatment, validation_data.outcome

# Gather all data
data = (torch.from_numpy(instrumental_in), torch.from_numpy(treatment_in), torch.from_numpy(outcome_in), torch.from_numpy(instrumental_out), torch.from_numpy(treatment_out), torch.from_numpy(outcome_out), torch.from_numpy(treatment_test), torch.from_numpy(outcome_test))
#validation_data = (instrumental_val, treatment_val, outcome_val)

# Print configuration
run_name = "Dsprites KernelIV::="+" gamma:"+str(gamma)+", lam1:"+str(lam1)+", lam2:"+str(lam2)+", batch_size:"+str(batch_size)+", max_epochs:"+str(max_epochs)+", stepsize:"+str(stepsize)
print("Run configuration:", run_name)

# Set logging
wandb.init(group="Dsprites_KernelIV", name=run_name)

# Loss helper functions
MSE = nn.MSELoss()
K_inner_Z = rbf_kernel(data[0], data[0], gamma)
K_inner_X = rbf_kernel(data[1], data[1], gamma)

# Outer objective function
def fo(outer_param, g_z_out, Y):
    wandb.log({"out. loss": MSE(g_z_out, Y).item()})
    loss = (1/2)*(torch.norm(Y - g_z_out))**2 + lam1/2 * (outer_param.T @ K_inner_X @ outer_param)
    return loss

# Inner objective function
def fi(outer_param, value, K_X):
    f_x_in = K_X @ outer_param
    wandb.log({"inn. loss": MSE(f_x_in, value).item()})
    loss = (1/2)*(torch.norm(f_x_in - value))**2 + lam2/2 * (value.T @ torch.linalg.lstsq(K_inner_Z, value)[0])
    return loss

mu0 = torch.rand((len(K_inner_X), 1), dtype=torch.float64) - 0.5

# Optimize using kernel implicit differention
bp_neural = BilevelProblem(fo, fi, "kernel_implicit_diff", data, batch_size=batch_size, reg_param=(lam1, lam2), gamma=gamma)
mu_opt, iters, n_iters, times, inner_loss, outer_loss, h_star = bp_neural.train(mu0=mu0, maxiter=max_epochs, step=stepsize)
print("mu:", mu_opt)