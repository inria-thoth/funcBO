import sys
import time
import torch
import torch.nn as nn
import wandb
import numpy as np
from torch.nn.utils import spectral_norm
from torch.nn import functional as func
from statistics import mean

# Add main project directory path
sys.path.append('/home/clear/ipetruli/projects/bilevel-optimization/src')
sys.path.append('/home/clear/ipetruli/packages/miniconda3/lib/python3.10/site-packages')

from model.InnerSolution.InnerSolution import InnerSolution
from sklearn.metrics import accuracy_score
from model.utils import tensor_to_state_dict
from torchviz import make_dot
from my_data.dsprite.trainer import *

# Loss helper functions
MSE = nn.MSELoss()

class BilevelProblem:
  """
  Instanciates the bilevel problem and solves it using Neural Implicit Differentiation.
  """

  def __init__(self, outer_loss, inner_loss, outer_dataloader, inner_dataloader, outer_model, inner_models, device, batch_size=64, max_inner_epochs=100, max_inner_dual_epochs=100, args=None):
    """
    Init method.
      param outer_loss: outer level loss function
      param inner_loss: inner level loss function
      param outer_dataloader: input data and labels for the outer loss
      param inner_dataloader: input data and labels for the inner loss
      param batch_size: batch size for both inner and outer optimizers
      param device: CPU or GPU
    """
    self.__input_check__(outer_loss, inner_loss, outer_dataloader, inner_dataloader, outer_model, inner_models, device, batch_size)
    self.outer_loss = outer_loss
    self.inner_loss = inner_loss
    self.outer_dataloader = outer_dataloader
    self.inner_dataloader = inner_dataloader
    self.batch_size = batch_size
    self.device = device
    self.outer_model, self.outer_optimizer, self.outer_scheduler = outer_model
    self.inner_solution = InnerSolution(inner_loss, inner_dataloader, inner_models, device, batch_size, max_epochs=max_inner_epochs, max_dual_epochs=max_inner_dual_epochs, args=args, outer_model=self.outer_model)
    self.lam_u = args[0]
    self.lam_V = args[1]

  def __input_check__(self, outer_loss, inner_loss, outer_dataloader, inner_dataloader, outer_model, inner_models, device, batch_size):
    """
    Ensure that the inputs are of the right form and all necessary inputs have been supplied.
    """
    if (outer_loss is None) or (inner_loss is None):
      raise AttributeError("You must specify inner and outer losses.")
    if (outer_dataloader is None) or (inner_dataloader is None):
      raise AttributeError("You must give outer and inner dataloaders.")
    if (outer_model is None) or not(len(outer_model)==3):
      raise AttributeError("You must specify the outer model and its optimizer.")
    if (inner_models is None) or not(len(inner_models)==6):
      raise AttributeError("You must specify both inner and dual inner models and their optimizers.")
    if (device is None):
      raise AttributeError("You must specify a device: GPU/CPU.")
    if not (type(batch_size) is int):
      raise TypeError("Batch size must be an integer value.")

  def optimize(self, outer_param, max_epochs=100, eval_every_n=10, validation_data=None, test_data=None):
    """
    Find the optimal outer solution.
      param outer_param: initial value of the outer variable
      param maxiter: maximum number of iterations
    """
    iters, outer_losses, inner_losses, val_losses, times, evaluated, acc_smooth = 0, [], [], [], [], False, 10
    for epoch in range(max_epochs):
      for Z, X, Y in self.outer_dataloader:
        start = time.time()
        # Move data to GPU
        Z_outer = Z.to(self.device, dtype=torch.float64)
        X_outer = X.to(self.device, dtype=torch.float64)
        Y_outer = Y.to(self.device, dtype=torch.float64)
        # Inner value corresponds to h*(Z)
        outer_param.requires_grad = True
        forward_start = time.time()
        # Here is the only place where we need to optimize the inner solution
        self.inner_solution.optimize_inner = True
        # Get the value of h*(Z_outer)
        inner_value = self.inner_solution.forward(outer_param, Z_outer, Y_outer)
        # Make sure that the inner solution is not optimized
        self.inner_solution.optimize_inner = False
        self.inner_solution.model.train(False)
        self.inner_solution.dual_model.train(False)
        #wandb.log({"duration of forward": time.time() - forward_start})
        loss, u = self.outer_loss(outer_param, inner_value, Y_outer)
        # For checking the computational <autograd> graph.
        #make_dot(loss, params={ "outer param.":outer_param}, show_attrs=True, show_saved=True).render("graph_loss", format="png")
        #exit(0)
        #wandb.log({"inn. loss": self.inner_solution.loss})
        #wandb.log({"inn. dual loss": self.inner_solution.dual_loss})
        #wandb.log({"out. loss": loss.item()})
        # Backpropagation
        self.outer_optimizer.zero_grad()
        backward_start = time.time()
        loss.backward()
        wandb.log({"outer var. norm": torch.norm(outer_param).item()})
        wandb.log({"outer var. grad. norm": torch.norm(outer_param.grad).item()})
        self.outer_optimizer.step()
        #wandb.log({"duration of backward": time.time() - backward_start})
        # Update loss and iteration count
        iters += 1
        duration = time.time() - start
        #wandb.log({"iter. time": duration})
        times.append(duration)
        # Inner losses
        inner_losses.append(self.inner_solution.loss)
        # Outer losses
        outer_losses.append(loss.item())
        # Evaluate on validation data and check the stopping condition
      if (not (validation_data is None)) and (iters % eval_every_n == 0):#(not evaluated) and (len(val_accs)>acc_smooth) and (mean(val_accs[-acc_smooth:]) <= mean(val_accs[-(acc_smooth*2):-(acc_smooth)])):
        wandb.log({"val. loss": self.evaluate(validation_data, outer_param, last_layer=u)})
      if (not (test_data is None)):
        wandb.log({"test loss": self.evaluate(test_data, outer_param, last_layer=u)})
        #evaluated = True
    return iters, outer_losses, inner_losses, val_losses, times

  def evaluate(self, data, outer_param, outer_model=None, last_layer=None):
    """
    Evaluate the prediction quality on the test dataset.
      param data: data to evaluate on
      param outer_param: outer variable
    """
    if outer_model is None:
      outer_model = self.outer_model
    self.outer_model.train(False)
    self.inner_solution.model.train(False)
    self.inner_solution.dual_model.train(False)
    with torch.no_grad():
      Y = (torch.from_numpy(data.outcome)).to(self.device, dtype=torch.float64)
      X = (torch.from_numpy(data.treatment)).to(self.device, dtype=torch.float64)
      outer_NN_dic = tensor_to_state_dict(outer_model, outer_param, self.device)
      # Get the value of f(X)
      pred = torch.func.functional_call(outer_model, parameter_and_buffer_dicts=outer_NN_dic, args=X)
      if last_layer is None:
        loss = MSE(pred, Y)
      else:
        loss = MSE((pred @ last_layer[:-1] + last_layer[-1]), Y)
    return loss.item()