import torch
import torch.nn as nn
from funcBO.utils import config_to_instance

import wandb

class Solver:
  """
  Base class for solvers.
  """
  def __init__(self,model, objective):
    self.model = model
    self.objective = objective
  def run(self,outer_param):
    raise NotImplementedError

class IterativeSolver(Solver):
  """
  Iterative solver.
  """
  def __init__(self, model,objective, optimizer, scheduler=None,num_iter=1):
    super(IterativeSolver, self).__init__(model, objective)
    self.optimizer = config_to_instance(**optimizer, params=model.parameters())
    if scheduler:
      self.scheduler =  config_to_instance(**scheduler, optimizer=self.optimizer)
    self.num_iter = num_iter

  def run(self,*objective_args):
    for i in range(self.num_iter):
      self.optimizer.zero_grad()
      loss = self.objective(self.model,*objective_args)
      wandb.log({"in. loss": loss.item()})
      loss.backward()
      self.optimizer.step()

class CompositeSolver(Solver):
  """
  Iterative solver that finds W* when a*( ) is of the form a*( ) = W* h( ).
  """
  def __init__(self, model, objective, reg_objective, optimizer, scheduler=None,num_iter=1):
    super(CompositeSolver, self).__init__(model, objective)
    self.optimizer = config_to_instance(**optimizer, params=model.parameters())
    if scheduler:
      self.scheduler =  config_to_instance(**scheduler, optimizer=self.optimizer)
    self.num_iter = num_iter
    self.reg_objective = reg_objective
  def run(self,*objective_args):
    for i in range(self.num_iter):
      self.optimizer.zero_grad()
      data = self.objective.get_data()
      inner_model_inputs, inner_loss_inputs =  self.objective.data_projector(data)
      # These are the features (last 'model' corresponds to sequential object in the NN architecture)
      inner_model_outputs = self.model.model(inner_model_inputs)
      loss, weight = self.reg_objective(*objective_args, inner_model_outputs, inner_loss_inputs)
      wandb.log({"in. loss": loss.item()})
      #loss = self.objective(self.model,*objective_args)
      loss.backward()
      self.optimizer.step()
      # Here we do one more fit to get the closed-form weights of the last linear layer
      if i == self.num_iter-1:
        with torch.no_grad():
          self.model.eval()

          inner_model_outputs = self.model.model(inner_model_inputs)
          loss, weight = self.reg_objective(*objective_args, inner_model_outputs, inner_loss_inputs)
      # Set last linear layer weights with their closed form values
      self.model.linear.weight.data = weight.t().detach()

class ClosedFormSolver(Solver):
  """
  Solver that finds a*( ) in closed form.
  """
  def __init__(self, model, objective, reg=0.):
    super(ClosedFormSolver, self).__init__(model, objective)
    self.reg = reg
  def run(self,*objective_args):
    hessian, B = self.objective.make_linear_system(self.model, *objective_args)
    weight_shape = B.shape
    B = B.flatten()
    hessian = hessian.flatten(start_dim=2)
    hessian = torch.permute(hessian,(2, 0, 1))
    hessian = hessian.flatten(start_dim=1)
    hessian = torch.permute(hessian,(1, 0))
    hessian += self.reg*torch.eye(hessian.shape[0], dtype= hessian.dtype, device= hessian.device)
    W = -torch.linalg.solve(hessian, B)
    W = torch.unflatten(W, dim=0, sizes=weight_shape)
    weights = list(self.model.parameters())
    weights[0].data = W.detach()

class IVClosedFormSolver(Solver):
  """
  Solver that finds a*( ) in closed form.
  """
  def __init__(self, model, objective, reg=0.):
    super(IVClosedFormSolver, self).__init__(model, objective)
    self.reg = reg
  def run(self,*objective_args):
    hessian, B = self.objective.make_reduced_linear_system(self.model, *objective_args)
    weight_shape = B.shape
    hessian += self.reg*torch.eye(hessian.shape[0], dtype= hessian.dtype, device= hessian.device)
    H_in = torch.inverse(hessian)
    W = -0.5*torch.einsum('dc, ic->id', H_in, B)
    weights = list(self.model.parameters())
    weights[0].data = W.detach()