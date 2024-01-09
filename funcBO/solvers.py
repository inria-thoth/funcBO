import torch
import torch.nn as nn
from funcBO.utils import config_to_instance



class Solver:
  def __init__(self,model, objective):
    self.model = model
    self.objective = objective
  def run(self,outer_param):
    raise NotImplementedError

class IterativeSolver(Solver):
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
      loss.backward()
      self.optimizer.step()


class CompositeSolver(Solver):

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
      inner_model_outputs = self.model.model(inner_model_inputs)
      loss, weight = self.reg_objective(*objective_args,inner_model_outputs,inner_loss_inputs)
      
      #loss = self.objective(self.model,*objective_args)
      loss.backward()
      self.optimizer.step()
      self.model.linear.weight.data = weight.t().detach()





class ClosedFormSolver(Solver):
  def __init__(self,model,objective,reg=0.):
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
    W = torch.unflatten(W,dim=0,sizes=weight_shape)
    weights = list(self.model.parameters())
    weights[0].data = W.detach()


    






