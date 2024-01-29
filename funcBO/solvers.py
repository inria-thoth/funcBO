import torch
import torch.nn as nn
from funcBO.utils import config_to_instance, compute_batch_hessian


class Solver:
  """
  Base class for solvers.
  """
  def __init__(self, model, objective, reg=0.):
    self.model = model
    self.objective = objective
    self.reg = reg
    self.init_logs()

  def init_logs(self):
    self.data_logs = []

  def run(self):
    raise NotImplementedError

  """def add_reg(self, loss, params):
    reg_term = 0.
    if self.reg:
      reg_term = 0.5*self.reg*torch.sum(torch.cat([torch.sum(param)**2 for param in params],axis=0))
    return loss + reg_term"""

  def add_ridge(self,hessian):
    hessian += self.reg*torch.eye(hessian.shape[0], dtype= hessian.dtype, device= hessian.device)
    return hessian


class IterativeSolver(Solver):
  """
  Iterative solver.
  """
  def __init__(self, model, objective, optimizer, scheduler=None, reg=0, num_iter=1):
    super(IterativeSolver, self).__init__(model, objective, reg=reg)
    self.optimizer = config_to_instance(**optimizer, params=model.parameters())
    if scheduler:
      self.scheduler = config_to_instance(**scheduler, optimizer=self.optimizer)
    self.num_iter = num_iter

  def run(self, *objective_args):
    self.init_logs()
    for i in range(self.num_iter):
      self.optimizer.zero_grad()
      loss = self.objective(self.model, *objective_args)
      self.objective.outer_model.eval()
      self.data_logs.append({'inner_iter': i,
                              'loss': loss.item()})
      loss.backward()
      self.optimizer.step()


class CompositeSolver(Solver):
  """
  Iterative solver that finds W* when a*( ) is of the form a*( ) = W* h( ).
  """
  def __init__(self, model, objective, reg_objective, optimizer, scheduler=None, num_iter=1):
    super(CompositeSolver, self).__init__(model, objective)
    self.optimizer = config_to_instance(**optimizer, params=model.parameters())
    if scheduler:
      self.scheduler =  config_to_instance(**scheduler, optimizer=self.optimizer)
    self.num_iter = num_iter
    self.reg_objective = reg_objective

  def run(self, *objective_args):
    self.init_logs()
    for i in range(self.num_iter):
      self.optimizer.zero_grad()
      data = self.objective.get_data()
      inner_model_inputs, outer_model_inputs, inner_loss_inputs = self.objective.data_projector(data)
      # inner_model_outputs are the features phi(Z) (last 'model' corresponds to sequential object in the NN architecture)
      inner_model_outputs = self.model.model(inner_model_inputs)
      outer_model_outputs = self.objective.outer_model(outer_model_inputs)
      loss, weight = self.reg_objective(outer_model_outputs, inner_model_outputs)
      self.objective.outer_model.eval()
      self.data_logs.append({'inner_iter': i,
                              'loss': loss.item()})
      print(loss.item())
      loss.backward()
      self.optimizer.step()
      # Here we do one more fit to get the closed-form weights of the last linear layer
#      if i == self.num_iter-1:
        # Set last linear layer weights with their closed form values
#        self.model.linear.weight.data = self.fit_weights(inner_model_inputs, 
#                                                       inner_loss_inputs,
#                                                        *objective_args)


  def run_last_fit(self,*objective_args):
      # setting inner model to eval and outer model to train. 
      data = self.objective.get_data(use_previous_data=True)
      inner_model_inputs, outer_model_inputs, inner_loss_inputs = self.objective.data_projector(data)
      inner_model_outputs = self.model.model(inner_model_inputs)
      outer_model_outputs = self.objective.outer_model(outer_model_inputs)
      loss, weight = self.reg_objective(outer_model_outputs, inner_model_outputs)
      self.model.linear.weight.data=weight.t().detach()





class ClosedFormSolver(Solver):
  """
  Solver that finds a*( ) in closed form.
  """
  def __init__(self, model, objective, reg = 0):
    super(ClosedFormSolver, self).__init__(model, objective, reg=reg)

  def run(self,*objective_args):
    # Solving for the weights of the last linear layer
    hessian, B = self.make_linear_system(*objective_args)
    weight_shape = B.shape
    B = B.flatten()
    hessian = hessian.flatten(start_dim=2)
    hessian = torch.permute(hessian,(2, 0, 1))
    hessian = hessian.flatten(start_dim=1)
    hessian = torch.permute(hessian,(1, 0))
    hessian = self.add_ridge(hessian)
    W = -torch.linalg.solve(hessian, B)
    W = torch.unflatten(W, dim=0, sizes=weight_shape)
    weights = list(self.model.parameters())
    weights[0].data = W.detach()
  


  def make_linear_system(self,
                               dual_model_inputs, 
                               outer_grad):
    data = self.objective.objective.get_data(use_previous_data=True)
    inner_model_inputs, outer_model_inputs, inner_loss_inputs =  self.objective.objective.data_projector(data)
    inner_model = self.objective.inner_model
    inner_model.eval()
    with torch.no_grad():
      inner_model_output = inner_model(inner_model_inputs)
      outer_model_outputs = self.objective.outer_model(outer_model_inputs)
      _,inner_features = self.model(inner_model_inputs,with_features=True)
      _,outer_features = self.model(dual_model_inputs,with_features=True)
      B_inner = inner_features.shape[0]
      B_outer = outer_features.shape[0]
    inner_model_output = inner_model_output.flatten(start_dim=1)

    hessian = compute_batch_hessian(self.objective.objective.inner_loss,
                                          outer_model_outputs, 
                                          inner_model_output, 
                                          inner_loss_inputs)

    hessian = (1/B_inner)*torch.einsum('bij, bc, bd->icjd', hessian, inner_features, inner_features)
    
    outer_grad = outer_grad.flatten(start_dim=1)
    B = (1/B_outer)*torch.einsum('bi, bc->ic',outer_grad,outer_features)
    return hessian, B


class IVClosedFormSolver(Solver):
  """
  Solver that finds a*( ) in closed form.
  """
  def __init__(self, model, objective,reg=0):
    super(IVClosedFormSolver, self).__init__(model, objective,reg=reg)
    self.diag_hessian_inner = None
  
  def run(self,*objective_args):
    # Solving for the weights of the last linear layer
    hessian, B = self.make_linear_system(*objective_args)
    weight_shape = B.shape
    #hessian = hessian.double()
    H_in = torch.inverse(hessian)
    #H_in = H_in.float()
    W = -torch.einsum('dc, ic->id', H_in, B)
    weights = list(self.model.parameters())
    weights[0].data = W.detach()

  def make_linear_system(self,
                                dual_model_inputs, 
                                outer_grad):
      data = self.objective.objective.get_data(use_previous_data=True)
      inner_model_inputs, outer_model_inputs, inner_loss_inputs =  self.objective.objective.data_projector(data)
      inner_model = self.objective.inner_model
      inner_model.eval()
      with torch.no_grad():
        inner_model_output = inner_model(inner_model_inputs)
        outer_model_outputs = self.objective.outer_model(outer_model_inputs)
        _,inner_features = self.model(inner_model_inputs,with_features=True)
        _,outer_features = self.model(dual_model_inputs,with_features=True)
        B_inner = inner_features.shape[0]
        B_outer = outer_features.shape[0]
      
      hessian = (1/B_inner)*torch.einsum('bc, bd->cd',inner_features,inner_features)
      outer_grad = outer_grad.flatten(start_dim=1)
      B = (1/B_outer)*torch.einsum('bi, bc->ic',outer_grad,outer_features)
      
      if not self.diag_hessian_inner:
        hessian_inner = compute_batch_hessian(self.objective.objective.inner_loss,
                                          outer_model_outputs, 
                                          inner_model_output, 
                                          inner_loss_inputs)
        self.diag_hessian_inner = hessian_inner[0,0,0]
      hessian *= self.diag_hessian_inner
      hessian = self.add_ridge(hessian)

      return hessian, B

