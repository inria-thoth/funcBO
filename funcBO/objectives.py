import torch
import torch.nn as nn

from torch import autograd
from torch.autograd.functional import hessian, vjp, jvp, jacobian
from torch.func import jacrev

# Add main project directory path

from torch.func import functional_call
from torch.nn import functional as func
from funcBO.utils import RingGenerator




class Objective:
  """
  
  """
  def __init__(self,inner_loss, 
                    inner_dataloader,data_projector,
                    device, dtype):
      """
      inner_loss must be a function of the form g(theta, Z, Y) where 
      theta is the outer parameter, Z = f(X) is the output of the inner model f() given a data input X and   
      Y is additional input. Both X,Y are obtained by first getting a sample 'U' from the dataloader 
      and then applying the data_projector to it: X,Y = data_projector(U).

      """
      self.inner_loss = inner_loss
      self.data_projector = data_projector
      self.device= device
      self.dtype = dtype
      self.inner_dataloader = RingGenerator(inner_dataloader, self.device, self.dtype)
      #self.use_previous_data = False
      self.data = None
  def get_data(self,use_previous_data=False):
      if use_previous_data and self.data:
        data = self.data
      else: 
        data = next(self.inner_dataloader)
        self.data = data
      return data    

  def __call__(self,inner_model,outer_param):
      data = self.get_data()
      inner_model_inputs, inner_loss_inputs =  self.data_projector(data)
      func_val = inner_model(inner_model_inputs)
      if inner_loss_inputs:
        loss = self.inner_loss(outer_param, func_val, inner_loss_inputs)
      else:
        loss = self.inner_loss(outer_param, func_val)

      return loss

  def get_inner_model_input(self):
      data = next(self.inner_dataloader)
      inner_model_inputs, inner_loss_inputs =  self.data_projector(data)
      return inner_model_inputs
#  def eval_model(self,inner_model_inputs):
#      self.inner_model.eval()
#      with torch.no_grad():
#        return self.inner_model(inner_model_inputs)



class DualObjective:
  def __init__(self, objective, inner_model):
    self.objective = objective
    self.inner_model = inner_model

  def __call__(self, dual_model, 
                    outer_param, 
                    dual_model_inputs, 
                    outer_grad):
    """
    Loss function for optimizing a*.
    """
    # Specifying the inner objective as a function of h*(X)
    data = self.objective.get_data(use_previous_data=True)
    inner_model_inputs, inner_loss_inputs =  self.objective.data_projector(data)

    self.inner_model.eval()
    with torch.no_grad():
      inner_model_output = self.inner_model(inner_model_inputs)

    f = lambda inner_model_output: self.objective.inner_loss(outer_param, inner_model_output, inner_loss_inputs)
    # Find the product of a*(X) with the hessian wrt h*(X)
    dual_val_inner = dual_model(inner_model_inputs)
    dual_val_outer = dual_model(dual_model_inputs)
    hessvp = autograd.functional.hvp(f, inner_model_output, dual_val_inner)[1]
    #hessvp = hessvp.detach()
    # Compute the loss
    term1 = (torch.einsum('b...,b...->', dual_val_inner, hessvp))
    term2 = torch.einsum('b...,b...->', dual_val_outer, outer_grad)
    #assert(term1.size() == (inner_model_output.size()[0],))
    #assert(term1.size() == term2.size())
    loss = term1 + term2
    return loss

  def make_linear_system(self, dual_model,
                               outer_param,
                               dual_model_inputs, 
                               outer_grad):
    data = self.objective.get_data(use_previous_data=True)
    inner_model_inputs, inner_loss_inputs =  self.objective.data_projector(data)
    inner_model = self.inner_model
    inner_model.eval()
    with torch.no_grad():
      inner_model_output = inner_model(inner_model_inputs)
      _,inner_features = dual_model(inner_model_inputs,with_features=True)
      _,outer_features = dual_model(dual_model_inputs,with_features=True)

    output_shape = inner_model_output.shape[1:]
    inner_model_output = inner_model_output.flatten(start_dim=1)
    def loss(inner_model_output, inner_loss_inputs):
      if len(output_shape)>1:
        inner_model_output = torch.unflatten(inner_model_output,dim=0,sizes=output_shape)
      return self.objective.inner_loss(outer_param, inner_model_output, inner_loss_inputs)

    batch_hess = torch.func.vmap(torch.func.hessian(loss, argnums=0), in_dims= (0,0))
    hessian = batch_hess(inner_model_output,inner_loss_inputs)
    
    hessian = torch.einsum('bij, bc, bd->icjd',hessian,inner_features,inner_features)
    outer_grad = outer_grad.flatten(start_dim=1)
    B = torch.einsum('bi, bc->ic',outer_grad,outer_features)
    return hessian, B

  def make_reduced_linear_system(self, dual_model,
                                outer_param,
                                dual_model_inputs, 
                                outer_grad):
      data = self.objective.get_data(use_previous_data=True)
      inner_model_inputs, inner_loss_inputs =  self.objective.data_projector(data)
      inner_model = self.inner_model
      inner_model.eval()
      with torch.no_grad():
        inner_model_output = inner_model(inner_model_inputs)
        _,inner_features = dual_model(inner_model_inputs,with_features=True)
        _,outer_features = dual_model(dual_model_inputs,with_features=True)

      output_shape = inner_model_output.shape[1:]
      hessian = torch.einsum('bc, bd->cd',inner_features,inner_features)
      outer_grad = outer_grad.flatten(start_dim=1)
      B = torch.einsum('bi, bc->ic',outer_grad,outer_features)
      return hessian, B