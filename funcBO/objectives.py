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
                    inner_dataloader, data_projector,
                    outer_model,
                    device, dtype):
      """
      inner_loss must be a function of the form g(theta, Z, Y) where 
      theta is the outer parameter, Z = f(X) is the output of the inner model f() given a data input X and   
      Y is additional input. Both X,Y are obtained by first getting a sample 'U' from the dataloader 
      and then applying the data_projector to it: X,Y = data_projector(U).
      """
      self.inner_loss = inner_loss
      self.data_projector = data_projector
      self.outer_model = outer_model
      self.device= device
      self.dtype = dtype
      self.inner_dataloader = RingGenerator(inner_dataloader, self.device, self.dtype)
      self.data = None

  def get_data(self, use_previous_data=False):
      if use_previous_data and self.data:
        data = self.data
      else: 
        data = next(self.inner_dataloader)
        self.data = data
      return data    

  def __call__(self,inner_model):
      data = self.get_data()
      inner_model_inputs, outer_model_inputs, inner_loss_inputs =  self.data_projector(data)
      func_val = inner_model(inner_model_inputs)
      outer_model_val = self.outer_model(outer_model_inputs)
      if inner_loss_inputs is not None:
        loss = self.inner_loss(outer_model_val,func_val)
      else:
        loss = self.inner_loss(outer_model_val, func_val)

      return loss

  def get_inner_model_input(self):
      data = next(self.inner_dataloader)
      inner_model_inputs,outer_model_inputs, inner_loss_inputs =  self.data_projector(data)
      return inner_model_inputs


class DualObjective:
  def __init__(self, inner_model,
                objective,
                reg=0.):
    self.objective = objective
    self.outer_model = objective.outer_model
    self.inner_model = inner_model
    self.reg = reg

  def __call__(self, dual_model, 
                    dual_model_inputs, 
                    outer_grad):
    """
    Loss function for optimizing a*.
    """
    # Specifying the inner objective as a function of h*(X)
    data = self.objective.get_data(use_previous_data=True)
    inner_model_inputs, outer_model_inputs, inner_loss_inputs =  self.objective.data_projector(data)

    self.inner_model.eval()
    self.outer_model.eval()
    with torch.no_grad():
      inner_model_output = self.inner_model(inner_model_inputs)
      outer_model_val    = self.outer_model(outer_model_inputs)

    if inner_loss_inputs is not None:
        f = lambda inner_model_output: self.objective.inner_loss(outer_model_val, inner_model_output, inner_loss_inputs)
    else:
        f = lambda inner_model_output: self.objective.inner_loss(outer_model_val, inner_model_output)

    
    
    # Find the product of a*(X) with the hessian wrt h*(X)
    dual_val_inner = dual_model(inner_model_inputs)
    dual_val_outer = dual_model(dual_model_inputs)
    B_inner = dual_val_inner.shape[0]
    B_outer = dual_val_outer.shape[0]
    hessvp = autograd.functional.hvp(f, inner_model_output, dual_val_inner)[1]

    # Compute the loss
    term1 = (1/B_inner)*(torch.einsum('b...,b...->', dual_val_inner, hessvp))
    term2 = (1/B_outer)*torch.einsum('b...,b...->', dual_val_outer, outer_grad)

    loss = term1 + term2 
    return loss



