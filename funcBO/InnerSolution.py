import sys
import torch
import torch.nn as nn
#import wandb
from torch import autograd
from torch.autograd.functional import hessian, vjp, jvp, jacobian
from torch.func import jacrev

# Add main project directory path

from torch.func import functional_call
from torch.nn import functional as func
import funcBO
from funcBO.objectives import Objective, DualObjective
from funcBO.solvers import ClosedFormSolver
from funcBO.dual_networks import LinearDualNetwork


from funcBO.utils import config_to_instance

class InnerSolution(nn.Module):
  """
  Instanciates the inner solution of the bilevel problem.
  """

  def __init__(self, inner_model,
                     inner_loss, 
                     inner_dataloader,
                     inner_data_projector, 
                     outer_model,
                     inner_solver_args = {'name': 'funcBO.solver.IterativeSolver',
                                      'optimizer': {'name':'torch.optim.SGD'},
                                      'num_iter': 1},  
                     dual_solver_args = {'name': 'funcBO.solver.ClosedFormSolver'}, 
                     dual_model_args = {'name':'funcBO.dual_networks.LinearDualNetwork'},
                     ):
    super(InnerSolution, self).__init__()
    self.inner_model = inner_model
    assert isinstance(self.inner_model,nn.Module)
    dummpy_param = next(inner_model.parameters())
    self.device= dummpy_param.device
    self.dtype = dummpy_param.dtype

    self.inner_objective = Objective(inner_loss,
                                      inner_dataloader, inner_data_projector,
                                      self.device, self.dtype)

    self.inner_solver = config_to_instance(**inner_solver_args, 
                                      model= self.inner_model,
                                      objective=self.inner_objective)

    self.make_dual(dual_model_args, dual_solver_args)
    self.register_outer_parameters(outer_model)
    self.inner_loss = 0

  def make_dual(self,dual_model_args, dual_solver_args):
    dual_model_name = dual_model_args['name']
    inner_model_inputs = self.inner_objective.get_inner_model_input()

    if 'network' in dual_model_args:
      # if a custom network is provided it needs to have the same input and output spaces as the inner model. 
      network = config.pop('network')
      dual_out =  network(inner_model_inputs)
      inner_out = self.inner_model(inner_model_inputs)
      assert dual_out.shape==inner_out.shape
    else:
      # Add comment
      network = self.inner_model

    if dual_model_name=='funcBO.dual_networks.LinearDualNetwork':
      dual_model_args['network_inputs'] = inner_model_inputs

    self.dual_model = config_to_instance(**dual_model_args, 
                                          network=network)

    # Warning: here we really need inner model and not dual model
    self.dual_objective = DualObjective(self.inner_model, self.inner_objective)

    self.dual_solver = config_to_instance(**dual_solver_args,
                                      model=self.dual_model,
                                      objective=self.dual_objective)
    if isinstance(self.dual_solver, ClosedFormSolver):
      assert isinstance(self.dual_model, LinearDualNetwork)

  def register_outer_parameters(self,outer_model):
    if isinstance(outer_model,nn.Module):
      for name, param in outer_model.named_parameters():
        self.register_parameter(name="outer_param_"+name, param=param)
      self.outer_params = tuple(outer_model.parameters())
    elif isinstance(outer_model,tuple) and isinstance(outer_model[0], nn.parameter.Parameter):
      for i,param in enumerate(outer_model):
        self.register_parameter(name="outer_param_"+str(i), param=param)
      self.outer_params = outer_model
    elif isinstance(outer_model,nn.parameter.Parameter):
      self.register_parameter(name="outer_param", param=outer_model)

  def forward(self, inner_model_inputs):
    """
    Forward pass of a neural network that approximates the function h* for Neur. Imp. Diff.
      param outer_param: the current outer variable
      param Y_outer: the outer data that the dual model needs access to
    """
    # We use an intermediate ArgMinOp because we can only write a custom backward for functions
    # of type torch.autograd.Function, nn.Module doesn't allow to custumize the backward.
    
    if self.training:
      self.inner_model.train()
      return ArgMinOp.apply(self, self.outer_param, inner_model_inputs)
    else:
      with torch.no_grad():
        self.inner_model.eval()
        val = self.inner_model(inner_model_inputs)
        return val  
  
  def cross_derivative_dual_prod(self, outer_param, inner_model_inputs, inner_loss_inputs):
    self.inner_model.eval()
    self.dual_model.eval()
    with torch.no_grad():
      inner_value = self.inner_model(inner_model_inputs)
      dual_value = self.dual_model(inner_model_inputs)
    inner_value.requires_grad = True

    f = lambda outer_param, inner_value: self.inner_objective.inner_loss(outer_param, inner_value, inner_loss_inputs)
    # Here v has to be a tuple of the same shape as the args of f, so we put a zero vector and a*(X) into a tuple.
    # Here args has to be a tuple with args of f, so we put outer_param and h*(X) into a tuple.
    loss = f(outer_param, inner_value)
    grad = autograd.grad(
                  outputs=loss, 
                  inputs=inner_value, 
                  grad_outputs=None, 
                  retain_graph=True,
                  create_graph=True, 
                  only_inputs=True,
                  allow_unused=True)[0]
    dot_prod = torch.sum(grad*dual_value)
    cdvp = autograd.grad(
                  outputs=dot_prod, 
                  inputs=outer_param, 
                  grad_outputs=None, 
                  retain_graph=False,
                  create_graph=False, 
                  only_inputs=True,
                  allow_unused=True)[0]
    return cdvp


class ArgMinOp(torch.autograd.Function):
  """
  A pure function that approximates h*.
  """

  @staticmethod
  def forward(ctx, inner_solution, outer_param, inner_model_inputs):
    """
    Forward pass of a function that approximates h* for Neur. Imp. Diff.
    """
    # In forward autograd is disabled by default but we use it in optimize(outer_param).
    inner_model = inner_solution.inner_solver.model
    inner_model.train()
    with torch.enable_grad():
      # Train the model to approximate h* at outer_param_k
      inner_solution.inner_solver.run(outer_param)
      # Put optimize to False?
    # Remember the value h*(Z_outer)
    inner_model.eval()
    with torch.no_grad():
        inner_value = inner_model(inner_model_inputs)
    # Context ctx allows to communicate from forward to backward
    ctx.inner_solution = inner_solution
    ctx.save_for_backward(outer_param, inner_model_inputs, inner_value)
    return inner_value 

  @staticmethod
  def backward(ctx, outer_grad):
    # Computing the gradient of theta (param. of outer model) in closed form.
    # Context ctx allows to communicate from forward to backward
    inner_solution = ctx.inner_solution
    # Get the saved tensors
    outer_param, inner_model_inputs, inner_value = ctx.saved_tensors
    # Get the inner Z and X
    # Need to enable_grad because we use autograd in optimize_dual (disabled in backward() by default).
    with torch.enable_grad():
      # Here the model approximating a* needs to be trained on the same X_inner batches
      # as the h* model was trained on and on X_outer batches that h was evaluated on
      # in the outer loop where we optimize the outer objective g(outer_param, h).
      inner_solution.dual_solver.run(outer_param, inner_model_inputs, outer_grad)
    with torch.no_grad():  
      data = inner_solution.inner_objective.get_data(use_previous_data=True)
      inner_model_inputs, inner_loss_inputs =  inner_solution.inner_objective.data_projector(data)
    with torch.enable_grad():
      grad = inner_solution.cross_derivative_dual_prod(outer_param, inner_model_inputs, inner_loss_inputs)

    return None, grad, None
