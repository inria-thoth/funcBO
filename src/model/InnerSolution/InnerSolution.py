import sys
import torch
import torch.nn as nn
import wandb
from torch import autograd
from torch.autograd.functional import hessian, vjp, jvp, jacobian
from functorch import jacrev

# Add main project directory path
sys.path.append('/home/clear/ipetruli/projects/bilevel-optimization/src')

from torch.func import functional_call
from torch.nn import functional as func
from model.utils import get_memory_info, cos_dist, tensor_to_state_dict
from torchviz import make_dot
from my_data.dsprite.dspriteBilevel import OuterModel
from my_data.dsprite.trainer import augment_stage1_feature, augment_stage2_feature, linear_reg_pred, linear_reg_loss, fit_linear

device = "cuda" if torch.cuda.is_available() else "cpu"

class InnerSolution(nn.Module):
  """
  Instanciates the inner solution of the bilevel problem.
  """

  def __init__(self, inner_loss, inner_dataloader, inner_models, device, batch_size, max_epochs=100, max_dual_epochs=100, args=None, outer_model=None):
    """
    Init method.
      param inner_loss: inner level objective function
      param inner_dataloader: data loader for inner data
      param device: CPU or GPU
    """
    super(InnerSolution, self).__init__()
    self.inner_loss = inner_loss
    self.inner_dataloader = inner_dataloader
    self.model, self.optimizer, self.scheduler, self.dual_model, self.dual_optimizer, self.dual_scheduler = inner_models
    self.device = device
    self.loss = 0
    self.dual_loss = 0
    self.batch_size = batch_size
    self.max_epochs = max_epochs
    self.max_dual_epochs = max_dual_epochs
    self.optimize_inner = False
    self.lam_u = args[0]
    self.lam_V = args[1]
    self.a_star_method = args[2]
    self.outer_model = outer_model

  def forward(self, outer_param, Z_outer, Y_outer):
    """
    Forward pass of a neural network that approximates the function h* for Neur. Imp. Diff.
      param outer_param: the current outer variable
      param Y_outer: the outer data that the dual model needs access to
    """
    # We use an intermediate ArgMinOp because we can only write a custom backward for functions
    # of type torch.autograd.Function, nn.Module doesn't allow to custumize the backward.
    opt_inner_value = ArgMinOp.apply(self, outer_param, Z_outer, Y_outer)
    return opt_inner_value

  def optimize(self, outer_param):
    """
    Optimization loop for the inner-level model that approximates h*.
    """
    total_loss, total_epochs, total_iters = 0, 0, 0
    for epoch in range(self.max_epochs):
      total_epoch_loss, total_epoch_iters = 0, 0
      for Z, X, Y in self.inner_dataloader:
        # Move data to GPU
        Z = Z.to(self.device, dtype=torch.float)
        X = X.to(self.device, dtype=torch.float)
        Y = Y.to(self.device, dtype=torch.float)
        # Get the prediction
        g_z_in = self.model.forward(Z)
        # Compute the loss
        loss = self.inner_loss(outer_param, g_z_in, X)
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        total_epoch_loss += loss.item()
        total_iters += 1
        total_epoch_iters += 1
      epoch_loss = total_epoch_loss/total_epoch_iters
      total_loss += epoch_loss
      total_epochs += 1
      grad_norm = (sum([torch.norm(p.grad)**2 for p in self.model.parameters()]))
      wandb.log({"inner grad. norm": grad_norm})
      #print("instrumental_net first layer norm:", torch.norm(self.model.layer1.weight))
    self.loss = total_loss/total_epochs
  
  def optimize_dual_inH(self, outer_param, Z_outer, Y_outer, outer_grad):
    """
    Optimization loop for the inner-level model that approximates a*.
    """
    total_loss, total_epochs, total_iters = 0, 0, 0
    for epoch in range(self.max_dual_epochs):
      total_epoch_loss, total_epoch_iters = 0, 0
      for Z, X, Y in self.inner_dataloader:
        # Move data to GPU
        Z = Z.to(self.device, dtype=torch.float)
        X = X.to(self.device, dtype=torch.float)
        Y = Y.to(self.device, dtype=torch.float)
        # Get the predictions
        with torch.no_grad():
          self.evaluate = True
          h_X_i = self.model(Z)
          h_X_i.detach()
        a_X_i = self.dual_model(Z)
        a_X_o = self.dual_model(Z_outer)
        # Compute the loss
        loss = self.loss_H(outer_param, h_X_i, a_X_i, a_X_o, X, outer_grad)
        # Backpropagation
        self.dual_optimizer.zero_grad()
        loss.backward()
        self.dual_optimizer.step()
        total_epoch_loss += loss.item()
        total_iters += 1
        total_epoch_iters += 1
      epoch_loss = total_epoch_loss/total_epoch_iters
      total_loss += epoch_loss
      total_epochs += 1
      dual_grad_norm = (sum([torch.norm(p.grad)**2 for p in self.dual_model.parameters()]))
      wandb.log({"dual grad. norm": dual_grad_norm})
    self.dual_loss = total_loss/total_epochs
  
  def optimize_dual(self, outer_param, Z_outer, Y_outer, outer_grad):
    """
    Optimization loop for the inner-level model that approximates a*.
    """
    total_loss, total_epochs, total_iters = 0, 0, 0
    for epoch in range(self.max_dual_epochs):
      total_epoch_loss, total_epoch_iters = 0, 0
      for Z, X, Y in self.inner_dataloader:
        # Move data to GPU
        Z = Z.to(self.device, dtype=torch.float)
        X = X.to(self.device, dtype=torch.float)
        Y = Y.to(self.device, dtype=torch.float)
        # Get the predictions
        with torch.no_grad():
          h_X_i = self.model(Z)
          h_X_o = self.model(Z_outer)
          h_X_i.detach()
          h_X_o.detach()
        a_X_i = self.dual_model(h_X_i)
        a_X_o = self.dual_model(h_X_o)
        # Compute the loss
        loss = self.loss_H(outer_param, h_X_i, a_X_i, a_X_o, X, outer_grad)
        # Backpropagation
        self.dual_optimizer.zero_grad()
        loss.backward()
        self.dual_optimizer.step()
        total_epoch_loss += loss.item()
        total_iters += 1
        total_epoch_iters += 1
      epoch_loss = total_epoch_loss/total_epoch_iters
      total_loss += epoch_loss
      total_epochs += 1
      dual_grad_norm = (sum([torch.norm(p.grad)**2 for p in self.dual_model.parameters()]))
      wandb.log({"dual grad. norm": dual_grad_norm})
    self.dual_loss = total_loss/total_epochs

  def loss_H(self, outer_param, h_X_i, a_X_i, a_X_o, y_inner, outer_grad):
    """
    Loss function for optimizing a*.
    """
    # Specifying the inner objective as a function of h*(X)
    f = lambda h_X_i: self.inner_loss(outer_param, h_X_i, y_inner)
    # Find the product of a*(X) with the hessian wrt h*(X)
    hessvp = autograd.functional.hvp(f, h_X_i, a_X_i)[1]
    wandb.log({"torch.norm(a_X_i)": torch.norm(a_X_i)})
    wandb.log({"hessvp norm": torch.norm(hessvp)})
    # Compute the loss
    term1 = (torch.einsum('bi,bi->b', a_X_i, hessvp))#*(1/2)
    term2 = torch.einsum('bi,bi->b', a_X_o, outer_grad)
    assert(term1.size() == (h_X_i.size()[0],))
    assert(term1.size() == term2.size())
    loss = torch.mean(term1 + term2)
    return loss

  def compute_hessian_vector_prod(self, outer_param, Z_inner, X_inner, inner_value, dual_value):
    """
    Computing B*a where a is dual_value=a(Z_outer) and B is the functional derivative delta_{outer_param} delta_h g(outer_param,h*).
    """
    f = lambda outer_param, inner_value: self.inner_loss(outer_param, inner_value, X_inner)
    # Deatch items from the graph.
    dual_value.detach()
    inner_value.detach()
    # Here v has to be a tuple of the same shape as the args of f, so we put a zero vector and a*(X) into a tuple.
    v = (torch.zeros_like(outer_param), dual_value)
    # Here args has to be a tuple with args of f, so we put outer_param and h*(X) into a tuple.
    args = (outer_param, inner_value)
    hessvp = autograd.functional.hvp(f, args, v)[1][0]
    assert(hessvp.size() == outer_param.size())
    return hessvp


class ArgMinOp(torch.autograd.Function):
  """
  A pure function that approximates h*.
  """

  @staticmethod
  def forward(ctx, inner_solution, outer_param, Z_outer, Y_outer):
    """
    Forward pass of a function that approximates h* for Neur. Imp. Diff.
    """
    if inner_solution.optimize_inner:
      # In forward autograd is disabled by default but we use it in optimize(outer_param).
      with torch.enable_grad():
        # Train the model to approximate h* at outer_param_k
        inner_solution.model.train(True)
        inner_solution.optimize(outer_param)
        # Put optimize to False?
      # Remember the value h*(Z_outer)
      with torch.no_grad():
        inner_solution.model.eval()
        inner_value = inner_solution.model(Z_outer)
      # Context ctx allows to communicate from forward to backward
      ctx.inner_solution = inner_solution
      ctx.save_for_backward(outer_param, Z_outer, Y_outer, inner_value)
    else:
      with torch.no_grad():
        inner_solution.model.eval()
        inner_value = inner_solution.model(Z_outer)
    return inner_value

  @staticmethod
  def backward(ctx, outer_grad):
    #Computing the gradient of theta (param. of outer model) in closed form.
    # Context ctx allows to communicate from forward to backward
    inner_solution = ctx.inner_solution
    # Get the saved tensors
    outer_param, Z_outer, Y_outer, inner_value = ctx.saved_tensors
    # Get the inner Z and X
    for Z, X, Y in inner_solution.inner_dataloader:
      # Move data to GPU
      Z_inner = Z.to(device, dtype=torch.float)
      X_inner = X.to(device, dtype=torch.float)
      Y_inner = Y.to(device, dtype=torch.float)
    # Switch based on the method used to approximate a* or compute the gradient of outer variable.
    match inner_solution.a_star_method:
      # CASE : GD on a* = W phi(Z) (compare with autograd)
      case "GD":
        # Need to enable_grad because we use autograd in optimize_dual (disabled in backward() by default).
        with torch.enable_grad():
          # Here the model approximating a* needs to be trained on the same X_inner batches
          # as the h* model was trained on and on X_outer batches that h was evaluated on
          # in the outer loop where we optimize the outer objective g(outer_param, h).
          inner_solution.dual_model.train(True)
          inner_solution.optimize_dual(outer_param, Z_outer, Y_outer, outer_grad)
          inner_solution.dual_model.eval()
        with torch.no_grad():
          h_X_o = inner_solution.model(Z_outer)
          dual_value = inner_solution.dual_model(h_X_o)
          grad = inner_solution.compute_hessian_vector_prod(outer_param, Z_inner, X_inner, inner_value, dual_value)
      # CASE : GD on a* = NN_a(Z)
      case "GDinH":
        # Need to enable_grad because we use autograd in optimize_dual (disabled in backward() by default).
        with torch.enable_grad():
          # Here the model approximating a* needs to be trained on the same X_inner batches
          # as the h* model was trained on and on X_outer batches that h was evaluated on
          # in the outer loop where we optimize the outer objective g(outer_param, h).
          inner_solution.dual_model.train(True)
          inner_solution.optimize_dual_inH(outer_param, Z_outer, Y_outer, outer_grad)
          inner_solution.dual_model.eval()
        with torch.no_grad():
          dual_value = inner_solution.dual_model(Z_outer)
          grad = inner_solution.compute_hessian_vector_prod(outer_param, Z_inner, X_inner, inner_value, dual_value)
      # CASE : a* with W* from eq.63
      case "closed_form_a":
        # Get the predictions
        with torch.no_grad():
          h_X_i = inner_solution.model(Z_inner)
          h_X_o = inner_solution.model(Z_outer)
          h_X_i.detach()
          h_X_o.detach()
        # Find u*
        # Get the value of f(X) inner
        outer_NN_dic = tensor_to_state_dict(inner_solution.outer_model, outer_param, device)
        f_X_i = torch.func.functional_call(inner_solution.outer_model, parameter_and_buffer_dicts=outer_NN_dic, args=X, strict=True)
        res = fit_2sls_noprint(f_X_i, h_X_i, h_X_o, Y_inner, inner_solution.lam_V, inner_solution.lam_u)
        u = res["stage2_weight"]
        # Here a* is a matrix W multiplication with h(X) where W is the optimal weight matrix.
        sum1 = torch.zeros_like(u)
        h_X_o = augment_stage2_feature(h_X_o)
        for i in range(Y_outer.size(0)):
          uT_h_X_o = torch.einsum('ji,j->i', u, h_X_o[i])
          sum1 += (uT_h_X_o - Y_outer[i]) * (h_X_o[i]).unsqueeze(1)
        sum1 = -u @ sum1.T
        assert(sum1.size() == (u.size(0), u.size(0)))
        sum2 = torch.zeros_like(sum1)
        h_X_i = augment_stage1_feature(h_X_i)
        for i in range(Y_outer.size(0)):
          h_X_i_h_X_i_T = (h_X_i[i]).unsqueeze(1) @ (h_X_i[i]).unsqueeze(1).T
          sum2 += h_X_i_h_X_i_T
        inv_matrix = torch.inverse(sum2 + 1e-3 * (torch.eye(sum2.size()[0]).to(device)))
        W = sum1 @ inv_matrix
        wandb.log({"inverse norm": torch.norm(inv_matrix)})
        with torch.no_grad():
          dual_value = torch.einsum('ij,bj->bi', W, h_X_o)
          inner_value = augment_stage2_feature(inner_value)
          grad = inner_solution.compute_hessian_vector_prod(outer_param, Z_inner, X_inner, inner_value, dual_value)
      # CASE : Jacobian multiplication from eq.72
      case "closed_form_DFIV":
        with torch.no_grad():
          # Get the value of phi(Z_inner)
          phi_Z_inner = inner_solution.model(Z_inner)
          # Get the value of psi(X_inner)
          outer_NN_dic = tensor_to_state_dict(inner_solution.outer_model, outer_param, device)
          # Functional call of outer NN with the current outer parameters
          psi_X_inner = torch.func.functional_call(inner_solution.outer_model, parameter_and_buffer_dicts=outer_NN_dic, args=X_inner)
          # Augmenting just adds a column of ones
          feature = augment_stage1_feature(phi_Z_inner)
          # Find V*
          V_star = fit_linear(psi_X_inner, feature, inner_solution.lam_V)
          # Outer objective L_out as a function of V*
          func_Lout_V = lambda x: outer_functional_call_V(x, inner_value, Y_outer, inner_solution.lam_u)
          # Compute the Jacobian of L_out wrt V*
          vector = (jacobian(func_Lout_V, V_star))
          assert(vector.size() == V_star.size())
          # Augmenting just adds a column of ones
          phi_feature = augment_stage1_feature(phi_Z_inner)
          # V* as a function of psi
          func_V_psi = lambda psi_X_inner: fit_linear(psi_X_inner, phi_feature, inner_solution.lam_V)
          # Compute the Jacobian of V* wrt psi
          jac_V_v = ((vjp(func_V_psi, psi_X_inner, vector))[1])
          # Outer feature map psi as a function of theta
          func_psi_theta = lambda outer_param: outer_functional_call(outer_param, inner_solution.outer_model, X_inner)
          # Compute the Jacobian of psi wrt theta
          # Umm not sure why I need the transpose on v here
          grad = ((vjp(func_psi_theta, outer_param, jac_V_v))[1]).T
    return None, grad, None, None


# Some functions needed for some close-form gradient computations in the backward of Inner Solution,
# i.e. the functional calls for Jacobian computation, functions without logging.

def fit_2sls_noprint(treatment_1st_feature, instrumental_1st_feature, instrumental_2nd_feature, outcome_2nd_t, lam1, lam2):
    # stage1
    feature = augment_stage1_feature(instrumental_1st_feature)
    stage1_weight = fit_linear(treatment_1st_feature, feature, lam1)
    # predicting for stage 2
    feature = augment_stage1_feature(instrumental_2nd_feature)
    predicted_treatment_feature = linear_reg_pred(feature, stage1_weight)
    # stage2
    feature = augment_stage2_feature(predicted_treatment_feature)
    stage2_weight = fit_linear(outcome_2nd_t, feature, lam2)
    pred = linear_reg_pred(feature, stage2_weight)
    stage2_loss = torch.norm((outcome_2nd_t - pred)) ** 2 + lam2 * torch.norm(stage2_weight) ** 2
    return dict(stage1_weight=stage1_weight,
                predicted_treatment_feature=predicted_treatment_feature,
                stage2_weight=stage2_weight,
                stage2_loss=stage2_loss)

# Outer objective function as a function of V*
def outer_functional_call_V(stage1_weight, instrumental_2nd_feature, outcome_2nd_t, lam2):
    feature = augment_stage1_feature(instrumental_2nd_feature)
    predicted_treatment_feature = linear_reg_pred(feature, stage1_weight)
    # stage2
    feature = augment_stage2_feature(predicted_treatment_feature)
    stage2_weight = fit_linear(outcome_2nd_t, feature, lam2)
    pred = linear_reg_pred(feature, stage2_weight)
    return torch.norm((outcome_2nd_t - pred)) ** 2 + lam2 * torch.norm(stage2_weight) ** 2

# Features psi as a function of theta
def outer_functional_call(outer_param, outer_model, X):
    outer_model.eval()
    outer_NN_dic = tensor_to_state_dict(outer_model, outer_param, device)
    treatment_feature = (torch.func.functional_call(outer_model, parameter_and_buffer_dicts=outer_NN_dic, args=X))
    #augmented_treatment_feature = augment_stage1_feature(treatment_feature)
    return treatment_feature