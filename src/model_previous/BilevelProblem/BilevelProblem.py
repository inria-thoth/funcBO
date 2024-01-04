import sys
import math
import time
import torch
import wandb
import numpy as np
from torch.autograd import grad
from torch.autograd.functional import hessian

# Add main project directory path
sys.path.append('/home/clear/ipetruli/projects/bilevel-optimization/src')

from model_previous.FunctionApproximator.FunctionApproximator import FunctionApproximator
from my_data.dsprite.dspriteKernel import *
from model.utils import *
# Loss helper functions
MSE = nn.MSELoss()

class BilevelProblem:
  """
  Instanciates the bilevel problem and solves it using one of the methods:
  1) Classical Implicit Differentiation
  2) Neural Implicit Differentiation.
  """

  def __init__(self, outer_objective, inner_objective, method, data, find_theta_star=None, batch_size=64, reg_param=None, gamma=1.):
    """
    Init method.
      param outer_objective: outer level objective function
      param inner_objective: inner level objective function
      param method: method used to solve the bilevel problem
      param data: input data and labels for outer and inner objectives
      param find_theta_star: method to find the optimal theta* for classical imp. diff.
      param batch_size: batch size for approximating the function h*
    """
    self.outer_objective = outer_objective
    self.inner_objective = inner_objective
    self.method = method
    self.Z_inner, self.X_inner, self.Y_inner, self.Z_outer, self.X_outer, self.Y_outer, self.X_test, self.Y_test = data
    if self.method=="neural_implicit_diff":
      self.dim_x = self.X_inner.size()[1]
      dim_y = 1
      layer_sizes = [self.dim_x, 10, 20, 10, dim_y]
      # Neural network to approximate the function h*
      self.NN_h = FunctionApproximator(layer_sizes, loss_G=inner_objective, function='h')
      self.NN_h.load_data(self.X_inner, self.y_inner)
      # Neural network to approximate the function a*
      self.NN_a = FunctionApproximator(layer_sizes, function='a')
      self.NN_a.load_data(self.X_inner, self.y_inner, self.X_outer, self.y_outer)
    elif self.method=="kernel_implicit_diff":
      # Parameters for the RBF kernel
      self.gamma = gamma
      self.lam1, self.lam2 = reg_param
      # Compute the RBF kernel matrix using vectorized operations
      self.K_test_X = rbf_kernel(self.X_test, self.X_inner, self.gamma)
      self.K_inner_X = rbf_kernel(self.X_inner, self.X_inner, self.gamma)
      self.K_inner_Z = rbf_kernel(self.Z_inner, self.Z_inner, self.gamma)
      self.K_outer_inner_Z = rbf_kernel(self.Z_outer, self.Z_inner, self.gamma)
      self.Z_inner_outer = torch.cat((self.Z_inner, self.Z_outer), dim=0)
      self.K_all_Z = rbf_kernel(self.Z_inner_outer, self.Z_inner_outer, self.gamma)
      self.K_outer_all_Z = rbf_kernel(self.Z_outer, self.Z_inner_outer, self.gamma)
      self.K_inner_all_Z = rbf_kernel(self.Z_inner, self.Z_inner_outer, self.gamma)
      # Set functions h(x) and a(x)
      self.beta = None
      self.phi = None
      self.h_star = lambda x : rbf_kernel(x, self.Z_inner, self.gamma) @ self.beta
      self.a_star = lambda x : rbf_kernel(x, self.Z_inner_outer, self.gamma) @ self.phi
      # Compute optimal outer parameter
      self.eye_Z_inner = torch.eye(len(self.K_inner_Z), dtype=torch.float64)
      #inv_K_Z = torch.linalg.inv(self.K_inner_Z + self.eye_Z_inner * self.lam2)
      #assert torch.isclose(inv_K_Z @ (self.K_inner_Z + self.eye_Z_inner * self.lam2), torch.eye(len(self.K_inner_Z), dtype=torch.float64)).all()
      #A = self.K_inner_X @ inv_K_Z @ self.K_outer_inner_Z.T
      #self.opt_mu = self.least_squares_inverse((A @ self.K_outer_inner_Z @ inv_K_Z @ self.K_inner_X + self.lam1 * self.K_inner_X), (A @ self.Y_outer))
      
    elif self.method=="implicit_diff":
      self.find_theta_star = find_theta_star
    self.batch_size = batch_size
    self.__input_check__()

  def __input_check__(self):
    """
    Ensure that the inputs are of the right form and all necessary inputs have been supplied.
    """
    if (self.outer_objective is None) or (self.inner_objective is None):
      raise AttributeError("You must specify the inner and outer objectives")
    if (self.method == "implicit_diff")  and (self.find_theta_star is None):
      raise AttributeError("You must specify the closed form solution of the inner problem for class. imp. diff.")
    if not (self.method == "implicit_diff" or self.method == "neural_implicit_diff" or self.method == "kernel_implicit_diff"):
      raise ValueError("Invalid method for solving the bilevel problem")

  def train(self, mu0, maxiter=100, step=0.1):
    """
    Find the optimal solution.
      param maxiter: maximum number of iterations
      param step: stepsize for gradient descent on the outer variable
    """
    mu_new = mu0#self.opt_mu
    n_iters, iters, converged, inner_loss, outer_loss, times = 0, [mu_new], False, [], [], []
    while n_iters < maxiter and not converged:
      mu_old = mu_new.clone()
      start = time.time()
      mu_new, h_star = self.find_mu_new(mu_old, step)
      if self.method == "implicit_diff":
        inner_loss.append(self.inner_objective(mu_new, h_star, self.X_inner, self.y_inner))
        outer_loss.append(self.outer_objective(mu_new, h_star, self.X_outer, self.y_outer))
      elif self.method == "neural_implicit_diff":
        inner_loss.append(self.inner_objective(mu_new, h_star(self.X_inner), self.y_inner))
        outer_loss.append(self.outer_objective(mu_new, h_star(self.X_outer), self.y_outer))
      elif self.method == "kernel_implicit_diff":
        inner_loss.append((self.inner_objective(mu_new, h_star(self.Z_inner), self.K_inner_X)).item())
        outer_loss.append((self.outer_objective(mu_new, h_star(self.Z_outer), self.Y_outer)).item())
        wandb.log({"outer var. norm": torch.norm(mu_new).item()})
        wandb.log({"K norm": torch.norm(self.K_test_X).item()})
        wandb.log({"K @ mu norm": torch.norm((self.K_test_X @ mu_new)).item()})
        wandb.log({"y norm": torch.norm(self.Y_test).item()})
        wandb.log({"test loss": MSE((self.K_test_X @ mu_new), self.Y_test).item()})
      times.append(time.time() - start)
      #wandb.log({"out. param. norm": torch.norm(mu_new)})
      #wandb.log({"inn. param. norm": torch.norm(self.beta)})
      #converged = self.check_convergence(mu_old, mu_new)
      iters.append(mu_new)
      n_iters += 1
    return mu_new, iters, n_iters, times, inner_loss, outer_loss, h_star

  def find_mu_new(self, mu_old, step):
    """
    Find the next value in gradient descent.
      param mu_old: old value of the outer variable
      param step: stepsize for gradient descent on the outer variable
    """
    if self.method=="implicit_diff":
      X_in, y_in = self.X_inner, self.y_inner
      X_out, y_out = self.X_outer, self.y_outer
      # 1) Find a parameter vector h* the argmin of the inner objective G(mu,h)
      theta_star = self.find_theta_star(self.X_inner, self.y_inner, mu_old)
      # 2) Get Jh* the Jacobian of the inner objective wrt h
      Jac = -1*torch.linalg.solve((self.inner_grad22(mu_old, theta_star, X_in, y_in)), (self.inner_grad12(mu_old, theta_star, X_in, y_in)))
      # 3) Compute grad L(mu): the gradient of L(mu) wrt mu
      grad = self.outer_grad1(mu_old, theta_star, X_out, y_out) + Jac.T @ self.outer_grad2(mu_old, theta_star, X_out, y_out)
      self.theta_star = theta_star
      h_star = theta_star
    elif self.method=="neural_implicit_diff":
      # 1) Find a function that approximates h*(x)
      h_star_cuda, loss_values = self.NN_h.train(mu_k=mu_old, num_epochs = 10)
      # Here autograd and manual grad already differ? Not rlly
      h_star = self.get_h_star()
      # 2) Find a function that approximates a*(x)
      a_star_cuda, loss_values = self.NN_a.train(self.inner_grad22, self.outer_grad2, mu_k=mu_old, h_k=h_star_cuda, num_epochs = 10)
      a_star = self.get_a_star()
      # 3) Compute grad L(mu): the gradient of L(mu) wrt mu
      X_out, y_out = sample_X_y(self.X_outer, self.y_outer, self.batch_size)
      X_in, y_in = sample_X_y(self.X_inner, self.y_inner, self.batch_size)
      B = self.inner_grad12(mu_old, h_star, X_in, y_in)
      grad = self.outer_grad1(mu_old, h_star, X_out, y_out) + B.T @ (a_star(X_in))
      self.h_star = h_star
      self.a_star = a_star
    elif self.method=="kernel_implicit_diff":
      # 1) Find a function in RKHS that approximates h*(x)
      beta = self.least_squares_inverse(self.K_inner_Z + self.lam2 * self.eye_Z_inner, self.K_inner_X @ mu_old)
      self.beta = beta
      h_star = self.h_star
      # 2) Find a function that approximates a*(x)
      A = (self.K_inner_all_Z.T @ self.K_inner_all_Z + self.lam2 * self.K_all_Z)
      b = self.K_outer_all_Z.T @ (self.K_outer_inner_Z @ beta - self.Y_outer)
      self.phi = (-1)*self.least_squares_inverse(A, b)
      # 3) Compute grad L(mu): the gradient of L(mu) wrt mu
      term1 = self.outer_grad1(mu_old, self.h_star, self.Z_outer, self.Y_outer)
      term2 = (self.inner_grad12(mu_old, self.h_star, self.Z_inner, self.K_inner_X))
      term3 = (self.a_star(self.Z_inner))
      #grad = self.K_inner_X @ self.least_squares_inverse(self.K_inner_Z + self.eye_Z_inner * self.lam2, (self.K_outer_inner_Z.T @ (self.K_outer_inner_Z @ beta - self.Y_outer))) + self.lam1 * self.K_inner_X @ mu_old
      grad = term1 + term2 @ term3
    else:
      raise ValueError("Unkown method for solving a bilevel problem")
    # 4) Compute the next iterate x_{k+1} = x_k - grad L(x)
    mu_new = mu_old-step*grad
    # Remove the associated autograd
    mu_new = mu_new.detach()
    return mu_new, h_star

  def get_h_star(self, h_theta=None):
    """
    Return the function h*.
      param h_theta: a method that returns a function parametrized by theta* used in class. imp. diff.
    """
    if self.method == "neural_implicit_diff":
      h = lambda x : (self.NN_h.NN.forward(x)).to(torch.device("cpu"))
    elif self.method == "implicit_diff":
      h = h_theta(self.theta_star)
    return h

  def get_a_star(self, mu=0, outer_grad2_h=None, h_theta=None, h_theta_grad=None):
    """
    Return the function a* for neural imp. diff. and an equivalent for classical imp. diff.
    """
    if self.method == "neural_implicit_diff":
      a = lambda x : (self.NN_a.NN.forward(x)).to(torch.device("cpu"))
    elif self.method == "implicit_diff":
      a = lambda X, y: h_theta_grad(X) @ torch.linalg.solve(self.inner_grad22(mu, self.theta_star, X, y), h_theta_grad(X).T) @ outer_grad2_h(mu, h_theta, X, y)
    return a

  def check_convergence(self, mu_old, mu_new):
    """
    Checks convergence of the algorithm based on last iterates.
    """
    return torch.norm(mu_old-mu_new)<1e-4

  def outer_grad1(self, mu, h, X_out, y_out):
    """
    Returns the gradient of the outer objective wrt to the first argument mu.
    """
    mu.requires_grad = True
    mu.grad = None
    if self.method=="implicit_diff":
      theta = h
      theta.requires_grad = True
      theta.grad = None
      loss = self.outer_objective(mu, h, X_out, y_out)
    else:
      value = h(X_out)
      value.detach()
      loss = self.outer_objective(mu, value, y_out)
      loss.backward()
      f = lambda mu: self.outer_objective(mu, value, y_out)
      #assert torch.autograd.gradcheck(f, (mu), rtol=0.01, raise_exception=True)
    return mu.grad

  def outer_grad2(self, mu, h, X_out, y_out):
    """
    Returns the gradient of the outer objective wrt to the second argument h/theta.
    """
    mu.detach()
    if self.method=="implicit_diff":
      theta = h
      theta.requires_grad = True
      theta.grad = None
      loss = self.outer_objective(mu, theta, X_out, y_out)
      loss.backward()
      gradient = theta.grad
    else:
      value = h(X_out)
      value.requires_grad = True
      value.grad = None
      loss = self.outer_objective(mu, value, y_out)
      loss.backward()
      f = lambda value: self.outer_objective(mu, value, y_out)
      torch.autograd.gradcheck(f, (value), rtol=0.01, raise_exception=True)
      gradient = value.grad
    return gradient

  def inner_grad22(self, mu, h, X_in, y_in):
    """
    Returns the hessian of the inner objective wrt to the second argument h/theta.
    """
    mu.detach()
    if self.method=="implicit_diff":
      theta = h
      theta.requires_grad = True
      theta.grad = None
      f = lambda arg1, arg2: self.inner_objective(arg1, arg2, X_in, y_in)
      hess = hessian(f, (mu, theta))[1][1]
    else:
      value = h(X_in)
      value.requires_grad = True
      value.grad = None
      y_in = y_in
      f = lambda arg1, arg2: self.inner_objective(arg1, arg2, y_in)
      hess = hessian(f, (mu, value))[1][1]
      torch.autograd.gradcheck(f, (mu, value), rtol=0.01, raise_exception=True)
    return hess.squeeze()

  def inner_grad12(self, mu, h, X_in, y_in):
    """
    Returns part of the hessian of the inner objective.
    """
    mu.requires_grad = True
    mu.grad = None
    if self.method=="implicit_diff":
      theta = h
      theta.requires_grad = True
      theta.grad = None
      f = lambda arg1, arg2: self.inner_objective(arg1, arg2, X_in, y_in)
      hess = (hessian(f, (mu, theta))[0][1])
    else:
      value = h(X_in)
      value.requires_grad = True
      value.grad = None
      f = lambda arg1, arg2: self.inner_objective(arg1, arg2, y_in)
      hess = (hessian(f, (mu, value))[0][1])
      #assert torch.autograd.gradcheck(f, (mu, value), rtol=0.01, raise_exception=True)
    return hess.squeeze()
  
  def least_squares_inverse(self, A, b):
    inverse = torch.linalg.lstsq(A, b)[0]
    if not (torch.isclose(A @ inverse, b).all()):
      print("norm of A @ inverse - b:", torch.norm(A @ inverse - b))
    return inverse