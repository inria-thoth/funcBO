import sys
import os
import importlib
import json
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import NamedTuple, Optional
from itertools import product
from numpy.random import default_rng
from typing import Tuple, TypeVar


def import_module(module_name):
    module, attr = os.path.splitext(module_name)
    try:
        module = importlib.import_module(module)
        return getattr(module, attr[1:])
    except:
        try:
            module = import_module(module)
            return getattr(module, attr[1:])
        except:
            return eval(module+attr[1:])

def config_to_instance(config_module_name="name",**config):
  module_name = config.pop(config_module_name)
  attr = import_module(module_name)
  if config:
    attr = attr(**config)
  return attr

def set_device_and_type(data,device, dtype):
  if isinstance(data,list):
    data = tuple(data)
  if type(data) is tuple:
    data = tuple([ set_device_and_type(d,device,dtype) for d in data ])
    return data
  elif isinstance(data,torch.Tensor): 
    if dtype==torch.double:
      data = data.double()
    else:
      data = data.float()
    return  data.to(device)
  elif isinstance(data,int):
    return torch.tensor(data).to(device)
  else:
    raise NotImplementedError('unknown type')



class RingGenerator:
  def __init__(self, init_generator, device, dtype):
    self.init_generator = init_generator
    self.generator = None
    self.device= device
    self.dtype = dtype
  # def make_generator(self):
  #   return iter(self.init_generator)

  def make_generator(self):
    return (set_device_and_type(data,self.device,self.dtype) for data in self.init_generator)
  def __next__(self, *args):
    try:
      return next(self.generator)
    except:
      self.generator = self.make_generator()
      return next(self.generator)
  def __iter__(self):
    return self.make_generator()

  def __getstate__(self):
    return {'init_generator': self.init_generator,
        'generator': None}
  def __setstate__(self, d ):
    self.init_generator = d['init_generator']
    self.generator = None


def plot_2D_functions(figname, f1, f2, f3, points=None, plot_x_lim=[-5,5], plot_y_lim=[-5,5], plot_nb_contours=10, titles=["True function","Classical Imp. Diff.","Neural Imp. Diff."]):
  """
  A function to plot three continuos 2D functions side by side on the same domain.
  """
  # Create a part of the domain.
  xlist = np.linspace(plot_x_lim[0], plot_x_lim[1], plot_nb_contours)
  ylist = np.linspace(plot_y_lim[0], plot_y_lim[1], plot_nb_contours)
  X, Y = np.meshgrid(xlist, ylist)
  # Get mappings from both the true and the approximated functions.
  Z1, Z2, Z3 = np.zeros_like(X), np.zeros_like(X), np.zeros_like(X)
  for i in range(0, len(X)):
    for j in range(0, len(X)):
      a = np.array([X[i, j], Y[i, j]], dtype='float32')
      Z1[i, j] = f1(((torch.from_numpy(a))).float())
      Z2[i, j] = f2(((torch.from_numpy(a))).float())
      Z3[i, j] = f3(((torch.from_numpy(a))).float())
  # Visualize the true function.
  fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(9, 3))
  ax1.contour(X, Y, Z1, plot_nb_contours, cmap=plt.cm.magma, alpha=0.8, extend='both')
  ax1.set_title(titles[0])
  ax1.set_xlabel("Feature #0")
  ax1.set_ylabel("Feature #1")
  # Visualize the approximated function.
  ax2.contour(X, Y, Z2, plot_nb_contours, cmap=plt.cm.magma, alpha=0.8, extend='both')
  if not (points is None):
    ax2.scatter(points[:,0], points[:,1], marker='.')
  ax2.set_title(titles[1])
  ax2.set_xlabel("Feature #0")
  ax2.set_ylabel("Feature #1")
  ax3.contour(X, Y, Z3, plot_nb_contours, cmap=plt.cm.magma, alpha=0.8, extend='both')
  ax3.set_title(titles[2])
  ax3.set_xlabel("Feature #0")
  ax3.set_ylabel("Feature #1")
  plt.savefig(figname+".png")

def plot_1D_iterations(figname, iters1, iters2, f1, f2, plot_x_lim=[0,1], titles=["Classical Imp. Diff.","Neural Imp. Diff."]):
  """
  A function to plot three continuos 2D functions side by side on the same domain.
  """
  # Create a part of the domain.
  X = np.linspace(plot_x_lim[0], plot_x_lim[1], 100)
  # Visualize the true function.
  fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(7, 3))
  Z1, Z2 = np.zeros_like(X), np.zeros_like(X)
  for i in range(0, len(X)):
    a = (torch.from_numpy(np.array(X[i], dtype='float32')))
    Z1[i] = f1(a).float()
    Z2[i] = f2(a).float()
  Z3, Z4 = np.zeros(len(iters1)), np.zeros(len(iters2))
  X1, X2 = np.zeros(len(iters1)), np.zeros(len(iters2))
  for i in range(0, len(iters1)):
    a = iters1[i]
    X1[i] = a.float()
    Z3[i] = f1(a).float()
  for i in range(0, len(iters2)):
    a = iters2[i]
    X2[i] = a.float()
    Z4[i] = f2(a).float()
  ax1.plot(X, Z1, color='red')
  ax1.scatter(X1, Z3, marker='.')
  ax1.set_title(titles[0])
  ax1.set_xlabel("\mu")
  ax1.set_ylabel("f(\mu)")
  ax2.plot(X, Z2, color='red')
  ax2.scatter(X2, Z4, marker='.')
  ax2.set_title(titles[1])
  ax2.set_xlabel("\mu")
  ax2.set_ylabel("f(\mu)")
  plt.savefig(figname+".png")

def plot_loss(figname, train_loss=None, val_loss=None, test_loss=None, title="Segmentation"):
  """
  Plot the loss value over iterations.
    param figname: name of the figure
    param train_loss: list of train loss values
    param aux_loss: list of auxiliary loss values
    param test_loss: list of test loss values
  """
  plt.clf()
  # Plot and label the training and validation loss values
  if train_loss != None:
    ticks = np.arange(0, len(train_loss), 1)
    plt.yscale('log')
    plt.plot(ticks, train_loss, label='Inner Loss')
  if val_loss != None:
    ticks = np.arange(0, len(val_loss), 1)
    plt.yscale('log')
    plt.plot(ticks, val_loss, label='Outer Loss')
  if test_loss != None:
    ticks = np.arange(0, len(test_loss), 1)
    plt.plot(ticks, test_loss, label='Test Accuracy')
  # Add in a title and axes labels
  plt.xlabel('Iterations')
  #plt.ylabel('')
  plt.title(title)
  # Save the plot
  plt.legend(loc='best')
  plt.savefig(figname+".png")

def sample_X(X, n):
  """
  Take a uniform sample of size n from tensor X.
    param X: data tensor
    param n: sample size
  """
  probas = torch.full([n], 1/n)
  index = (probas.multinomial(num_samples=n, replacement=True)).to(dtype=torch.long)
  return X[index]

def sample_X_y(X, y, n):
  """
  Take a uniform sample of size n from tensor X.
    param X: data tensor
    param y: true value tensor
    param n: sample size
  """
  probas = torch.full([n], 1/n)
  index = (probas.multinomial(num_samples=n, replacement=True)).to(dtype=torch.long)
  return X[index], y[index]

def set_seed(seed=None):
  """
  A function to set the random seed.
    param seed: the random seed, 0 by default
  """
  if seed is None:
    seed = random.randrange(10000)
  print("Seed:", seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  return seed

def get_memory_info():
  """
  Prints cuda's reserved and allocated memory.
  """
  t = torch.cuda.get_device_properties(0).total_memory
  r = torch.cuda.memory_reserved(0)
  a = torch.cuda.memory_allocated(0)
  print("Reserved memory:", r)
  print("Allocated memory:", a)

def cos_dist(grad1, grad2):
  """
  Computes cos simillarity of gradients after flattening of tensors.
  It hasn't been stated in paper if batch normalization is considered as model trainable parameter,
  but from my perspective only convolutional layer's cosine similarities should be measured.
  """
  # perform min(max(-1, dist),1) operation for eventual rounding errors (there's about 1 every epoch)
  cos = nn.CosineSimilarity(dim=0, eps=1e-6)
  res = cos(grad1, grad2)
  return res

def get_accuracy(class_pred, labels):
  """
  Accuracy of a classification task.
  """
  acc = torch.tensor(torch.sum((class_pred) == labels).item() / len(class_pred))
  return acc

class Data(Dataset):
  """
  A class for input data.
  """
  def __init__(self, X, y):
    self.X = X
    self.y = y
    self.len = len(self.y)

  def __getitem__(self, index):
    return self.X[index], self.y[index][0], (self.y[index][1], self.y[index][2], self.y[index][3], self.y[index][4]), index

  def __len__(self):
    return self.len

def auxiliary_toy_data():
  seed=42
  # Setting the random seed.
  set_seed(seed)
  # Initialize dimesnions
  n, m = 2, 2048
  k_shot = 20
  # The coefficient tensor of size (n,1) filled with values uniformally sampled from the range (0,1)
  coef = np.array([[1],[1]]).astype('float32')
  coef_harm = np.array([[-1],[-1]]).astype('float32')
  # The data tensor of size (m,n) filled with values uniformally sampled from the range (0,1)
  X = np.random.uniform(size=(m, n)).astype('float32')
  # True h_star
  h_true = lambda X: X @ coef
  h_harm = lambda X: X @ coef_harm
  # Main labels
  y_main = h_true(X)+np.random.normal(scale=0.2, size=(m,1)).astype('float32')
  # Useful auxiliary labels
  y_aux1 = h_true(X)#+np.random.normal(scale=0.1, size=(m,1)).astype('float32')
  # Useful auxiliary labels
  y_aux2 = h_true(X)#+np.random.normal(scale=0.1, size=(m,1)).astype('float32')
  # Harmful auxiliary labels
  y_aux3 = h_harm(X)#+np.random.normal(scale=0.1, size=(m,1)).astype('float32')
  # Harmful auxiliary labels
  y_aux4 = h_harm(X)#+np.random.normal(scale=0.1, size=(m,1)).astype('float32')
  # Put all labels together
  y = np.hstack((y_main, y_aux1, y_aux2, y_aux3, y_aux4))
  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, shuffle=True, random_state=seed)
  # Set k training labels to -1
  indices = np.random.choice(np.arange(y_train.shape[0]), replace=False, size=k_shot)
  y_train[indices] = -1
  # Convert everything to PyTorch tensors
  X_train, X_val, y_train, y_val, coef = (torch.from_numpy(X_train)), (torch.from_numpy(X_val)), (torch.from_numpy(y_train)), (torch.from_numpy(y_val)), (torch.from_numpy(coef))
  print("X shape:", X.shape)
  print("y shape:", y.shape)
  print("True coeficients:", coef)
  print("X training data:", X_train[1:5])
  print("y training labels:", y_train[1:5])
  print()
  return n, m, X_train, X_val, y_train, y_val, coef

def tensor_to_state_dict(model, params, device):
  """
  A function to wrap a tensor as parameter dictionary of a neural network.
    param model: neural network
    param params: a tensor to be wrapped
  """
  start = 0
  current_dict = model.state_dict()
  for name, param in model.named_parameters():
      dim = torch.tensor(param.size())
      length = torch.prod(dim).item()
      end = start + length
      new_weights = params[start:end].view_as(param).to(device)
      current_dict[name] = new_weights
      start = end
  return current_dict

def state_dict_to_tensor(model, device):
  """
  A function to wrap a tensor as parameter dictionary of a neural network.
    param model: neural network
  """
  start = 0
  current_tensor_list = []
  current_dict = model.state_dict()
  for name, param in model.named_parameters():
    t = param.clone().detach().flatten()
    current_tensor_list.append(t)
  params = (torch.cat(current_tensor_list, 0)).to(device)#put to cuda
  return params

def assign_device(device):
    """
    Assigns a device for PyTorch based on the provided device identifier.

    Parameters:
    - device (int): Device identifier. If positive, it represents the GPU device
                   index; if -1, it sets the device to 'cuda'; if -2, it sets
                   the device to 'cpu'.

    Returns:
    - device (str): The assigned device, represented as a string. 
                    'cuda:X' if device > -1 and CUDA is available, where X is 
                    the provided device index. 'cuda' if device is -1.
                    'cpu' if device is -2.
    """
    if device >-1:
        device = (
            'cuda:'+str(device) 
            if torch.cuda.is_available() and device>-1 
            else 'cpu'
        )
    elif device==-1:
        device = 'cuda'
    elif device==-2:
        device = 'cpu'
    return device

def get_dtype(dtype):
    """
    Returns the PyTorch data type based on the provided integer identifier.

    Parameters:
    - dtype (int): Integer identifier representing the desired data type.
                   64 corresponds to torch.double, and 32 corresponds to torch.float.

    Returns:
    - torch.dtype: PyTorch data type corresponding to the provided identifier.

    Raises:
    - NotImplementedError: If the provided identifier is not recognized (not 64 or 32).
    """
    if dtype==64:
        return torch.double
    elif dtype==32:
        return torch.float
    else:
        raise NotImplementedError('Unkown type')
