from numpy.random import default_rng
from itertools import product
from filelock import FileLock
import pathlib
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
from functorch import make_functional_with_buffers
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.metrics import classification_report
import numpy as np
from pathlib import Path
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import NamedTuple, Optional
from itertools import product
from numpy.random import default_rng
from typing import Tuple, TypeVar

DATA_PATH = pathlib.Path(__file__).resolve().parent

def image_id(latent_bases: np.ndarray, posX_id_arr: np.ndarray, posY_id_arr: np.ndarray,
             orientation_id_arr: np.ndarray,
             scale_id_arr: np.ndarray):
    data_size = posX_id_arr.shape[0]
    color_id_arr = np.array([0] * data_size, dtype=int)
    shape_id_arr = np.array([2] * data_size, dtype=int)
    idx = np.c_[color_id_arr, shape_id_arr, scale_id_arr, orientation_id_arr, posX_id_arr, posY_id_arr]
    return idx.dot(latent_bases)

def structural_func(image, weights):
    return (np.mean((image.dot(weights))**2, axis=1) - 5000) / 1000

def generate_dsprite_data(train_data_size):
    # Get the data path and load the image array.
    with FileLock("./data.lock"):
        dataset_zip = np.load(DATA_PATH.joinpath("dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"),
                              allow_pickle=True, encoding="bytes")
        weights = np.load(DATA_PATH.joinpath("dsprite_mat.npy"))
    
    imgs = dataset_zip['imgs']
    latents_values = dataset_zip['latents_values']
    metadata = dataset_zip['metadata'][()]
    latents_sizes = metadata[b'latents_sizes']
    latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:], np.array([1, ])))

    # Define latent variables (scale, orientation, posX, posY) for test data.
    posX_id_arr, posY_id_arr = [0, 5, 10, 15, 20, 25, 30], [0, 5, 10, 15, 20, 25, 30]
    scale_id_arr, orientation_arr = [0, 3, 5], [0, 10, 20, 30]
    
    # Generate test data
    latent_idx_arr_test = []
    for posX, posY, scale, orientation in product(posX_id_arr, posY_id_arr, scale_id_arr, orientation_arr):
        latent_idx_arr_test.append([0, 2, scale, orientation, posX, posY])
    test_data_size = 7 * 7 * 3 * 4
    latent_idx_arr_test = np.array(latent_idx_arr_test).dot(latents_bases)
    # Use the 64*64=4096 dimensional image as the treatment variable X.
    treatment_test = imgs[latent_idx_arr_test].reshape((test_data_size, 64 * 64))
    # Apply the structural function to generate the outcome Y.
    outcome_test = structural_func(treatment_test, weights)
    # Add an extra dimension to the outcome.
    outcome_test = outcome_test[:, np.newaxis]

    # Generate train data
    rng = default_rng()
    # Uniformly sample the latent variables (scale, orientation, posX, posY) for training data.
    posX_id_arr, posY_id_arr = rng.integers(32, size=train_data_size), rng.integers(32, size=train_data_size)
    scale_id_arr, orientation_arr = rng.integers(6, size=train_data_size), rng.integers(40, size=train_data_size)
    train_image_idx_arr = image_id(latents_bases, posX_id_arr, posY_id_arr, orientation_arr, scale_id_arr)
    # Use the 64*64=4096 dimensional image + noise as the treatment variable X.
    treatment_train = imgs[train_image_idx_arr].reshape((train_data_size, 64 * 64)).astype(np.float64)
    print("treatment_train[34]:", treatment_train[34])
    print("treatment_train[34] sum:", sum(treatment_train[34]))
    print("treatment_train[34] size:", len(treatment_train[34]))
    print("treatment_train[24]:", treatment_train[24])
    print("treatment_train[24] sum:", sum(treatment_train[24]))
    print("treatment_train[24] size:", len(treatment_train[24]))
    treatment_train += rng.normal(0.0, 0.1, treatment_train.shape)
    # Use three latent variables as the instrument variable Z.
    latent_feature = latents_values[train_image_idx_arr]  # (color, shape, scale, orientation, posX, posY)
    instrumental_train = latent_feature[:, 2:5]  # (scale, orientation, posX)
    print("instrumental_train[34]:", instrumental_train[34])
    print("instrumental_train[34] size:", len(instrumental_train[34]))
    # Use the position y as the hidden confounder epsilon.
    #outcome_noise = (posY_id_arr - 16.0) + rng.normal(0.0, 0.5, train_data_size)
    outcome_noise = (latent_feature[:, 5] - 16.0) + rng.normal(0.0, 0.5, train_data_size)
    print("latent_feature[:, 5]:", latent_feature[:, 5])
    print("latent_feature[:, 5] sum:", sum(latent_feature[:, 5]))
    print("outcome_train_noise[34]:", outcome_noise[34])
    print("outcome_train_noise[24]:", outcome_noise[24])
    # Apply the structural function to generate the outcome Y.
    outcome_train = structural_func(treatment_train, weights)
    print("outcome_train[34]:", outcome_train[34])
    print("outcome_train[24]:", outcome_train[24])
    outcome_train =+ outcome_noise
    # Add an extra dimension to the outcome.
    outcome_train = outcome_train[:, np.newaxis]

    # Put all features together
    X_full = np.hstack((instrumental_train, treatment_train))
    X_train, X_val, y_train, y_val = train_test_split(X_full, outcome_train, test_size=0.5, shuffle=True)
    m = X_train.shape[0]
    n = 3 # Shape of instrumental variable Z

    # Print data details
    print("Training feature data shape:", X_train.shape)
    print("Training feature label shape:", y_train.shape)
    print("Training feature data:", X_train[1:5])
    print("Training feature label:", y_train[1:5])
    print("n:", n, "m:", m)

    return n, m, X_train, X_val, y_train, y_val

class DspritesData(Dataset):
  """
  A class for input data.
  """
  def __init__(self, X, y):
    self.X = X
    self.y = y
    self.len = len(self.y)

  def __getitem__(self, index):
    # return Z, X, Y
    return self.X[index][-3:], self.X[index][:-3], self.y[index]

  def __len__(self):
    return self.len

def build_net_for_dsprite():
  instrumental_net = nn.Sequential(spectral_norm(nn.Linear(3, 256)),
                nn.ReLU(),
                spectral_norm(nn.Linear(256, 128)),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                spectral_norm(nn.Linear(128, 128)),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                spectral_norm(nn.Linear(128, 32)),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Linear(32, 1))
  instrumental_net_dual = nn.Sequential(spectral_norm(nn.Linear(3, 256)),
                                        nn.ReLU(),
                                        spectral_norm(nn.Linear(256, 128)),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(128),
                                        spectral_norm(nn.Linear(128, 128)),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(128),
                                        spectral_norm(nn.Linear(128, 32)),
                                        nn.BatchNorm1d(32),
                                        nn.ReLU(),
                                        nn.Linear(32, 1))
  response_net = nn.Sequential(spectral_norm(nn.Linear(64 * 64, 1024)),
                                nn.ReLU(),
                                spectral_norm(nn.Linear(1024, 512)),
                                nn.ReLU(),
                                nn.BatchNorm1d(512),
                                spectral_norm(nn.Linear(512, 128)),
                                nn.ReLU(),
                                spectral_norm(nn.Linear(128, 32)),
                                nn.BatchNorm1d(32),
                                nn.Tanh(),
                                nn.Linear(32, 1))
  return instrumental_net, instrumental_net_dual, response_net