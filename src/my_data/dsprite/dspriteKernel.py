import sys
import torch
import random
import pathlib
import numpy as np
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from filelock import FileLock
from itertools import product
from torch.utils.data import Dataset
from numpy.random import default_rng
from typing import NamedTuple, Optional, Tuple
from sklearn.model_selection import train_test_split


class TrainDataSet(NamedTuple):
    treatment: np.ndarray
    instrumental: np.ndarray
    covariate: Optional[np.ndarray]
    outcome: np.ndarray
    structural: np.ndarray

class TestDataSet(NamedTuple):
    treatment: np.ndarray
    covariate: Optional[np.ndarray]
    structural: np.ndarray

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


def generate_test_dsprite(device):
    with FileLock("./data.lock"):
        dataset_zip = np.load(DATA_PATH.joinpath("dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"),
                              allow_pickle=True, encoding="bytes")
        weights = np.load(DATA_PATH.joinpath("dsprite_mat.npy"))

    imgs = dataset_zip['imgs']
    latents_values = dataset_zip['latents_values']
    metadata = dataset_zip['metadata'][()]

    latents_sizes = metadata[b'latents_sizes']
    latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:], np.array([1, ])))

    posX_id_arr = [0, 5, 10, 15, 20, 25, 30]
    posY_id_arr = [0, 5, 10, 15, 20, 25, 30]
    scale_id_arr = [0, 3, 5]
    orientation_arr = [0, 10, 20, 30]
    latent_idx_arr = []
    for posX, posY, scale, orientation in product(posX_id_arr, posY_id_arr, scale_id_arr, orientation_arr):
        latent_idx_arr.append([0, 2, scale, orientation, posX, posY])

    image_idx_arr = np.array(latent_idx_arr).dot(latents_bases)
    data_size = 7 * 7 * 3 * 4
    treatment = imgs[image_idx_arr].reshape((data_size, 64 * 64))
    structural = structural_func(treatment, weights)
    structural = structural[:, np.newaxis]
    return TestDataSet(treatment=treatment, covariate=None, structural=structural)


def generate_train_dsprite(data_size, rand_seed, device, val_size=100):
    with FileLock("./data.lock"):
        dataset_zip = np.load(DATA_PATH.joinpath("dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"),
                              allow_pickle=True, encoding="bytes")
        weights = np.load(DATA_PATH.joinpath("dsprite_mat.npy"))

    imgs = dataset_zip['imgs']
    latents_values = dataset_zip['latents_values']
    metadata = dataset_zip['metadata'][()]

    latents_sizes = metadata[b'latents_sizes']
    latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:], np.array([1, ])))

    rng = default_rng(seed=rand_seed)
    posX_id_arr = rng.integers(32, size=data_size)
    posY_id_arr = rng.integers(32, size=data_size)
    scale_id_arr = rng.integers(6, size=data_size)
    orientation_arr = rng.integers(40, size=data_size)
    image_idx_arr = image_id(latents_bases, posX_id_arr, posY_id_arr, orientation_arr, scale_id_arr)
    treatment = imgs[image_idx_arr].reshape((data_size, 64 * 64)).astype(np.float64)
    treatment += rng.normal(0.0, 0.1, treatment.shape)
    latent_feature = latents_values[image_idx_arr]  # (color, shape, scale, orientation, posX, posY)
    instrumental = latent_feature[:, 2:5]  # (scale, orientation, posX)
    outcome_noise = (posY_id_arr - 16.0) + rng.normal(0.0, 0.5, data_size)
    structural = structural_func(treatment, weights)
    outcome = structural + outcome_noise
    structural = structural[:, np.newaxis]
    outcome = outcome[:, np.newaxis]
    if val_size == 0:
        train_data_final = TrainDataSet(treatment=treatment,
                            instrumental=instrumental,
                            covariate=None,
                            structural=structural,
                            outcome=outcome)
        validation_data_final = None
    else:
        train_data_final = TrainDataSet(treatment=treatment[:-val_size, :],
                            instrumental=instrumental[:-val_size, :],
                            covariate=None,
                            structural=structural[:-val_size, :],
                            outcome=outcome[:-val_size, :])
        validation_data_final = TrainDataSet(treatment=treatment[-val_size:, :],
                            instrumental=instrumental[-val_size:, :],
                            covariate=None,
                            structural=structural[-val_size:, :],
                            outcome=outcome[-val_size:, :])
    return train_data_final, validation_data_final

def split_train_data(train_data, split_ratio):
    n_data = train_data[0].shape[0]
    idx_train_1st, idx_train_2nd = train_test_split(np.arange(n_data), train_size=split_ratio)

    def get_data(data, idx):
        return data[idx] if data is not None else None

    train_1st_data = TrainDataSet(*[get_data(data, idx_train_1st) for data in train_data])
    train_2nd_data = TrainDataSet(*[get_data(data, idx_train_2nd) for data in train_data])

    return train_1st_data, train_2nd_data

class DspritesData(Dataset):
  """
  A class for input data.
  """
  def __init__(self, instrumental, treatment, outcome):
    self.instrumental = instrumental
    self.treatment = treatment
    self.outcome = outcome
    self.len = len(self.outcome)

  def __getitem__(self, index):
    # return Z, X, Y
    return self.instrumental[index], self.treatment[index], self.outcome[index]

  def __len__(self):
    return self.len

class DspritesTestData(Dataset):
  """
  A class for input data.
  """
  def __init__(self, treatment, outcome):
    self.treatment = treatment
    self.outcome = outcome
    self.len = len(self.outcome)

  def __getitem__(self, index):
    # return X, Y
    return self.treatment[index], self.outcome[index]

  def __len__(self):
    return self.len
        
"""
def rbf_kernel(X, Y, gamma):
    X = X.unsqueeze(1)
    Y = Y.unsqueeze(0)
    diff = X - Y
    squared_distance = torch.sum(diff ** 2, dim=-1)
    return torch.exp(-gamma * squared_distance)
"""

def rbf_kernel(X, Y, gamma):
    # Ensure X and Y have the same dtype
    X = X.to(Y.dtype)
    # Unsqueezing to match dimensions
    X = X.unsqueeze(1)
    Y = Y.unsqueeze(0)
    # Calculate the RBF kernel with loops
    n, m, d = X.shape[0], Y.shape[1], X.shape[2]
    squared_distance = torch.empty((n, m), dtype=torch.float64, device=X.device)
    for i in range(n):
        for j in range(m):
            diff = X[i] - Y[:, j]
            squared_distance[i, j] = -gamma * torch.sum(diff ** 2)
    # Exponentiate in-place
    squared_distance.exp_()
    return squared_distance

class KernelRidge(nn.Module):
    def __init__(self, alpha=1.0, kernel='linear', gamma=None, dual_coef_=None):
        super(KernelRidge, self).__init__()
        self.alpha = alpha
        self.kernel = kernel
        self.gamma = gamma
        self.dual_coef_ = dual_coef_

    def fit(self, X, y):
        if self.kernel == 'precomputed':
            self.K = X  # Assuming X is the precomputed kernel matrix
        else:
            # Compute the kernel matrix based on the chosen kernel function
            if self.kernel == 'linear':
                self.K = torch.mm(X, X.t())
            elif self.kernel == 'rbf':
                self.K = rbf_kernel(X, X, self.gamma)
            else:
                raise ValueError("Unsupported kernel")

        # Solve for dual coefficients using the kernel matrix
        I = torch.eye(len(X))
        self.dual_coef_ = torch.linalg.solve((self.K + self.alpha * I), y)

    def predict(self, X):
        if self.dual_coef_ is None:
            raise Exception("Model has not been fitted yet.")
        
        if self.kernel == 'precomputed':
            K_test = X  # Assuming X is the precomputed kernel matrix
        else:
            # Compute kernel matrix for prediction data
            if self.kernel == 'linear':
                K_test = torch.mm(X, X.t())
            elif self.kernel == 'rbf':
                K_test = rbf_kernel(X, X, self.gamma)
            else:
                raise ValueError("Unsupported kernel")

        # Perform predictions
        y_pred = torch.mm(K_test, self.dual_coef_)

        return y_pred