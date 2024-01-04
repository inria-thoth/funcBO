import numpy as np
import pathlib
import random
import sys
import torch
import torch.nn as nn
from pathlib import Path
from filelock import FileLock
from itertools import product
from torch.utils.data import Dataset
from numpy.random import default_rng
from torch.nn.utils import spectral_norm

DATA_PATH = pathlib.Path(__file__).resolve().parent

def generate_dsprites_image(scale, orientation, posX, posY, imgs, latents_sizes, latents_bases):
    """
    Generate an image from the dSprites dataset based on given parameters.

    Args:
    - posX (int): X position index (0 to 31).
    - posY (int): Y position index (0 to 31).
    - scale (int): Scale index (0 to 5).
    - orientation (int): Orientation index (0 to 39).

    Returns:
    - image (numpy.ndarray): The generated image as a numpy array.
    """

    # Convert input parameters to a latent vector
    latent_vector = [0, 2, scale, orientation, posX, posY]
    latent_idx = np.array(latent_vector).dot(latents_bases)

    # Get the image corresponding to the latent vector
    image = imgs[int(latent_idx)].squeeze()  # Squeeze to remove single dimensions

    return image.reshape((64 * 64, 1)).astype(np.float64)

def image_id(latent_bases: np.ndarray, posX_id_arr: np.ndarray, posY_id_arr: np.ndarray,
             orientation_id_arr: np.ndarray,
             scale_id_arr: np.ndarray):
    data_size = posX_id_arr.shape[0]
    color_id_arr = np.array([0] * data_size, dtype=int)
    shape_id_arr = np.array([2] * data_size, dtype=int)
    idx = np.c_[color_id_arr, shape_id_arr, scale_id_arr, orientation_id_arr, posX_id_arr, posY_id_arr]
    return idx.dot(latent_bases)

def structural_func(image, B):
    return (np.power((np.linalg.norm(B @ image)), 2) - 5000) / 1000

def generate_dsprite_data(train_size, val_size, seed=42):
    # Random seed
    rng = default_rng(seed=seed)
    # Get the data path and load the image array.
    with FileLock("./data.lock"):
        dataset_zip = np.load(DATA_PATH.joinpath("dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"),
                              allow_pickle=True, encoding="bytes")
        weights = (np.load(DATA_PATH.joinpath("dsprite_mat.npy")).T) * 0.01
    
    # Extract relevant data from the dataset, dataset_zip (npz file): the loaded dSprites dataset.   
    imgs = dataset_zip['imgs']
    metadata = dataset_zip['metadata'][()]
    latents_sizes = metadata[b'latents_sizes']
    latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:], np.array([1, ])))

    # Define latent variables for test data (scale, orientation, posX, posY) for test data.
    posX_id_arr, posY_id_arr = [0, 5, 10, 15, 20, 25, 30], [0, 5, 10, 15, 20, 25, 30]
    scale_id_arr, orientation_arr = [0, 3, 5], [0, 10, 20, 30]
    
    # Generate test data
    treatment_test, outcome_test = [], []
    for posX, posY, scale, orientation in product(posX_id_arr, posY_id_arr, scale_id_arr, orientation_arr):
        # Use the 64*64 dimensional images as the treatment variable X.
        treatment = (generate_dsprites_image(scale, orientation, posX, posY, imgs, latents_sizes, latents_bases))
        treatment_test.append(treatment.flatten())
        # Apply the structural function to generate the outcome Y.
        outcome_test.append(structural_func(treatment, weights).flatten())
    treatment_test = torch.from_numpy(np.vstack(treatment_test))
    outcome_test = torch.from_numpy(np.vstack(outcome_test))

    # Sample latent variables for train data (scale, orientation, posX, posY) for test data.
    posX_id_arr, posY_id_arr = rng.integers(32, size=train_size*2), rng.integers(32, size=train_size*2)
    scale_id_arr, orientation_arr = rng.integers(6, size=train_size), rng.integers(40, size=train_size)
    
    # Generate train data
    instrumental_train, treatment_train, outcome_train, structural_train = [], [], [], []
    for posX, posY, scale, orientation in product(posX_id_arr, posY_id_arr, scale_id_arr, orientation_arr):
        # Use posX, scale and orientation as the instrumental variable Z.
        instrumental_train.append(np.array([0.5 + (scale / 5) * 0.5, (orientation / 39) * (2 * np.pi), posX / 31]).flatten())
        # Use the 64*64 dimensional images as the treatment variable X.
        treatment_noiseless = generate_dsprites_image(scale, orientation, posX, posY, imgs, latents_sizes, latents_bases)
        treatment = (treatment_noiseless + rng.normal(0.0, 0.01, (64*64, 1)))
        treatment_train.append (treatment.flatten())
        # Apply the structural function to generate the outcome Y, here position Y = posY / 31, since posY is the index.
        outcome_noise = 32 * ((posY / 31) - 0.5) + rng.normal(0.0, 0.5)
        structural = structural_func(treatment_noiseless, weights)
        outcome_train.append((structural + outcome_noise).flatten())
        structural_train.append(structural.flatten())
    instrumental_train = torch.from_numpy(np.vstack(instrumental_train))
    treatment_train = torch.from_numpy(np.vstack(treatment_train))
    outcome_train = torch.from_numpy(np.vstack(outcome_train))
    structural_train = torch.from_numpy(np.vstack(structural_train))

    # Sample latent variables for validation data (scale, orientation, posX, posY) for test data.
    posX_id_arr, posY_id_arr = rng.integers(32, size=val_size*2), rng.integers(32, size=val_size*2)
    scale_id_arr, orientation_arr = rng.integers(6, size=val_size), rng.integers(40, size=val_size)

    # Generate validation data
    instrumental_val, treatment_val, outcome_val, structural_val = [], [], [], []
    for posX, posY, scale, orientation in product(posX_id_arr, posY_id_arr, scale_id_arr, orientation_arr):
        # Use posX, scale and orientation as the instrumental variable Z.
        instrumental_val.append(np.array([0.5 + (scale / 5) * 0.5, (orientation / 39) * (2 * np.pi), posX / 31]).flatten())
        # Use the 64*64 dimensional images as the treatment variable X.
        treatment_noiseless = generate_dsprites_image(scale, orientation, posX, posY, imgs, latents_sizes, latents_bases)
        treatment = ((treatment_noiseless + rng.normal(0.0, 0.01, (64*64, 1))))
        treatment_val.append (treatment.flatten())
        # Apply the structural function to generate the outcome Y, here position Y = posY / 31, since posY is the index.
        outcome_noise = 32 * ((posY / 31) - 0.5) + rng.normal(0.0, 0.5)
        structural = structural_func(treatment_noiseless, weights)
        outcome_val.append((structural + outcome_noise).flatten())
        structural_val.append(structural.flatten())
    instrumental_val = torch.from_numpy(np.vstack(instrumental_val))
    treatment_val = torch.from_numpy(np.vstack(treatment_val))
    outcome_val = torch.from_numpy(np.vstack(outcome_val))
    structural_val = torch.from_numpy(np.vstack(structural_val))

    print("Test data type:", (treatment_test.type()), (outcome_test.type()))
    print("Test data shape:", (treatment_test.size()), (outcome_test.size()))
    print("Train data shape:", (instrumental_train.size()), (treatment_train.size()), (outcome_train.size()))
    print("Validation data shape:", (instrumental_val.size()), (treatment_val.size()), (outcome_val.size()))
    
    return instrumental_train, treatment_train, outcome_train, instrumental_val, treatment_val, outcome_val, treatment_test, outcome_test

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

def build_net_for_dsprite(outer_u_dim, seed):
  torch.manual_seed(seed)
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
                nn.ReLU())
  torch.manual_seed(seed)
  response_net = nn.Sequential(spectral_norm(nn.Linear(64 * 64, 1024)),
                                nn.ReLU(),
                                spectral_norm(nn.Linear(1024, 512)),
                                nn.ReLU(),
                                nn.BatchNorm1d(512),
                                spectral_norm(nn.Linear(512, 128)),
                                nn.ReLU(),
                                spectral_norm(nn.Linear(128, 32)),
                                nn.BatchNorm1d(32),
                                nn.Tanh())
  return instrumental_net, None, response_net