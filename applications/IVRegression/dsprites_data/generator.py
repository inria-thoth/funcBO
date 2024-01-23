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
from typing import NamedTuple, Optional, Tuple
from sklearn.model_selection import train_test_split
from funcBO.utils import set_seed

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

class TrainDataSetTorch(NamedTuple):
    treatment: torch.Tensor
    instrumental: torch.Tensor
    covariate: torch.Tensor
    outcome: torch.Tensor
    structural: torch.Tensor

    @classmethod
    def from_numpy(cls, train_data: TrainDataSet, device='cpu', dtype=torch.float32):
        covariate = None
        if train_data.covariate is not None:
            covariate = torch.tensor(train_data.covariate, dtype=dtype).to(device)
        return TrainDataSetTorch(treatment=torch.tensor(train_data.treatment, dtype=dtype).to(device),
                                 instrumental=torch.tensor(train_data.instrumental, dtype=dtype).to(device),
                                 covariate=covariate,
                                 outcome=torch.tensor(train_data.outcome, dtype=dtype).to(device),
                                 structural=torch.tensor(train_data.structural, dtype=dtype).to(device))

    def to_gpu(self):
        covariate = None
        if self.covariate is not None:
            covariate = self.covariate.cuda()
        return TrainDataSetTorch(treatment=self.treatment.cuda(),
                                 instrumental=self.instrumental.cuda(),
                                 covariate=covariate,
                                 outcome=self.outcome.cuda(),
                                 structural=self.structural.cuda())


class TestDataSetTorch(NamedTuple):
    treatment: torch.Tensor
    covariate: torch.Tensor
    structural: torch.Tensor

    @classmethod
    def from_numpy(cls, test_data: TestDataSet, device='cpu', dtype=torch.float32):
        covariate = None
        if test_data.covariate is not None:
            covariate = torch.tensor(test_data.covariate, dtype=dtype).to(device)
        return TestDataSetTorch(treatment=torch.tensor(test_data.treatment, dtype=dtype).to(device),
                                covariate=covariate,
                                structural=torch.tensor(test_data.structural, dtype=dtype).to(device))
    def to_gpu(self):
        covariate = None
        if self.covariate is not None:
            covariate = self.covariate.cuda()
        return TestDataSetTorch(treatment=self.treatment.cuda(),
                                covariate=covariate,
                                structural=self.structural.cuda())

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

def generate_train_dsprite(data_size, rand_seed, val_size=0):
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


def split_train_data(train_data, split_ratio, rand_seed=42, device='cpu', dtype=torch.float32):
    n_data = train_data[0].shape[0]
    idx_train_1st, idx_train_2nd = train_test_split(np.arange(n_data), train_size=split_ratio, random_state=rand_seed)

    def get_data(data, idx):
        return data[idx] if data is not None else None

    train_1st_data = TrainDataSet(*[get_data(data, idx_train_1st) for data in train_data])
    train_2nd_data = TrainDataSet(*[get_data(data, idx_train_2nd) for data in train_data])
    train_1st_data_t = TrainDataSetTorch.from_numpy(train_1st_data, device=device, dtype=dtype)
    train_2nd_data_t = TrainDataSetTorch.from_numpy(train_2nd_data, device=device, dtype=dtype)

    return train_1st_data_t, train_2nd_data_t

class DspritesTrainData(Dataset):
  """
  A class for input data.
  """
  def __init__(self, train_data: TrainDataSetTorch):
    self.train_data = train_data
    self.len = len(self.train_data.outcome)

  def __getitem__(self, index):
    # return Z, X, Y
    return self.train_data.instrumental[index], self.train_data.treatment[index], self.train_data.outcome[index]

  def __len__(self):
    return self.len

class DspritesTestData(Dataset):
  """
  A class for input data.
  """
  def __init__(self, test_data: TestDataSet):
    self.test_data = test_data
    self.len = len(self.test_data.structural)

  def __getitem__(self, index):
    # return X, Y
    return self.test_data.treatment[index], self.test_data.structural[index]

  def __len__(self):
    return self.len

class InnerModel(nn.Module):
    def __init__(self, sequential):
        super(InnerModel, self).__init__()
        self.model = sequential
        
    def forward(self, x):
        return self.model(x)

class InnerModelLinearHead(nn.Module):
    def __init__(self, sequential, output_dim=32):
        super(InnerModelLinearHead, self).__init__()
        self.model = sequential
        # Add a linear layer to the existing model
        self.linear = nn.Linear(output_dim+1, output_dim, bias=False)
    
    # Define a new forward function incorporating the added linear layer
    def forward(self, x):
        x = self.model(x)
        ones = torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
        x = torch.cat((x, ones), dim=1)
        return self.linear(x)

class OuterModel(nn.Module):
    def __init__(self):
        super(OuterModel, self).__init__()
        self.model = nn.Sequential(nn.Linear(64 * 64, 1024),#spectral_norm(nn.Linear(64 * 64, 1024)),
                                    nn.ReLU(),
                                    nn.Linear(1024, 512),#spectral_norm(nn.Linear(1024, 512)),
                                    nn.ReLU(),
                                    #nn.BatchNorm1d(512),
                                    nn.Linear(512, 128),#spectral_norm(nn.Linear(512, 128)),
                                    nn.ReLU(),
                                    nn.Linear(128, 32),#spectral_norm(nn.Linear(128, 32)),
                                    #nn.BatchNorm1d(32),
                                    nn.Tanh()
                                )

    def forward(self, x):
        res = self.model(x)
        return res

class OuterModelWithNorms(nn.Module):
    def __init__(self):
        super(OuterModelWithNorms, self).__init__()
        self.model = nn.Sequential(spectral_norm(nn.Linear(64 * 64, 1024)),
                                    nn.ReLU(),
                                    spectral_norm(nn.Linear(1024, 512)),
                                    nn.ReLU(),
                                    nn.LayerNorm(512),#nn.BatchNorm1d(512),
                                    spectral_norm(nn.Linear(512, 128)),
                                    nn.ReLU(),
                                    spectral_norm(nn.Linear(128, 32)),
                                    nn.LayerNorm(32),#nn.BatchNorm1d(32),
                                    nn.Tanh()
                                )

        # self.model = nn.Sequential(nn.Linear(64 * 64, 1024),
        #                             nn.ReLU(),
        #                             nn.Linear(1024, 512),
        #                             nn.ReLU(),
        #                             nn.Linear(512, 128),
        #                             nn.ReLU(),
        #                             nn.Linear(128, 32),
        #                             nn.Tanh()
        #                         )


    def forward(self, x):
        res = self.model(x)
        return res

def build_net_for_dsprite(seed, method='sequential'):
    torch.manual_seed(seed)
    sequential = nn.Sequential(nn.Linear(3, 256),#spectral_norm(nn.Linear(3, 256)),
                                nn.ReLU(),
                                nn.Linear(256, 128),#spectral_norm(nn.Linear(256, 128)),
                                nn.ReLU(),
                                #nn.BatchNorm1d(128),
                                nn.Linear(128, 128),#spectral_norm(nn.Linear(128, 128)),
                                nn.ReLU(),
                                #nn.BatchNorm1d(128),
                                nn.Linear(128, 32),#spectral_norm(nn.Linear(128, 32)),
                                #nn.BatchNorm1d(32),
                                nn.ReLU()
                            )
    torch.manual_seed(seed)
    response_net = OuterModel()
    if method == 'sequential':
        instrumental_net = InnerModel(sequential)
    elif method == 'sequential+linear':
        instrumental_net = InnerModelLinearHead(sequential)
    return instrumental_net, response_net

def build_net_for_dsprite_with_norms(seed, method='sequential'):
    set_seed(seed)
    sequential = nn.Sequential(spectral_norm(nn.Linear(3, 256)),
                                    nn.ReLU(),
                                    spectral_norm(nn.Linear(256, 128)),
                                    nn.ReLU(),
                                    nn.LayerNorm(128),#nn.BatchNorm1d(128),
                                    spectral_norm(nn.Linear(128, 128)),
                                    nn.ReLU(),
                                    nn.LayerNorm(128),#nn.BatchNorm1d(128),
                                    spectral_norm(nn.Linear(128, 32)),
                                    nn.LayerNorm(32),#nn.BatchNorm1d(32),
                                    nn.ReLU()
                            )

    # sequential = nn.Sequential(nn.Linear(3, 256),
    #                                 nn.ReLU(),
    #                                 nn.Linear(256, 128),
    #                                 nn.ReLU(),
    #                                 nn.Linear(128, 128),
    #                                 nn.ReLU(),
    #                                 nn.Linear(128, 32),
    #                                 nn.ReLU()
    #                         )


    set_seed(seed)
    response_net = OuterModelWithNorms()
    if method == 'sequential':
        instrumental_net = InnerModel(sequential)
    elif method == 'sequential+linear':
        instrumental_net = InnerModelLinearHead(sequential)
    return instrumental_net, response_net