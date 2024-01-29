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