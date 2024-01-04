import sys
import math
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Add main project directory path
sys.path.append('/home/clear/ipetruli/projects/bilevel-optimization/src')

from model_previous.FunctionApproximator.FunctionApproximator import FunctionApproximator
from model_previous.BilevelProblem.BilevelProblem import BilevelProblem
from model.NeuralNetworks.BirdsResNet import ResNet
from model.NeuralNetworks.NeuralNetworkOuterModel import NeuralNetworkOuterModel
from model_previous.utils import *

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

set_seed(seed=0)

# Initialize dimesnions
n, m, m_v, m_t, batch = 2, 1000, 300, 700, 64
# The coefficient tensor of size (n,1) filled with values uniformally sampled from the range (0,1)
coef = np.random.uniform(size=(n,1)).astype('float32')
# The data tensor of size (m,n) filled with values uniformally sampled from the range (0,1)
X = np.random.uniform(size=(m, n)).astype('float32')
# True h_star
h_true = lambda X: X @ coef
y = h_true(X)
# Split X into 2 tensors with sizes [m_t, m_v] along dimension 0
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)
# Convert everything to PyTorch tensors
X_train, X_val, y_train, y_val, coef = torch.from_numpy(X_train), torch.from_numpy(X_val), torch.from_numpy(y_train), torch.from_numpy(y_val), torch.from_numpy(coef)
print("True coeficients:", coef)
print("X training data:", X_train[1:5])
print("y training labels:", y_train[1:5])
print()

dataset = [X_val,y_val,X_train,y_train]
maxiter = 1000
step = 0.1
mu_0_value = 3.
mu0 = torch.full((1,1), mu_0_value)

plt.rcParams['text.usetex'] = True
fig = plt.figure(figsize=(4.5,4))
ax = fig.add_subplot(projection='3d')
ax.scatter(X_train[:,0], X_train[:,1], y_train, marker='.')
#ax.set_xticks([])
#ax.set_yticks([])
ax.set_xlabel(r'$x_1$', fontsize=12)
ax.set_ylabel(r'$x_2$', fontsize=12)
ax.set_zlabel(r'$h(x)$', fontsize=12)
plt.tight_layout()
plt.savefig("data.png", dpi=500, transparent=True)

# A function to find optimal or closed form solution of h*
def find_theta_star(X, y, mu):
    """
    Find a closed form solution of theta for a fixed mu.
    """
    return torch.linalg.inv((1+2*mu) * X.T @ X) @ X.T @ y

# Objective functions
fo = lambda mu, theta, X_out, y_out: torch.mean((1/2)*torch.pow(((X_out @ theta) - y_out),2))
fi = lambda mu, theta, X_in, y_in: torch.mean((1/2)*torch.pow(((X_in @ theta) - y_in),2) + mu*torch.pow((X_in @ theta),2))

# Gradients
og1 = lambda mu, theta, X_out, y_out: torch.tensor([[0]])
og2 = lambda mu, theta, X_out, y_out: X_out.T @ (X_out @ theta - y_out)
ig22 = lambda mu, theta, X_in, y_in: X_in.T @ X_in + 2 * mu * X_in.T @ X_in
ig12 = lambda mu, theta, X_in, y_in: 2 * X_in.T @ X_in @ theta

# Optimize using classical implicit differention
bp_classic = BilevelProblem(outer_objective=fo, inner_objective=fi, method="implicit_diff", data=dataset, gradients=[og1,og2,ig22,ig12], find_theta_star=find_theta_star)
mu_opt_c, iters_c, n_iters_c, times, inner_loss, outer_loss, theta = bp_classic.optimize(mu0, maxiter=maxiter, step=step)
plot_loss("Inner loss clas.", inner_loss, title="Inner loss of clas. im. diff.")
plot_loss("Outer loss clas.", outer_loss, title="Outer loss of clas. im. diff.")
h_star_c = lambda X: X @ theta

# Show results
print("CLASSICAL IMPLICIT DIFFERENTIATION")
print("Argmin of the outer objective:", mu_opt_c)
print("Average iteration time:", np.average(times))
print("Number of iterations:", n_iters_c)
print("True coefficients theta:", coef)
print("Fitted coefficients theta:", theta)

# Objective functions
fo = lambda mu, h, X_out, y_out: torch.mean((1/2)*torch.pow((h(X_out) - y_out),2))
fi = lambda mu, h, X_in, y_in: torch.mean((1/2)*torch.pow((h(X_in) - y_in),2) + mu*torch.pow(h(X_in),2))

# Gradients
og1 = lambda mu, h, X_out, y_out: torch.tensor([[0]])
og2 = lambda mu, h, X_out, y_out: (h(X_out) - y_out)
ig22 = lambda mu, h, X_in, y_in: torch.eye(len(y_in)) * (1+2*mu)
ig12 = lambda mu, h, X_in, y_in: 2*h(X_in)

# Optimize using neural implicit differention
bp_neural = BilevelProblem(outer_objective=fo, inner_objective=fi, fo_h_X=fo, fi_h_X=fi, method="neural_implicit_diff", data=dataset, gradients=[og1,og2,ig22,ig12])
mu_opt_n, iters_n, n_iters_n, times, inner_loss, outer_loss, h_star_n = bp_neural.optimize(mu0, maxiter=maxiter, step=step)
plot_loss("Inner loss neur.", inner_loss, title="Inner loss of neur. im. diff.")
plot_loss("Outer loss neur.", outer_loss, title="Outer loss of neur. im. diff.")

# Show results
print("NEURAL IMPLICIT DIFFERENTIATION")
print("Argmin of the outer objective:", mu_opt_n)
print("Average iteration time:", np.average(times))
print("Number of iterations:", n_iters_n)
print()

plot_2D_functions("2Dfunctions", h_true, h_star_c, h_star_n, points=None, plot_x_lim=[0,mu_0_value], plot_y_lim=[0,1], plot_nb_contours=80, titles=["True Imp. Diff.","Classical Imp. Diff.","Neural Imp. Diff."])
plot_1D_iterations("1Dfunctions", [i for i in range(len(iters_c))], [i for i in range(len(iters_n))], iters_c, iters_n, plot_x_lim=[0,mu_0_value], titles=["Classical Imp. Diff.","Neural Imp. Diff."])