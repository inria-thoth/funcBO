import torch
import random
import numpy as np
import matplotlib.pyplot as plt

def plot_2D_functions(figname, f1, f2, f3, points=None, plot_x_lim=[-5,5], plot_y_lim=[-5,5], plot_nb_contours=10, titles=["True function","Classical Imp. Diff.","Functional Imp. Diff."]):
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
    ax1.set_title(r'True $h^*(x)$', fontsize=12)
    ax1.set_xlabel(r'$x_1$', fontsize=12)
    ax1.set_ylabel(r'$x_2$', fontsize=12)
    ax1.set_xticks([])
    ax1.set_yticks([])
    # Visualize the approximated function.
    ax2.contour(X, Y, Z2, plot_nb_contours, cmap=plt.cm.magma, alpha=0.8, extend='both')
    if not (points is None):
        ax2.scatter(points[:,0], points[:,1], marker='.')
    ax2.set_title(r'Classical Imp. Diff.', fontsize=12)
    ax2.set_xlabel(r'$x_1$', fontsize=12)
    ax2.set_ylabel(r'$x_2$', fontsize=12)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax3.contour(X, Y, Z3, plot_nb_contours, cmap=plt.cm.magma, alpha=0.8, extend='both')
    ax3.set_title(r'Functional Imp. Diff.', fontsize=12)
    ax3.set_xlabel(r'$x_1$', fontsize=12)
    ax3.set_ylabel(r'$x_2$', fontsize=12)
    ax3.set_xticks([])
    ax3.set_yticks([])
    plt.tight_layout()
    plt.savefig(figname+".png", dpi=600, transparent=True)

def plot_1D_iterations(figname, iters1, iters2, mu1, mu2, plot_x_lim=[0,1], titles=["Classical Imp. Diff.","Functional Imp. Diff."]):
    # Create a part of the domain.
    X = np.linspace(plot_x_lim[0], plot_x_lim[1], 100)
    # Visualize the true function.
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(7, 3))
    #Z1, Z2 = np.zeros_like(X), np.zeros_like(X)
    #for i in range(0, len(X)):
    #    a = (torch.from_numpy(np.array(X[i], dtype='float32')))
    #    Z1[i] = f1(a).float()
    #    Z2[i] = f2(a).float()
    #Z3, Z4 = np.zeros(len(iters1)), np.zeros(len(iters2))
    #X1, X2 = np.zeros(len(iters1)), np.zeros(len(iters2))
    #for i in range(0, len(iters1)):
    #    a = iters1[i]
    #    X1[i] = a.float()
    #    Z3[i] = f1(a).float()
    #for i in range(0, len(iters2)):
    #    a = iters2[i]
    #    X2[i] = a.float()
    #    Z4[i] = f2(a).float()
    ax1.scatter(iters1, mu1, marker='.')
    ax1.scatter(iters1[-1], mu1[-1], marker='*')
    ax1.set_title(r'Classical Imp. Diff.', fontsize=12)
    ax1.set_xlabel(r'iteration', fontsize=12)
    ax1.set_ylabel(r'outer variable $\omega$', fontsize=12)
    ax2.scatter(iters2, mu2, marker='.')
    ax2.scatter(iters2[-1], mu2[-1], marker='*')
    ax2.set_title(r'Functional Imp. Diff.', fontsize=12)
    ax2.set_xlabel(r'iteration', fontsize=12)
    ax2.set_ylabel(r'outer variable $\omega$', fontsize=12)
    plt.tight_layout()
    plt.savefig(figname+".png", dpi=600, transparent=True)

def plot_loss(figname, loss_values, title="Step-wise Loss", labels=None):
    """
    Plot the loss value over iterations.
    	param loss_values: list of values to be plotted
    """
    loss_values = [tensor.item() for tensor in loss_values]
    ticks = np.arange(0, len(loss_values), 1)
    fig, ax = plt.subplots(figsize=(4,3))
    plt.plot(ticks, loss_values)
    plt.title(title)
    n_ticks = len(ticks)
    if n_ticks > 10:
        labels = ['']*n_ticks
    plt.xticks(ticks=ticks, labels=labels)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
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

def set_seed(seed=0):
    """
    A function to set the random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False