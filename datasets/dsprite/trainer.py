import torch
import numpy as np
import wandb
from torch import nn

# Loss helper functions
MSE = nn.MSELoss()

def augment_stage1_feature(instrumental_feature):
    feature = instrumental_feature
    feature = add_const_col(feature)
    return feature

def augment_stage2_feature(predicted_treatment_feature):
    feature = predicted_treatment_feature
    feature = add_const_col(feature)
    return feature

def outer_prod(mat1: torch.Tensor, mat2: torch.Tensor):
    mat1_shape = tuple(mat1.size())
    mat2_shape = tuple(mat2.size())
    assert mat1_shape[0] == mat2_shape[0]
    nData = mat1_shape[0]
    aug_mat1_shape = mat1_shape + (1,) * (len(mat2_shape) - 1)
    aug_mat1 = torch.reshape(mat1, aug_mat1_shape)
    aug_mat2_shape = (nData,) + (1,) * (len(mat1_shape) - 1) + mat2_shape[1:]
    aug_mat2 = torch.reshape(mat2, aug_mat2_shape)
    return aug_mat1 * aug_mat2

def add_const_col(mat: torch.Tensor):
    assert mat.dim() == 2
    n_data = mat.size()[0]
    device = mat.device
    return torch.cat([mat, torch.ones((n_data, 1), device=device)], dim=1)

def linear_reg_pred(feature, weight):
    assert weight.dim() >= 2
    if weight.dim() == 2:
        return torch.matmul(feature, weight)
    else:
        return torch.einsum("nd,d...->n...", feature, weight)

def linear_reg_loss(target, feature, reg):
    weight = fit_linear(target, feature, reg)
    pred = linear_reg_pred(feature, weight)
    loss = torch.norm((target - pred)) ** 2 + reg * torch.norm(weight) ** 2
    return loss

def fit_linear(target, feature, reg):
    assert feature.dim() == 2
    assert target.dim() >= 2
    nData, nDim = feature.size()
    A = torch.matmul(feature.t(), feature)
    device = feature.device
    A = A + reg * torch.eye(nDim, device=device)
    # U = torch.cholesky(A)
    # A_inv = torch.cholesky_inverse(U)
    #TODO use cholesky version in the latest pytorch
    A_inv = torch.inverse(A)
    if target.dim() == 2:
        b = torch.matmul(feature.t(), target)
        weight = torch.matmul(A_inv, b)
    else:
        b = torch.einsum("nd,n...->d...", feature, target)
        weight = torch.einsum("de,d...->e...", A_inv, b)
    return weight

def fit_2sls(treatment_1st_feature, instrumental_1st_feature, instrumental_2nd_feature, outcome_2nd_t, lam1, lam2):
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

def fit_2n_stage(instrumental_2nd_feature, 
                 outcome_2nd_t, 
                 lam2):
    # stage1
    #feature = augment_stage1_feature(instrumental_1st_feature)
    #stage1_weight = fit_linear(treatment_1st_feature, feature, lam1)
    # predicting for stage 2
    #feature = augment_stage1_feature(instrumental_2nd_feature)
    #predicted_treatment_feature = linear_reg_pred(feature, stage1_weight)
    # stage2
    feature = augment_stage2_feature(instrumental_2nd_feature)
    stage2_weight = fit_linear(outcome_2nd_t, feature, lam2)
    pred = linear_reg_pred(feature, stage2_weight)
    stage2_loss = torch.norm((outcome_2nd_t - pred)) ** 2 + lam2 * torch.norm(stage2_weight) ** 2
    return dict(stage2_weight=stage2_weight,
                stage2_loss=stage2_loss)


class DFIVTrainer:
    """
    Solves an instrumental regression problem using the DFIV method.
    """

    def __init__(self, treatment_net, instrumental_net, treatment_opt, instrumental_opt, train_params):
        # Number of iterations in stage 1
        self.stage1_iter: int = train_params["stage1_iter"]
        # Number of iterations in stage 2
        self.stage2_iter: int = train_params["stage2_iter"]
        # Maximum number of training epochs
        self.max_epochs: int = train_params["max_epochs"]
        # Regularizer in stage 1
        self.lam1: int = train_params["lam1"]
        # Regularizer in stage 2
        self.lam2: int = train_params["lam2"]
        # The feature map f(X)
        self.treatment_net = treatment_net
        self.treatment_opt = treatment_opt
        # The feature map g(Z)
        self.instrumental_net = instrumental_net
        self.instrumental_opt = instrumental_opt

    def train(self, stage1_dataset, stage2_dataset, test_dataset):
        """
        Solves the IV regression problem.
        """
        self.lam1 *= stage1_dataset[0].size()[0]
        self.lam2 *= stage2_dataset[0].size()[0]
        for epoch in range(self.max_epochs):
            self.stage1_update(stage1_dataset)
            u = self.stage2_update(stage1_dataset, stage2_dataset)
            self.evaluate(test_dataset, u)

    def stage1_update(self, stage1_dataset):
        """
        Perform first stage of DFIV.
        """
        self.treatment_net.train(False)
        # Train only the feature map g(Z)
        self.instrumental_net.train(True)
        # Get the value of f(X)
        treatment_feature = self.treatment_net(stage1_dataset.treatment).detach()
        # Train the feature mapping g(Z)
        for i in range(self.stage1_iter):
            self.instrumental_opt.zero_grad()
            # Get the value of g(Z)
            instrumental_feature = self.instrumental_net(stage1_dataset.instrumental)
            feature = augment_stage1_feature(instrumental_feature)
            loss = linear_reg_loss(treatment_feature, feature, self.lam1)
            wandb.log({"in. loss": loss.item()})
            loss.backward()
            grad_norm = (sum([torch.norm(p.grad)**2 for p in self.instrumental_net.parameters()]))
            self.instrumental_opt.step()
            
    def stage2_update(self, stage1_dataset, stage2_dataset):
        """
        Perform second stage of DFIV.
        """
        # Train only the feature map f(X)
        self.treatment_net.train(True)
        self.instrumental_net.eval()
        # Get the value of g(Z)_stage1
        instrumental_1st_feature = self.instrumental_net(stage1_dataset.instrumental).detach()
        # Get the value of g(Z)_stage2
        instrumental_2nd_feature = self.instrumental_net(stage2_dataset.instrumental).detach()
        # Train the feature mapping f(X)
        for i in range(self.stage2_iter):
            self.treatment_opt.zero_grad()
            # Get the value of f(X)_stage1
            treatment_1st_feature = self.treatment_net(stage1_dataset.treatment)
            res = fit_2sls(treatment_1st_feature, instrumental_1st_feature, instrumental_2nd_feature, stage2_dataset.outcome, self.lam1, self.lam2)
            loss = res["stage2_loss"]
            wandb.log({"out. loss": loss.item()})
            loss.backward()
            wandb.log({"outer var. norm": (torch.sqrt(sum([torch.norm(p.data)**2 for p in self.treatment_net.parameters()]))).item()})
            wandb.log({"outer var. grad. norm": (torch.sqrt(sum([torch.norm(p.grad)**2 for p in self.treatment_net.parameters()]))).item()})
            self.treatment_opt.step()
        return res["stage2_weight"]
    
    def evaluate(self, test_dataset, u):
        """
        Evaluate the prediction quality on the test dataset.
        """
        self.treatment_net.train(False)
        self.instrumental_net.train(False)
        # Get the value of f(X)
        treatment_feature = self.treatment_net(test_dataset.treatment).detach()
        loss = MSE((treatment_feature @ u[:-1]) + u[-1], test_dataset.structural)