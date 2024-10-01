import torch
from applications.IVRegression.dataset_networks_dsprites.twoSLS import *
from applications.IVRegression.dataset_networks_dsprites.generator import *
import time
from funcBO.utils import assign_device, get_dtype
from torchviz import make_dot

class Trainer:
    """
    Solves an instrumental regression problem using the Gradient Penalty method.
    """
    def __init__(self, config, logger):
        """
        Initializes the Trainer class with the provided configuration and logger.

        Parameters:
        - config (object): Configuration object containing various settings.
        - logger (object): Logger object for logging metrics and information.
        """
        self.logger = logger
        self.args = config
        self.device = assign_device(self.args.system.device)
        self.dtype = get_dtype(self.args.system.dtype)
        torch.set_default_dtype(self.dtype)
        self.build_trainer()

    def log(self, dico, log_name='metrics'):
        self.logger.log_metrics(dico, log_name=log_name)

    def build_trainer(self):
        """
        Builds the trainer by setting up data, models, and optimization components (number of epochs, optimizer, etc.).
        """
        # Generate synthetic dsprites data
        self.test_data = generate_test_dsprite(device=self.device)
        self.train_data, validation_data = generate_train_dsprite(data_size=self.args.data_size,
                                                            rand_seed=self.args.seed,
                                                            val_size=self.args.val_size)
        if validation_data:
            self.validation_data = TrainDataSetTorch.from_numpy(validation_data, device=self.device, dtype=self.dtype)
        else:
            self.validation_data = validation_data
        self.inner_data, self.outer_data = split_train_data(self.train_data, split_ratio=self.args.split_ratio, rand_seed=self.args.seed, device=self.device, dtype=self.dtype)
        self.test_data = TestDataSetTorch.from_numpy(self.test_data, device=self.device, dtype=self.dtype)

        self.instrumental_network, self.treatment_network = build_net_for_dsprite(self.args.seed, method='sequential')

        # Build the auxiliary network by copying the instrumental network
        self.auxiliary_network = copy.deepcopy(self.instrumental_network)
        self.auxiliary_network.to(self.device)
        self.aux_optimizer = torch.optim.Adam(self.auxiliary_network.parameters(),
                                            lr=self.args.aux_optimizer.lr, 
                                            weight_decay=self.args.aux_optimizer.wd)

        self.instrumental_network.to(self.device)
        self.treatment_network.to(self.device)
        # Optimizer that improves the treatment network
        params = list(self.treatment_network.parameters()) + list(self.instrumental_network.parameters())
        self.optimizer = torch.optim.Adam(params,
                                        lr=self.args.optimizer.lr, 
                                        weight_decay=self.args.optimizer.wd)
        # Hyperparameters
        self.penalty = self.args.penalty
        self.lam1 = self.args.lam_V
        self.lam2 = self.args.lam_u
        self.aux_iters = self.args.aux_iters

    def train(self):
        """
        The main optimization loop for the Penalty method.
        """
        self.lam1 *= self.inner_data[0].size()[0]
        self.lam2 *= self.outer_data[0].size()[0]
        for epoch in range(self.args.max_epochs):
            self.optimizer.zero_grad()
            # Compute the loss
            loss = self.penalty_loss()
            # Backpropagation
            loss.backward()
            self.optimizer.step()
            # Log the metrics
            self.log({'iter': epoch, 'loss': loss.item()}, log_name='metrics')
        # Cmopute the validation loss
        if self.validation_data:
            self.log({'iter': epoch, 'val_loss': (self.evaluate(self.inner_data, self.outer_data, data_type='validation_data')).item()}, log_name='val_metrics')
        # Evaluate the model
        self.log({'iter': epoch, 'test loss': (self.evaluate(self.inner_data, self.outer_data)).item()}, log_name='test_metrics')

    def penalty_loss(self):
        ### Optimize the value of the loss function
        for i in range(self.aux_iters):
            self.aux_optimizer.zero_grad()
            # Get the value of f(X)
            treatment_feature = self.treatment_network(self.inner_data.treatment).detach()
            # Get the value of aux(Z)
            instrumental_feature = self.auxiliary_network(self.inner_data.instrumental)
            feature = augment_stage1_feature(instrumental_feature)
            aux_loss = linear_reg_loss(treatment_feature, feature, self.lam1)
            aux_loss.backward()
            self.aux_optimizer.step()
        ### Compute the min value of the loss function
        # Get the value of f(X)
        treatment_feature = self.treatment_network(self.inner_data.treatment)
        # Get the value of aux(Z)
        instrumental_feature = self.auxiliary_network(self.inner_data.instrumental).detach()
        feature = augment_stage1_feature(instrumental_feature)
        aux_loss = linear_reg_loss(treatment_feature, feature, self.lam1)
        ### Compute the total loss
        # Get the value of f(X)
        treatment_feature = self.treatment_network(self.inner_data.treatment)
        # Get the value of g(Z)
        instrumental_feature = self.instrumental_network(self.inner_data.instrumental)
        feature = augment_stage1_feature(instrumental_feature)
        loss_inn = linear_reg_loss(treatment_feature, feature, self.lam1)
        # Compute the difference between inner loss and min value of the inner loss
        loss1 = loss_inn - aux_loss
        # Compute stage 2 loss
        # Get the value of g(Z)_stage1
        instrumental_1st_feature = self.instrumental_network(self.inner_data.instrumental)
        # Get the value of g(Z)_stage2
        instrumental_2nd_feature = self.instrumental_network(self.outer_data.instrumental)
        # Get the value of f(X)_stage1
        treatment_1st_feature = self.treatment_network(self.inner_data.treatment)
        res = fit_2sls(treatment_1st_feature, instrumental_1st_feature, instrumental_2nd_feature, self.outer_data.outcome, self.lam1, self.lam2)
        loss2 = res["stage2_loss"]
        total_loss = loss2 + self.penalty*loss1
        #dot = make_dot(total_loss, params={"treatment_net_layer0_wight0": list(self.treatment_network.parameters())[0], "instrumental_net_layer0_wight0": list(self.instrumental_network.parameters())[0]})
        #dot.render("attached", format="png")
        return total_loss
    
    def evaluate(self, stage1_dataset, stage2_dataset, data_type='test_data'):
        """
        Evaluate the prediction quality on the test dataset.
        """
        self.treatment_network.eval()
        self.instrumental_network.eval()
        with torch.no_grad():
            # Find stage2_weight
            treatment_1st_feature = self.treatment_network(stage1_dataset.treatment).detach()
            instrumental_1st_feature = self.instrumental_network(stage1_dataset.instrumental).detach()
            instrumental_2nd_feature = self.instrumental_network(stage2_dataset.instrumental).detach()
            feature = augment_stage1_feature(instrumental_1st_feature)
            stage1_weight = fit_linear(treatment_1st_feature, feature, self.lam1)
            feature = augment_stage1_feature(instrumental_2nd_feature)
            predicted_treatment_feature = linear_reg_pred(feature, stage1_weight)
            feature = augment_stage2_feature(predicted_treatment_feature)
            stage2_weight = fit_linear(stage2_dataset.outcome, feature, self.lam2)
            # Compute the test loss
            if data_type == 'test_data':
                Y_test = self.test_data.structural
                X_test = self.test_data.treatment
            elif data_type == 'validation_data':
                Y_test = self.validation_data.outcome
                X_test = self.validation_data.treatment
            treatment_feature = self.treatment_network(X_test).detach()
            test_feature = augment_stage2_feature(treatment_feature)
            test_pred = linear_reg_pred(test_feature, stage2_weight)
            loss = (torch.norm((Y_test - test_pred)) ** 2) / Y_test.size(0)
        return loss