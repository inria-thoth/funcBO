import torch
from applications.IVRegression.dsprites_data.trainer import *
from applications.IVRegression.dsprites_data.generator import *
import time
from funcBO.utils import assign_device, get_dtype

class Trainer:
    """
    Solves an instrumental regression problem using the DFIV method.
    """
    def __init__(self, config, logger, NNs_with_norms=False):
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
        self.NNs_with_norms = NNs_with_norms
        self.build_trainer()

    def log(self,dico, log_name='metrics'):
        self.logger.log_metrics(dico, log_name=log_name)

    def build_trainer(self):
        """
        Builds the trainer by setting up data, models, and optimization components (number of epochs, optimizers, etc.).
        """
        device = self.device

        # Generate synthetic dsprites data
        test_data = generate_test_dsprite(device=device)
        train_data, validation_data = generate_train_dsprite(data_size=self.args.data_size,
                                                            rand_seed=self.args.seed,
                                                            val_size=self.args.val_size)
        self.inner_data, self.outer_data = split_train_data(train_data, split_ratio=self.args.split_ratio, rand_seed=self.args.seed, device=device, dtype=self.dtype)
        self.test_data = TestDataSetTorch.from_numpy(test_data, device=device, dtype=self.dtype)

        # Neural networks for dsprites data
        #if self.NNs_with_norms:
        instrumental_network, treatment_network = build_net_for_dsprite_with_norms(self.args.seed, method='sequential')
        #else:
        #    instrumental_network, treatment_network = build_net_for_dsprite(self.args.seed, method='sequential')
        instrumental_network.to(device)
        treatment_network.to(device)

        # Optimizer that improves the treatment network
        treatment_optimizer = torch.optim.Adam(treatment_network.parameters(),
                                        lr=self.args.treatment_optimizer.lr, 
                                        weight_decay=self.args.treatment_optimizer.wd)

        # Optimizer that improves the instrumental network
        instrumental_optimizer = torch.optim.Adam(instrumental_network.parameters(),
                                        lr=self.args.instrumental_optimizer.lr, 
                                        weight_decay=self.args.instrumental_optimizer.wd)

        # Initialize and train the DFIVTrainer
        self.dfiv_trainer = DFIVTrainer(
            treatment_network, instrumental_network, treatment_optimizer, instrumental_optimizer,
            train_params = {"stage1_iter": self.args.instrumental_optimizer.num_iter, "stage2_iter": self.args.treatment_optimizer.num_iter, "max_epochs": self.args.max_epochs, "lam1": self.args.lam_V, "lam2": self.args.lam_u},
            logger = self.logger
        )
        
    def train(self):
        """
        The main optimization loop for the DFIV method.
        """
        self.dfiv_trainer.train(self.inner_data, self.outer_data, self.test_data)