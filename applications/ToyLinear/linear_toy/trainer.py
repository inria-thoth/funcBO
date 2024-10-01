import torch
from applications.IVRegression.dataset_networks_dsprites.twoSLS import *
from applications.IVRegression.dataset_networks_dsprites.generator import *
import time
from funcBO.utils import assign_device, get_dtype, tensor_to_state_dict, state_dict_to_tensor, auxiliary_toy_data
from torch.utils.data import TensorDataset, DataLoader
from funcBO.InnerSolution import InnerSolution

class Trainer:
    """
    Solves an instrumental regression problem using the bilevel functional method.
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
        self.iters = 0

    def log(self,dico, log_name='metrics'):
        self.logger.log_metrics(dico, log_name=log_name)

    def log_metrics_list(self, dico_list, iteration, prefix="", log_name='metrics'):
        total_iter = len(dico_list)
        for dico in dico_list:
            dico['outer_iter'] = iteration
            dico['iter'] = iteration*total_iter + dico['inner_iter']
            dico = {prefix+key:value for key,value in dico.items()}
            self.log(dico, log_name=log_name)

    def build_trainer(self):
        """
        Builds the trainer by setting up data, models, and optimization components (number of epochs, optimizers, etc.).
        """
        device = self.device

        # Generate synthetic dsprites data
        self.nb_aux_tasks = 4
        self.max_epochs = self.args.max_epochs

        # Data for auxiliary task learning
        n, m, X_outer, X_inner, y_outer, y_inner, coef = auxiliary_toy_data(self.dtype, self.device, seed=self.args.seed)
        # Create a TensorDataset from your tensors
        outer_dataset = TensorDataset(X_outer, y_outer)
        self.outer_dataloader = DataLoader(outer_dataset, batch_size=len(outer_dataset), shuffle=False)
        self.inner_dataset = TensorDataset(X_inner, y_inner)
        self.inner_dataloader = DataLoader(self.inner_dataset, batch_size=len(self.inner_dataset), shuffle=False)

        # Neural networks
        self.outer_model = nn.Sequential(nn.Linear(self.nb_aux_tasks, 1, bias=False))
        self.inner_model = nn.Sequential(nn.Linear(n, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 128),
                                            nn.ReLU(),
                                            nn.Linear(128, 128),
                                            nn.ReLU(),
                                            nn.BatchNorm1d(128),
                                            nn.Linear(128, 32),
                                            nn.ReLU(self.nb_aux_tasks+1))
        
        self.inner_model.to(device)
        self.outer_model.to(device)

        # The outer neural network parametrized by the outer variable
        self.outer_param = torch.nn.parameter.Parameter(state_dict_to_tensor(self.outer_model, device))
        # Print the weights of the first layer
        print("Weights of the first layer:")
        print(self.outer_model[0].weight.data)
        print("Outer parameter weights: ", self.outer_param.data)
        self.outer_param.requires_grad = True

        self.inner_param = torch.nn.parameter.Parameter(state_dict_to_tensor(self.inner_model, device))

        self.outer_optimizer = torch.optim.Adam([self.outer_param], 
                                        lr=self.args.outer_optimizer.outer_lr, 
                                        weight_decay=self.args.outer_optimizer.outer_wd)

        inner_scheduler = None
        inner_dual_scheduler = None
        outer_scheduler = None

        MSE = torch.nn.MSELoss(reduction='mean')
        relu = torch.nn.ReLU()

        def masked_mse_loss(input, target, mask_value=-1):
            mask = target != mask_value
            masked_input = input[mask]
            masked_target = target[mask]
            # Return tensor with 0 if all values are mask_value
            if len(masked_input) == 0:
                return torch.tensor(0.0, dtype=input.dtype, device=input.device)
            return MSE(masked_input, masked_target)

        # Outer objective function
        def fo(h_X_out, y_out):
            main_pred, aux_pred = h_X_out[:,0], h_X_out[:,1:self.nb_aux_tasks+1]
            main_label, aux_label = y_out[:,0], y_out[:,1:self.nb_aux_tasks+1]
            loss = MSE(main_pred, main_label)
            return loss

        # Inner objective function
        def fi(outer_model_val, h_X_in):
            main_pred, aux_pred = h_X_in[:,0], h_X_in[:,1:self.nb_aux_tasks+1]
            for X, y in self.inner_dataloader:
                y_inner = y.to(device)
            main_label, aux_label = y_inner[:,0], y_inner[:,1:self.nb_aux_tasks+1]
            loss = MSE(main_pred.to(device), main_label)
            aux_loss_vector = torch.zeros((self.nb_aux_tasks,1)).to(device)
            for task in range(self.nb_aux_tasks):
                aux_loss_vector[task] = MSE(aux_pred[:,task].to(device), aux_label[:,task].to(device))
            outer_model_inputs = aux_loss_vector.T
            return loss + outer_model_val

        def projector(data):
            # Should return inner_model_inputs, outer_model_inputs, inner_loss_inputs
            X, y = data
            return X, None, None
        
        self.inner_loss = fi
        self.outer_loss = fo

        self.inner_solution = InnerSolution(self.inner_model,
                                            self.inner_loss, 
                                            self.inner_dataloader,
                                            projector,
                                            self.outer_model,
                                            self.outer_param,
                                            dual_model_args = self.args.dual_model,
                                            inner_solver_args = self.args.inner_solver,
                                            dual_solver_args= self.args.dual_solver
                                            )


    def train(self):
        """
        The main optimization loop for the bilevel functional method.
        """
        # This is an atribute to be able to start in the middle if starting from a checkpoint
        done = False
        while not done:
            for X, Y in self.outer_dataloader:
                metrics_dict = {}
                metrics_dict['iter'] = self.iters
                start = time.time()
                # Move data to GPU
                X_outer = X.to(self.device)
                Y_outer = Y.to(self.device)
                # Inner value corresponds to h*(Z)
                forward_start = time.time()
                # Get the value of h*(Z_outer)
                inner_value = self.inner_solution(X_outer)
                inner_value.retain_grad()
                loss = self.outer_loss(inner_value, Y_outer)
                # Backpropagation
                self.outer_optimizer.zero_grad()
                backward_start = time.time()
                loss.backward()
                self.outer_optimizer.step()
                metrics_dict['outer_loss'] = loss.item()
                inner_logs = self.inner_solution.inner_solver.data_logs
                if inner_logs:
                    self.log_metrics_list(inner_logs, self.iters, log_name='inner_metrics')
                dual_logs = self.inner_solution.dual_solver.data_logs
                if dual_logs:
                    self.log_metrics_list(dual_logs, self.iters, log_name='dual_metrics')
                self.log(metrics_dict)
                print(metrics_dict)
                self.iters += 1
                done = (self.iters >= self.max_epochs)
                if done:
                    break
                print("Outer parameter weights: ", self.outer_param.data)
        if self.validation_data:
            val_log = [{'inner_iter': 0,
                        'val loss': (self.evaluate(data_type="validation")).item()}]
            self.log_metrics_list(val_log, 0, log_name='val_metrics')
        if self.test_data:
            test_log = [{'inner_iter': 0,
                        'test loss': (self.evaluate(data_type="test")).item()}]
            self.log_metrics_list(test_log, 0, log_name='test_metrics')

    def evaluate(self, data_type="validation"):
        """
        Evaluates the performance of the model on the given data.
        Returns:
        - float: The evaluation loss on the provided data.
        """
        return None











