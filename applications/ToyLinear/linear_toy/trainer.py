import torch
from applications.IVRegression.dataset_networks_dsprites.twoSLS import *
from applications.IVRegression.dataset_networks_dsprites.generator import *
import time
from funcBO.utils import assign_device, get_dtype, tensor_to_state_dict, state_dict_to_tensor
from torch.utils.data import DataLoader
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
        self.nb_aux_tasks = self.args.nb_aux_tasks
        self.max_epochs = self.args.max_epochs
        
        # Neural networks
        self.inner_model, self.outer_model = None, None
        
        # The outer neural network parametrized by the outer variable
        self.outer_param = torch.nn.parameter.Parameter(state_dict_to_tensor(self.outer_model, device))

        self.inner_param = torch.nn.parameter.Parameter(state_dict_to_tensor(self.inner_model, device))

        self.outer_optimizer = torch.optim.Adam([self.outer_param], 
                                        lr=self.args.outer_optimizer.outer_lr, 
                                        weight_decay=self.args.outer_optimizer.outer_wd)

        inner_scheduler = None
        inner_dual_scheduler = None
        outer_scheduler = None

        MSE = torch.nn.MSELoss(reduction='mean')

        # Outer objective function
        def fo(h_X_out, y_out):
            main_pred, aux_pred = h_X_out[0], h_X_out[1:self.nb_aux_tasks+1]
            main_label, aux_label = y_out[0], y_out[1:self.nb_aux_tasks+1]
            loss = MSE(main_pred, main_label)
            return loss

        # Inner objective function
        def fi(h_X_in, y_in):
            # Keep the weights of the labels positive
            self.outer_param = relu(self.outer_param)
            main_pred, aux_pred = h_X_in[0], h_X_in[1:self.nb_aux_tasks+1]
            aux_pred = torch.sigmoid(aux_pred)
            main_label, aux_label = y_in[0], y_in[1:self.nb_aux_tasks+1]
            aux_label = aux_label
            loss = MSE(main_pred, main_label)
            aux_loss = self.outer_model(aux_pred)
            return loss + aux_loss

        def projector(data):
            """
            Extracts and returns the relevant components (z, x) from the input data.

            Parameters:
            - data (tuple): A tuple containing information, where the first element represents z,
                        the second element represents x, and the third element may contain
                        additional information (ignored in this context).

            Returns:
            - tuple: A tuple containing the relevant components (z, x) extracted from the input data.
            """
            z,x,_ = data
            return z,x,None
        
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











