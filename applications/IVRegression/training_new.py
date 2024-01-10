import torch
import wandb

from torch.utils.data import DataLoader
from torch.nn import MSELoss
import time

from funcBO.InnerSolution_new import InnerSolution

import os
os.environ['WANDB_DISABLED'] = 'true'
os.chdir('/home/ipetruli/funcBO')
from datasets.dsprite.dspriteBilevel import *
from datasets.dsprite.trainer import *

from funcBO.utils import state_dict_to_tensor, tensor_to_state_dict

def assign_device(device):
    """
    Assigns a device for PyTorch based on the provided device identifier.

    Parameters:
    - device (int): Device identifier. If positive, it represents the GPU device
                   index; if -1, it sets the device to 'cuda'; if -2, it sets
                   the device to 'cpu'.

    Returns:
    - device (str): The assigned device, represented as a string. 
                    'cuda:X' if device > -1 and CUDA is available, where X is 
                    the provided device index. 'cuda' if device is -1.
                    'cpu' if device is -2.
    """
    if device >-1:
        device = (
            'cuda:'+str(device) 
            if torch.cuda.is_available() and device>-1 
            else 'cpu'
        )
    elif device==-1:
        device = 'cuda'
    elif device==-2:
        device = 'cpu'
    return device

def get_dtype(dtype):
    """
    Returns the PyTorch data type based on the provided integer identifier.

    Parameters:
    - dtype (int): Integer identifier representing the desired data type.
                   64 corresponds to torch.double, and 32 corresponds to torch.float.

    Returns:
    - torch.dtype: PyTorch data type corresponding to the provided identifier.

    Raises:
    - NotImplementedError: If the provided identifier is not recognized (not 64 or 32).
    """
    if dtype==64:
        return torch.double
    elif dtype==32:
        return torch.float
    else:
        raise NotImplementedError('Unkown type')

def make_run_name(inner_lr,
                  inner_dual_lr,
                  outer_lr,
                  inner_wd,
                  inner_dual_wd,
                  outer_wd,
                  max_inner_dual_epochs,
                  max_inner_epochs,
                  seed,
                  lam_u,
                  lam_V,
                  batch_size,
                  max_epochs):
    run_name = "Dsprites bilevel::="
    run_name+=" inner_lr:"+str(inner_lr)
    run_name+=", dual_lr:"+str(inner_dual_lr)
    run_name+=", outer_lr:"+str(outer_lr)
    run_name+=", inner_wd:"+str(inner_wd)
    run_name+=", dual_wd:"+str(inner_dual_wd)
    run_name+=", outer_wd:"+str(outer_wd)
    run_name+=", max_inner_dual_epochs:"+str(max_inner_dual_epochs)
    run_name+=", max_inner_epochs:"+str(max_inner_epochs)
    run_name+=", seed:"+str(seed)
    run_name+=", lam_u:"+str(lam_u)
    run_name+=", lam_V:"+str(lam_V)
    run_name+=", batch_size:"+str(batch_size)
    run_name+=", max_epochs:"+str(max_epochs)
    return run_name

class Trainer:
    """
    Solves an instrumental regression problem using the bilevel functional method.
    """
    def __init__(self,config,logger):
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
        self.build_trainer() 

    def log(self,dico):
        self.logger.log_metrics(dico, log_name="metrics")

    def build_trainer(self):
        """
        Builds the trainer by setting up data, models, and optimization components (losses, optimizers).
        """
        device = self.device
        # Get data
        test_data = generate_test_dsprite(device=device)
        train_data, validation_data = generate_train_dsprite(data_size=self.args.data_size, 
                                                rand_seed=self.args.seed, 
                                                device=device, 
                                                val_size=self.args.val_size)
        inner_data, outer_data = split_train_data(train_data, split_ratio=self.args.split_ratio)

        # Weird scaling of lambdas done in training
        lam_V = self.args.lam_V*inner_data[0].size()[0]
        lam_u = self.args.lam_u*outer_data[0].size()[0]

        instrumental_in= inner_data.instrumental
        treatment_in = inner_data.treatment
        outcome_in = inner_data.outcome
        instrumental_out = outer_data.instrumental
        treatment_out = outer_data.treatment
        outcome_out = outer_data.outcome
        treatment_test = test_data.treatment
        outcome_test = test_data.structural

        if not (validation_data is None):
            instrumental_val = validation_data.instrumental
            treatment_val = validation_data.treatment
            outcome_val = validation_data.outcome

        # Dataloaders for inner and outer data
        inner_data = DspritesData(instrumental_in, treatment_in, outcome_in)
        self.inner_dataloader = DataLoader(dataset=inner_data, batch_size=self.args.batch_size, shuffle=False)#, drop_last=True)
        outer_data = DspritesData(instrumental_out, treatment_out, outcome_out)
        self.outer_dataloader = DataLoader(dataset=outer_data, batch_size=self.args.batch_size, shuffle=False)#, drop_last=True)
        self.test_data = DspritesTestData(treatment_test, outcome_test)
        if not (validation_data is None):
            self.validation_data = DspritesData(instrumental_val, treatment_val, outcome_val)
        else:
            self.validation_data = validation_data 
        inner_data.instrumental = inner_data.instrumental.to(device)
        inner_data.treatment = inner_data.treatment.to(device)

        # Neural networks for dsprites data
        self.inner_model, self.outer_model = build_net_for_dsprite(self.args.seed)
        self.inner_model.to(device)
        self.outer_model.to(device)
        print("First inner layer:", list(self.inner_model.parameters())[0].data)
        print("First outer layer:", list(self.outer_model.parameters())[0].data)

        # The outer neural network parametrized by the outer variable
        self.outer_param = torch.nn.parameter.Parameter(state_dict_to_tensor(self.outer_model, device))

        # Optimizer that improves the outer variable
        self.outer_optimizer = torch.optim.Adam([self.outer_param], 
                                        lr=self.args.outer_optimizer.outer_lr, 
                                        weight_decay=self.args.outer_optimizer.outer_wd)

        inner_scheduler = None
        inner_dual_scheduler = None
        outer_scheduler = None

        # # Print configuration
        # run_name = make_run_name(self.args.optimizer.inner_lr,
        #                           self.args.optimizer.inner_dual_lr,
        #                           self.args.optimizer.outer_lr,
        #                           self.args.optimizer.inner_wd,
        #                           self.args.optimizer.inner_dual_wd,
        #                           self.args.optimizer.outer_wd,
        #                           self.args.max_inner_dual_epochs,
        #                           self.args.max_inner_epochs,
        #                           self.args.seed,
        #                           self.args.lam_u,
        #                           self.args.lam_V,
        #                           self.args.batch_size,
        #                           self.args.max_epochs)

        # print("Run configuration:", run_name)

        # Set logging
        wandb.init(group="Dsprites_bilevelIV_general")

        # Loss helper functions
        self.MSE = nn.MSELoss()

        # Outer objective function
        def fo(g_z_out, Y):
            res = fit_2n_stage(
                            g_z_out, 
                            Y,  
                            lam_u)
            return res["stage2_loss"], res["stage2_weight"]

        # Inner objective function that depends only on the inner prediction
        def fi(outer_param, instrumental_feature, X):
            outer_NN_dic = tensor_to_state_dict(self.outer_model, 
                                                self.outer_param, 
                                                device)
            treatment_feature = (torch.func.functional_call(self.outer_model, 
                                parameter_and_buffer_dicts=outer_NN_dic, 
                                args=X, 
                                strict=True))
            loss = torch.norm((instrumental_feature - treatment_feature)) ** 2
            return loss

        # Inner objective function with regularization
        def reg_objective(outer_param, g_z_in, X):
            outer_NN_dic = tensor_to_state_dict(self.outer_model, 
                                                self.outer_param, 
                                                device)
            treatment_feature = (torch.func.functional_call(self.outer_model, 
                                parameter_and_buffer_dicts=outer_NN_dic, 
                                args=X, 
                                strict=True))
            # Get the value of g(Z)
            instrumental_feature = g_z_in
            feature = augment_stage1_feature(instrumental_feature)
            weight = fit_linear(treatment_feature, feature, lam_V)
            pred = linear_reg_pred(feature, weight)
            loss = torch.norm((treatment_feature - pred)) ** 2 + lam_V * torch.norm(weight) ** 2
            return loss, weight

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
            return z,x

        if self.args.inner_solver['name']=='funcBO.solvers.CompositeSolver':
            self.args.inner_solver['reg_objective'] = reg_objective

        if self.args.dual_solver['name']=='funcBO.solvers.ClosedFormSolver':
            self.args.dual_solver['reg'] = lam_V
        
        self.inner_loss = fi
        self.outer_loss = fo

        self.inner_solution = InnerSolution(self.inner_model,
                                            self.inner_loss, 
                                            self.inner_dataloader,
                                            projector,
                                            self.outer_param,
                                            dual_model_args = self.args.dual_model,
                                            inner_solver_args = self.args.inner_solver,
                                            dual_solver_args= self.args.dual_solver
                                            )

    def train(self):
        """
        The main optimization loop for the bilevel functional method.
        """
        iters = 0
        for epoch in range(self.args.max_epochs):
          for Z, X, Y in self.outer_dataloader:
            metrics_dict = {}
            start = time.time()
            # Move data to GPU
            Z_outer = Z.to(self.device, dtype=torch.float)
            X_outer = X.to(self.device, dtype=torch.float)
            Y_outer = Y.to(self.device, dtype=torch.float)
            # Inner value corresponds to h*(Z)
            forward_start = time.time()
            # Get the value of h*(Z_outer)
            inner_value = self.inner_solution(Z_outer)
            metrics_dict['forward_time']= time.time() - forward_start
            loss, u = self.outer_loss(inner_value, Y_outer)
            # Backpropagation
            self.outer_optimizer.zero_grad()
            backward_start = time.time()
            loss.backward()
            self.outer_optimizer.step()
            wandb.log({"out. loss": loss.item()})
            wandb.log({"outer var. norm": torch.norm(self.outer_param).item()})
            wandb.log({"outer var. grad. norm": torch.norm(self.outer_param.grad).item()})
            metrics_dict['norm_grad_outer_param']= torch.norm(self.outer_param.grad).item()
            metrics_dict['forward_time']= time.time() - backward_start
            metrics_dict['iter_time']= time.time() - start
            metrics_dict['iters']= iters
            metrics_dict['outer_loss']= loss.item()
            # Evaluate on validation data and check the stopping condition
            if (iters % 10 == 0):
                print(metrics_dict)
            self.log(metrics_dict)
            iters += 1
          if (not (self.validation_data is None)) and (iters % self.args.eval_every_n == 0):
            val_dict = {'val_loss': self.evaluate(self.validation_data, self.outer_param, last_layer=u),
                        'val_iters': iters}
            self.log(val_dict)
          if (not (self.test_data is None)):
            test_dict= {'test_loss': self.evaluate(self.test_data, last_layer=u),
                        'test_iter': iters}

    def evaluate(self, data, outer_model=None, last_layer=None):
        """
        Evaluates the performance of the model on the given data.

        Parameters:
        - data (object): Data object containing outcome and treatment information.
        - outer_model (object): Outer model used for evaluation. Defaults to self.outer_model.
        - last_layer (tensor): Last layer weights for an additional transformation. Defaults to None.

        Returns:
        - float: The evaluation loss on the provided data.
        """
        if outer_model is None:
          outer_model = self.outer_model
        self.outer_model.eval()
        with torch.no_grad():
          Y = (torch.from_numpy(data.outcome)).to(self.device, dtype=torch.float)
          X = (torch.from_numpy(data.treatment)).to(self.device, dtype=torch.float)
          outer_NN_dic = tensor_to_state_dict(self.outer_model, self.outer_param, self.device)
          pred = torch.func.functional_call(self.outer_model, parameter_and_buffer_dicts=outer_NN_dic, args=X)
          if last_layer is None:
            loss = self.MSE(pred, Y)
          else:
            loss = self.MSE((pred @ last_layer[:-1] + last_layer[-1]), Y)
        return loss.item()











