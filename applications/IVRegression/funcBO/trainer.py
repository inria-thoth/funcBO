import torch
from applications.IVRegression.dsprites_data.trainer import *
from applications.IVRegression.dsprites_data.generator import *
import time
from funcBO.utils import assign_device, get_dtype, state_dict_to_tensor, tensor_to_state_dict
from torch.utils.data import DataLoader
from torch.nn import MSELoss
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
        self.build_trainer()

    def log(self,dico, log_name='metrics'):
        self.logger.log_metrics(dico, log_name=log_name)

    def log_metrics_list(self, dico_list, iteration, prefix="", log_name='metrics'):
        total_iter = len(dico_list)
        for dico in dico_list:
            dico['outer_iter'] = iteration
            dico['total_iter'] = iteration*total_iter + dico['iter']
            dico = {prefix+key:value for key,value in dico.items()}
            self.log(dico, log_name=log_name)

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
        inner_data, outer_data = split_train_data(train_data, split_ratio=self.args.split_ratio, rand_seed=self.args.seed, device=device, dtype=self.dtype)
        test_data = TestDataSetTorch.from_numpy(test_data, device=device, dtype=self.dtype)

        # Scaling of the regularization parameters
        inner_data_size = inner_data[0].size(0)
        lam_V = self.args.lam_V
        lam_u = self.args.lam_u*outer_data[0].size(0)

        # Dataloaders for inner and outer data
        inner_data = DspritesTrainData(inner_data)
        self.inner_dataloader = DataLoader(dataset=inner_data, batch_size=self.args.batch_size, shuffle=False)
        outer_data = DspritesTrainData(outer_data)
        self.outer_dataloader = DataLoader(dataset=outer_data, batch_size=self.args.batch_size, shuffle=False)
        self.test_data = DspritesTestData(test_data)
        if validation_data is not None:
            self.validation_data = DspritesTrainData(validation_data)
        else:
            self.validation_data = validation_data

        # Neural networks for dsprites data
        self.inner_model, self.outer_model = build_net_for_dsprite(self.args.seed, method='sequential+linear')
        self.inner_model.to(device)
        self.outer_model.to(device)
        
        # The outer neural network parametrized by the outer variable
        self.outer_param = torch.nn.parameter.Parameter(state_dict_to_tensor(self.outer_model, device))

        # Optimizer that improves the outer variable
        self.outer_optimizer = torch.optim.Adam([self.outer_param], 
                                        lr=self.args.outer_optimizer.outer_lr, 
                                        weight_decay=self.args.outer_optimizer.outer_wd)

        inner_scheduler = None
        inner_dual_scheduler = None
        outer_scheduler = None

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
            loss = torch.norm((instrumental_feature - treatment_feature)) ** 2# + 0.07 * torch.norm(instrumental_feature) ** 2
            return loss

        def ridge_func(weight,lam):
            return lam* torch.norm(weight) ** 2

        reg_func = lambda weight: ridge_func(weight, lam_V)
        # Inner objective function with regularization

        def reg_objective(outer_param, instrumental_feature, X):
            Nlam_V= 0.5*lam_V*inner_data_size
            outer_NN_dic = tensor_to_state_dict(self.outer_model, 
                                                self.outer_param, 
                                                device)
            treatment_feature = (torch.func.functional_call(self.outer_model, 
                                parameter_and_buffer_dicts=outer_NN_dic, 
                                args=X, 
                                strict=True))
            # Get the value of g(Z)
            feature = augment_stage1_feature(instrumental_feature)
            weight = fit_linear(treatment_feature, feature, Nlam_V)
            pred = linear_reg_pred(feature, weight)
            loss = torch.norm((treatment_feature - pred)) ** 2 + ridge_func(weight,Nlam_V)
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

        if self.args.dual_solver['name']=='funcBO.solvers.IVClosedFormSolver':
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
            loss, u = self.outer_loss(inner_value, Y_outer)
            # Backpropagation
            self.outer_optimizer.zero_grad()
            backward_start = time.time()
            loss.backward()
            self.outer_optimizer.step()
            metrics_dict['outer_loss'] = loss.item()
            inner_logs = self.inner_solution.inner_solver.data_logs
            dual_logs = self.inner_solution.dual_solver.data_logs
            if inner_logs:
                self.log_metrics_list(inner_logs, iters, log_name='inner_metrics')
            if dual_logs:
                self.log_metrics_list(dual_logs, iters, log_name='dual_metrics')
            # Evaluate on validation data and check the stopping condition
            if (iters % 1 == 0):
                print(metrics_dict)
            self.log(metrics_dict)
            iters += 1
          if (self.validation_data is not None) and (iters % self.args.eval_every_n == 0):
            val_dict = {'val_loss': self.evaluate(self.validation_data, self.outer_param, last_layer=u),
                        'val_iters': iters}
          if (self.test_data is not None):
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
          Y = data.test_data.structural
          X = data.test_data.treatment
          outer_NN_dic = tensor_to_state_dict(self.outer_model, self.outer_param, self.device)
          pred = torch.func.functional_call(self.outer_model, parameter_and_buffer_dicts=outer_NN_dic, args=X)
          if last_layer is None:
            loss = self.MSE(pred, Y)
          else:
            loss = self.MSE((pred @ last_layer[:-1] + last_layer[-1]), Y)
        return loss.item()











