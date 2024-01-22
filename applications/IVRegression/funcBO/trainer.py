import torch
from applications.IVRegression.dsprites_data.trainer import *
from applications.IVRegression.dsprites_data.generator import *
import time
from funcBO.utils import assign_device, get_dtype, tensor_to_state_dict, state_dict_to_tensor
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from funcBO.InnerSolution import InnerSolution

class Trainer:
    """
    Solves an instrumental regression problem using the bilevel functional method.
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
        self.NNs_with_norms = NNs_with_norms
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
        test_data = generate_test_dsprite(device=device)
        train_data, validation_data = generate_train_dsprite(data_size=self.args.data_size,
                                                            rand_seed=self.args.seed,
                                                            val_size=self.args.val_size)
        inner_data, outer_data = split_train_data(train_data, split_ratio=self.args.split_ratio, rand_seed=self.args.seed, device=device, dtype=self.dtype)
        self.test_data = TestDataSetTorch.from_numpy(test_data, device=device, dtype=self.dtype)

        # Scaling of the regularization parameters
        inner_data_size = inner_data[0].size(0)
        self.lam_V = self.args.lam_V
        self.lam_u = self.args.lam_u*outer_data[0].size(0)
        # Scaled reg. param. (following the DFIV setting)
        self.Nlam_V= 0.5*self.lam_V*inner_data_size

        # Dataloaders for inner and outer data
        inner_data = DspritesTrainData(inner_data)
        self.inner_dataloader = DataLoader(dataset=inner_data, batch_size=self.args.batch_size, shuffle=False)
        outer_data = DspritesTrainData(outer_data)
        self.outer_dataloader = DataLoader(dataset=outer_data, batch_size=self.args.batch_size, shuffle=False)
        if validation_data is not None:
            self.validation_data = TrainDataSetTorch.from_numpy(validation_data, device=device)
        else:
            self.validation_data = validation_data

        # Neural networks for dsprites data
        #if self.NNs_with_norms:
        self.inner_model, self.outer_model = build_net_for_dsprite_with_norms(self.args.seed, method='sequential+linear')
        #else:
        #    self.inner_model, self.outer_model = build_net_for_dsprite(self.args.seed, method='sequential+linear')
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

        def ridge_func(weight,lam):
            return lam * torch.norm(weight) ** 2

        reg_func = lambda weight: ridge_func(weight, self.lam_V)

        # Outer objective function
        def fo(g_z_out, Y):
            self.outer_model.train(True)
            res = fit_2n_stage(
                            g_z_out, 
                            Y,  
                            self.lam_u)
            return res["stage2_loss"]

        # Inner objective function that depends only on the inner prediction
        def fi(outer_param, instrumental_feature, X):#, backward_mode=False):
            #if backward_mode:
            #    self.outer_model.train(True)
            #else:
            #    self.outer_model.train(False)
            outer_NN_dic = tensor_to_state_dict(self.outer_model, 
                                                outer_param, 
                                                device)
            treatment_feature = (torch.func.functional_call(self.outer_model, 
                                parameter_and_buffer_dicts=outer_NN_dic, 
                                args=X,
                                strict=True))
            # Fix reg. here, extract weight from the inner NN last linear layer
            loss = torch.norm((instrumental_feature - treatment_feature)) ** 2 + ridge_func(self.inner_model.linear.weight, self.Nlam_V)
            return loss

        # Inner objective function with regularization
        def reg_objective(outer_param, instrumental_feature, X, backward_mode=False):
            if backward_mode:
                self.outer_model.train(True)
            else:
                self.outer_model.train(False)
            outer_NN_dic = tensor_to_state_dict(self.outer_model,
                                                    outer_param,
                                                    device)
            treatment_feature = (torch.func.functional_call(self.outer_model, 
                                parameter_and_buffer_dicts=outer_NN_dic, 
                                args=X, 
                                strict=True))
            # Get the value of g(Z)
            feature = augment_stage1_feature(instrumental_feature)
            weight = fit_linear(treatment_feature, feature, self.Nlam_V)
            pred = linear_reg_pred(feature, weight)
            loss = torch.norm((treatment_feature - pred)) ** 2 + ridge_func(weight,self.Nlam_V)
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

        if self.args.dual_solver['name']=='funcBO.solvers.CompositeSolver':
            self.args.dual_solver['reg_objective'] = reg_objective

        if self.args.dual_solver['name']=='funcBO.solvers.ClosedFormSolver':
            self.args.dual_solver['reg'] = self.lam_V

        if self.args.dual_solver['name']=='funcBO.solvers.IVClosedFormSolver':
            self.args.dual_solver['reg'] = self.lam_V
        
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
            for Z, X, Y in self.outer_dataloader:
                metrics_dict = {}
                metrics_dict['iter'] = self.iters
                start = time.time()
                # Move data to GPU
                Z_outer = Z.to(self.device, dtype=torch.float)
                Y_outer = Y.to(self.device, dtype=torch.float)
                # Inner value corresponds to h*(Z)
                forward_start = time.time()
                # Get the value of h*(Z_outer)
                inner_value = self.inner_solution(Z_outer)
                loss = self.outer_loss(inner_value, Y_outer)
                # Backpropagation
                self.outer_optimizer.zero_grad()
                backward_start = time.time()
                loss.backward()
                self.outer_optimizer.step()
                metrics_dict['outer_loss'] = loss.item()
                if self.validation_data is not None:
                    metrics_dict['val_loss'] = (self.evaluate(data_type="validation")).item()
                inner_logs = self.inner_solution.inner_solver.data_logs
                if inner_logs:
                    self.log_metrics_list(inner_logs, self.iters, log_name='inner_metrics')
                dual_logs = self.inner_solution.dual_solver.data_logs
                if dual_logs:
                    self.log_metrics_list(dual_logs, self.iters, log_name='dual_metrics')
                self.log(metrics_dict)
                print(metrics_dict)
                self.iters += 1
                done = (self.iters >= self.args.max_epochs)
                if done:
                    break
        test_log = [{'inner_iter': 0,
                    'test loss': (self.evaluate(data_type="test")).item()}]
        self.log_metrics_list(test_log, 0, log_name='test_metrics')

    def evaluate(self, data_type="validation"):
        """
        Evaluates the performance of the model on the given data.

        Parameters:
        - data (object): Data object containing outcome and treatment information.
        - last_layer (tensor): Last layer weights for an additional transformation. Defaults to None.

        Returns:
        - float: The evaluation loss on the provided data.
        """
        outer_NN_dic = tensor_to_state_dict(self.outer_model, self.outer_param, self.device)
        previous_state_outer_model = self.outer_model.training
        previous_state_inner_solution = self.inner_solution.training
        previous_state_inner_model = self.inner_model.training
        self.outer_model.eval()
        self.inner_solution.eval()
        self.inner_model.eval()
        with torch.no_grad():
            if data_type == "test":
                for Z, X, Y in self.outer_dataloader:
                    Z_outer = Z.to(self.device, dtype=torch.float)
                    Y_outer = Y.to(self.device, dtype=torch.float)
                for Z, X, Y in self.inner_dataloader:
                    Z_inner = Z.to(self.device, dtype=torch.float)
                    X_inner = X.to(self.device, dtype=torch.float)
                treatment_1st_feature = torch.func.functional_call(self.outer_model, parameter_and_buffer_dicts=outer_NN_dic, args=X_inner)
                instrumental_1st_feature = self.inner_model.model(Z_inner)
                instrumental_2nd_feature = self.inner_model.model(Z_outer)
                feature = augment_stage1_feature(instrumental_1st_feature)
                stage1_weight = fit_linear(treatment_1st_feature, feature, self.Nlam_V)
                feature = augment_stage1_feature(instrumental_2nd_feature)
                predicted_treatment_feature = linear_reg_pred(feature, stage1_weight)
                feature = augment_stage2_feature(predicted_treatment_feature)
                stage2_weight = fit_linear(Y_outer, feature, self.lam_u)
                Y_test = self.test_data.structural
                X_test = self.test_data.treatment
                treatment_feature = torch.func.functional_call(self.outer_model, parameter_and_buffer_dicts=outer_NN_dic, args=X_test)
                test_feature = augment_stage2_feature(treatment_feature)
                test_pred = linear_reg_pred(test_feature, stage2_weight)
                loss = (torch.norm((Y_test - test_pred)) ** 2) / Y_test.size(0)

                #mdl.fit_t(train_1st_t, train_2nd_t, self.lam1, self.lam2)
                #oos_loss: float = mdl.evaluate_t(test_data_t).data.item()
                #loss = self.MSE((pred @ last_layer[:-1] + last_layer[-1]), Y)
            #elif data_type=="validation":
                #Y = self.validation_data.outcome
                #Z = self.validation_data.instrumental
                #stage2_feature = self.inner_solution(Z)
            #feature = augment_stage2_feature(stage2_feature)
            #pred = linear_reg_pred(feature, last_layer)
            #loss = self.MSE(pred, Y)
        #self.outer_model.train()
        #self.inner_solution.train()
        self.outer_model.train(previous_state_outer_model)
        self.inner_solution.train(previous_state_inner_solution)
        self.inner_model.train(previous_state_inner_model)
        return loss











