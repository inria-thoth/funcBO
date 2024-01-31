import torch
from applications.IVRegression.dataset_networks_dsprites.twoSLS import *
from applications.IVRegression.dataset_networks_dsprites.generator import *
import time
from funcBO.utils import assign_device, get_dtype, tensor_to_state_dict, state_dict_to_tensor
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from funcBO.InnerSolution import InnerSolution

#from parametricBO.BGS_opt.core.utils import Functional
from parametricBO.BGS_opt.core.selection import make_selection



 
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


    def check_config(self):
        if 'momentum' in self.args.selection.optimizer:
            if self.args.selection.optimizer.momentum==0.:
                self.args.selection.optimizer.momentum=None


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
            self.validation_data = TrainDataSetTorch.from_numpy(validation_data, device=device, dtype=self.dtype)
        else:
            self.validation_data = validation_data








        # Neural networks for dsprites data
        inner_model, self.outer_model = build_net_for_dsprite(self.args.seed, method='sequential+linear')
        inner_model.to(device)
        self.inner_model = inner_model.model
        self.outer_model.to(device)
        self.linear_inner = inner_model.linear
        # The outer neural network parametrized by the outer variable

        self.lower_var = tuple(self.inner_model.parameters())
        self.upper_var = tuple(self.outer_model.parameters())

        #self.outer_param = torch.nn.parameter.Parameter(state_dict_to_tensor(self.outer_model, device))

        #self.inner_param = torch.nn.parameter.Parameter(state_dict_to_tensor(self.inner_model, device))

        # Optimizer that improves the outer variable
        all_outer_params = tuple(self.upper_var)
        if 'iterative_2nd_stage' in self.args:
            if self.args.iterative_2nd_stage:
                self.linear_outer = nn.Linear(self.args.linear.in_dim, self.args.linear.out_dim, bias =True)
                self.linear_outer.to(device)
                all_outer_params = all_outer_params + tuple(self.linear_outer.parameters())
                self.inner_model = inner_model 
                self.lower_var = tuple(inner_model.parameters())


        self.outer_optimizer = torch.optim.Adam(all_outer_params, 
                                        lr=self.args.outer_optimizer.outer_lr, 
                                        weight_decay=self.args.outer_optimizer.outer_wd)

        inner_scheduler = None
        inner_dual_scheduler = None
        outer_scheduler = None


        def functional_model(model, params,inputs):

            model.eval()
            params_dict = model.state_dict()
            for (name, param),new_param in zip(model.named_parameters(),params):
                params_dict[name] = new_param
            model_outputs = (torch.func.functional_call(model, 
                          parameter_and_buffer_dicts=params_dict, 
                          args=inputs,
                          strict=True))
            return model_outputs





        def functional_inner_model(lower_var,inputs):
            return functional_model(self.inner_model,lower_var,inputs)

        def functional_outer_model(outer_var, inputs):
            return functional_model(self.outer_model,outer_var,inputs)


        self.functional_inner_model = functional_inner_model
        self.functional_outer_model = functional_outer_model
        # Loss helper functions
        self.MSE = nn.MSELoss()

        def ridge_func(weight,lam):
            return lam * torch.norm(weight) ** 2

        reg_func = lambda weight: ridge_func(weight, self.lam_V)

        # Outer objective function
        def fo(lower_var,Z_outer, Y_outer, upper_var): 
            inner_data  = self.inner_solution.optimizer.inputs
            z,x,_ = projector(inner_data)
            treatment_1st_feature = self.functional_outer_model(upper_var,x)
            instrumental_1st_feature = self.functional_inner_model(lower_var,z)
            instrumental_2nd_feature = self.functional_inner_model(lower_var,Z_outer)



            res = fit_2sls(treatment_1st_feature, instrumental_1st_feature, 
                            instrumental_2nd_feature, Y_outer, 
                            self.Nlam_V, self.lam_u)

            return res["stage2_loss"]

        def fi(data, upper_var, lower_var):

            z,x,_ = projector(data) 
            instrumental_feature = self.functional_inner_model(lower_var,z)
            treatment_feature = self.functional_outer_model(upper_var,x)
            feature = augment_stage1_feature(instrumental_feature)
            weight = fit_linear(treatment_feature, feature, self.Nlam_V)
            pred = linear_reg_pred(feature, weight)
            loss = torch.norm((treatment_feature - pred)) ** 2 + ridge_func(weight,self.Nlam_V)
            return loss



        if 'iterative_2nd_stage' in self.args:
            if self.args.iterative_2nd_stage:      

                def fo(g_z_out, Y):

                    pred = self.linear_outer(g_z_out)
                    reg = torch.norm(self.linear_outer.weight)**2 + torch.norm(self.linear_outer.bias)**2
                    loss = torch.norm((Y - pred)) ** 2 + self.lam_u*reg
                    return loss

                # Inner objective function that depends only on the inner prediction
                def fi(data, upper_var,lower_var):#, backward_mode=False):
                    
                    z,x,_ = projector(data) 
                    inner_val = self.functional_inner_model(lower_var,z)
                    outer_val = self.functional_outer_model(upper_var,x)
                    reg = torch.stack([torch.sum(param**2) for param in lower_var],dim=0)
                    reg = torch.sum(reg)
                    loss = torch.norm((outer_val - inner_val)) ** 2 + self.Nlam_V*reg
                    return loss




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

        #self.lower_loss = Functional(self.lower_loss_module)
        #self.upper_loss = Functional(self.upper_loss_module)


        self.check_config()
        self.inner_solution = make_selection(self.inner_loss,
                                    self.lower_var,
                                    self.inner_dataloader,
                                    self.args.selection,
                                    self.device,
                                    self.dtype)


    def update_lower_var(self,opt_lower_var):
        for p,new_p in zip(self.lower_var,opt_lower_var):
            p.data.copy_(new_p.data)

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
                Z_outer = Z.to(self.device)
                Y_outer = Y.to(self.device)
                # Inner value corresponds to h*(Z)
                forward_start = time.time()
                # Get the value of h*(Z_outer)
                params = self.lower_var + self.upper_var
                opt_lower_var,lower_loss = self.inner_solution(*params)
                inner_val = self.functional_inner_model(opt_lower_var,Z_outer)
                inputs= opt_lower_var,Z_outer, Y_outer, self.upper_var
                if 'iterative_2nd_stage' in self.args:
                    if self.args.iterative_2nd_stage: 
                        inputs = opt_lower_var, Y_outer

                loss = self.outer_loss(*inputs)

                # Backpropagation
                self.outer_optimizer.zero_grad()
                backward_start = time.time()
                loss.backward()
                self.outer_optimizer.step()
                self.update_lower_var(opt_lower_var)
                metrics_dict['outer_loss'] = loss.item()
                if self.validation_data is not None:
                    metrics_dict['val_loss'] = (self.evaluate(data_type="validation")).item()
                inner_logs = self.inner_solution.optimizer.data_logs
                if inner_logs:
                    self.log_metrics_list(inner_logs, self.iters, log_name='inner_metrics')
                #dual_logs = self.inner_solution.dual_solver.data_logs
                #if dual_logs:
                #    self.log_metrics_list(dual_logs, self.iters, log_name='dual_metrics')
                self.log(metrics_dict)
                print(metrics_dict)
                self.iters += 1
                #print(self.outer_param.grad)
                done = (self.iters >= self.args.max_epochs)
                if done:
                    break


        if self.validation_data:
            val_log = [{'inner_iter': 0,
                        'val loss': (self.evaluate(data_type="validation")).item()}]
            self.log_metrics_list(val_log, 0, log_name='val_metrics')
        test_log = [{'inner_iter': 0,
                    'test loss': (self.evaluate(data_type="test")).item()}]
        self.log_metrics_list(test_log, 0, log_name='test_metrics')

    def evaluate(self, data_type="validation"):
        """
        Evaluates the performance of the model on the given data.
        Returns:
        - float: The evaluation loss on the provided data.
        """
        #outer_NN_dic = tensor_to_state_dict(self.outer_model, self.outer_param, self.device)
        previous_state_outer_model = self.outer_model.training
        previous_state_inner_solution = self.inner_solution.training
        previous_state_inner_model = self.inner_model.training
        self.outer_model.eval()
        self.inner_solution.eval()
        self.inner_model.eval()
        with torch.no_grad():
            for Z, X, Y in self.outer_dataloader:
                Z_outer = Z.to(self.device)
                Y_outer = Y.to(self.device)
            for Z, X, Y in self.inner_dataloader:
                Z_inner = Z.to(self.device)
                X_inner = X.to(self.device)
            treatment_1st_feature = self.outer_model(X_inner) #torch.func.functional_call(self.outer_model, parameter_and_buffer_dicts=outer_NN_dic, args=X_inner)
            
            instrumental_1st_feature = self.inner_model(Z_inner)
            instrumental_2nd_feature = self.inner_model(Z_outer)            

            if 'iterative_2nd_stage' in self.args:
                if self.args.iterative_2nd_stage:
                    instrumental_1st_feature = self.inner_model.model(Z_inner)
                    instrumental_2nd_feature = self.inner_model.model(Z_outer)

            feature = augment_stage1_feature(instrumental_1st_feature)
            stage1_weight = fit_linear(treatment_1st_feature, feature, self.Nlam_V)
            feature = augment_stage1_feature(instrumental_2nd_feature)
            predicted_treatment_feature = linear_reg_pred(feature, stage1_weight)
            feature = augment_stage2_feature(predicted_treatment_feature)
            stage2_weight = fit_linear(Y_outer, feature, self.lam_u)
            if data_type == "test":
                Y_eval = self.test_data.structural
                X_eval = self.test_data.treatment
            elif data_type == "validation":
                Y_eval = self.validation_data.outcome
                X_eval = self.validation_data.treatment
            treatment_feature = self.outer_model(X_eval) #torch.func.functional_call(self.outer_model, parameter_and_buffer_dicts=outer_NN_dic, args=X_eval)
            eval_feature = augment_stage2_feature(treatment_feature)
            eval_pred = linear_reg_pred(eval_feature, stage2_weight)
            loss = (torch.norm((Y_eval - eval_pred)) ** 2) / Y_eval.size(0)
        self.outer_model.train(previous_state_outer_model)
        self.inner_solution.train(previous_state_inner_solution)
        self.inner_model.train(previous_state_inner_model)
        return loss
