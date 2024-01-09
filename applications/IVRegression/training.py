import torch
#import wandb

from torch.utils.data import DataLoader
from torch.nn import MSELoss
import time

from funcBO.InnerSolution import InnerSolution

from datasets.dsprite.dspriteBilevel import *
from datasets.dsprite.trainer import *

from funcBO.utils import state_dict_to_tensor, tensor_to_state_dict

import os
os.environ['WANDB_DISABLED'] = 'true'

# Set seed
#seed = 42#set_seed()


def assign_device(device):
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
    def __init__(self,config,logger):
        self.logger = logger
        self.args = config
        self.device = assign_device(self.args.system.device)
        self.dtype = get_dtype(self.args.system.dtype)
        self.build_trainer() 

    def log(self,dico):
        self.logger.log_metrics(dico, log_name="metrics")
        #wandb.log(dico)


    def build_trainer(self):


        # Setting hyper-parameters
        # Method for computing a*() : "closed_form_a", "GD", "GDinH", "closed_form_DFIV"
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
        self.inner_model, self.inner_dual_model, self.outer_model = build_net_for_dsprite(self.args.seed)
        self.inner_model.to(device)
        self.inner_dual_model.to(device)
        self.outer_model.to(device)
        print("First inner layer:", self.inner_model.layer1.weight.data)
        print("First outer layer:", self.outer_model.layer1.weight.data)

        # Optimizer that improves the approximation of h*
        self.inner_optimizer = torch.optim.Adam(self.inner_model.parameters(), 
                                            lr=self.args.optimizer.inner_lr, 
                                            weight_decay=self.args.optimizer.inner_wd)


        # Optimizer that improves the approximation of a*
        self.inner_dual_optimizer = torch.optim.Adam(self.inner_dual_model.parameters(), 
                                    lr=self.args.optimizer.inner_dual_lr, 
                                    weight_decay=self.args.optimizer.inner_dual_wd)


        # The outer neural network parametrized by the outer variable
        self.outer_param = state_dict_to_tensor(self.outer_model, device)#torch.cat((torch.rand(u_dim).to(device), state_dict_to_tensor(outer_model, device)), 0)


        # Optimizer that improves the outer variable
        self.outer_optimizer = torch.optim.Adam([self.outer_param], 
                                        lr=self.args.optimizer.outer_lr, 
                                        weight_decay=self.args.optimizer.outer_wd)

        inner_scheduler = None
        inner_dual_scheduler = None
        outer_scheduler = None

        # Print configuration
        run_name = make_run_name(self.args.optimizer.inner_lr,
                                  self.args.optimizer.inner_dual_lr,
                                  self.args.optimizer.outer_lr,
                                  self.args.optimizer.inner_wd,
                                  self.args.optimizer.inner_dual_wd,
                                  self.args.optimizer.outer_wd,
                                  self.args.max_inner_dual_epochs,
                                  self.args.max_inner_epochs,
                                  self.args.seed,
                                  self.args.lam_u,
                                  self.args.lam_V,
                                  self.args.batch_size,
                                  self.args.max_epochs)

        print("Run configuration:", run_name)

        # Set logging
        #wandb.init(group="Dsprites_bilevelIV_a=Wh_test", name=run_name)

        # Gather all models
        inner_models = (self.inner_model, self.inner_optimizer, inner_scheduler, self.inner_dual_model, self.inner_dual_optimizer, inner_dual_scheduler)

        # Loss helper functions
        self.MSE = nn.MSELoss()

        # Outer objective function
        def fo(outer_param, g_z_out, Y):
            # Get the value of g(Z) inner
            instrumental_1st_feature = self.inner_model(inner_data.instrumental).detach()
            # Get the value of g(Z) outer
            instrumental_2nd_feature = g_z_out
            # Get the value of f(X) inner
            outer_NN_dic = tensor_to_state_dict(self.outer_model, outer_param, device)
            treatment_1st_feature = torch.func.functional_call(self.outer_model, 
                                    parameter_and_buffer_dicts=outer_NN_dic, 
                                    args=inner_data.treatment, 
                                    strict=True)
            #print("before call instrumental_net first layer norm:", torch.norm(inner_model.layer1.weight))
            res = fit_2sls(treatment_1st_feature, 
                            instrumental_1st_feature, 
                            instrumental_2nd_feature, 
                            Y, 
                            lam_V, 
                            lam_u)
            return res["stage2_loss"], res["stage2_weight"]

        # Inner objective function
        def fi(outer_param, g_z_in, X):
            # Get the value of f(X) outer
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
            loss = linear_reg_loss(treatment_feature, 
                                   feature, 
                                   lam_V)
            return loss







        self.inner_loss = fi
        self.outer_loss = fo

        self.inner_solution = InnerSolution(self.inner_loss, 
                                            self.inner_dataloader, 
                                            inner_models, 
                                            self.device, 
                                            self.args.batch_size, 
                                            max_epochs=self.args.max_inner_epochs, 
                                            max_dual_epochs=self.args.max_inner_dual_epochs, 
                                            args=[lam_u, lam_V, self.args.a_star_method], 
                                            outer_model=self.outer_model)






        # Optimize using neural implicit differention
        
    def train(self):
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
            self.outer_param.requires_grad = True
            forward_start = time.time()
            # Here is the only place where we need to optimize the inner solution
            self.outer_model.train(True)
            self.inner_solution.optimize_inner = True
            # Get the value of h*(Z_outer)
            inner_value = self.inner_solution.forward(self.outer_param, Z_outer, Y_outer)
            # Make sure that the inner solution is not optimized
            self.inner_solution.optimize_inner = False
            self.inner_solution.model.train(False)
            self.inner_solution.dual_model.train(False)
            metrics_dict['forward_time']= time.time() - forward_start
            loss, u = self.outer_loss(self.outer_param, inner_value, Y_outer)
            # For checking the computational <autograd> graph.

            metrics_dict['dual_loss'] =self.inner_solution.dual_loss
            metrics_dict['norm_outer_param']= self.inner_solution.dual_loss
            metrics_dict['norm_grad_outer_param']= torch.norm(self.outer_param.grad).item()
            # Backpropagation
            self.outer_optimizer.zero_grad()
            backward_start = time.time()
            loss.backward()
            self.outer_optimizer.step()
            metrics_dict['forward_time']= time.time() - backward_start
            metrics_dict['iter_time']= time.time() - start
            metrics_dict['iters']= iters
            # Update loss and iteration count
            # Inner losses
            metrics_dict['inner_loss']= self.inner_solution.loss
            metrics_dict['outer_loss']= loss.item()
            # Evaluate on validation data and check the stopping condition
            if (iters % 10==0):
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
        Evaluate the prediction quality on the test dataset.
          param data: data to evaluate on
          param outer_param: outer variable
        """
        if outer_model is None:
          outer_model = self.outer_model
        self.outer_model.train(False)
        self.inner_solution.model.train(False)
        self.inner_solution.dual_model.train(False)
        with torch.no_grad():
          Y = (torch.from_numpy(data.outcome)).to(self.device, dtype=torch.float)
          X = (torch.from_numpy(data.treatment)).to(self.device, dtype=torch.float)
          outer_NN_dic = tensor_to_state_dict(self.outer_model, self.outer_param, self.device)
          # Get the value of f(X)
          pred = torch.func.functional_call(self.outer_model, parameter_and_buffer_dicts=outer_NN_dic, args=X)
          if last_layer is None:
            loss = self.MSE(pred, Y)
          else:
            loss = self.MSE((pred @ last_layer[:-1] + last_layer[-1]), Y)
        return loss.item()












