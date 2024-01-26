import  torch 
import torch.nn as nn
from torch import autograd
from core.utils import grad_with_none,config_to_instance
import core.utils as utils
from core.utils import RingGenerator

import jax ## need to import jax before torchopt otherwise get an error
import TorchOpt
from itertools import cycle
import copy
from functools import partial
import torch.optim as optim
import os


def make_selection(func,
					init_lower_var,
					loader,
					options,
					device,
					dtype):

	generator = RingGenerator(loader, device, dtype)
	linear_solver = LinearSolver(func,generator,options.linear_solver)
	if isinstance(linear_solver.residual_op,FiniteDiffResidual):
		track_grad_for_backward=False
	else:
		track_grad_for_backward=True
	optimizer = DiffOpt(func,generator,
							init_lower_var,options.optimizer,
							options.warm_start_iter,options.unrolled_iter,
							track_grad_for_backward=track_grad_for_backward)
	use_scheduler = options.scheduler.pop("use_scheduler", None)
	if use_scheduler:
		dummy_opt = optim.SGD(init_lower_var, lr = optimizer.lr)
		scheduler = config_to_instance(**options.scheduler, optimizer = dummy_opt)
	else:
		scheduler = None
	dual_var_warm_start = options.dual_var_warm_start
	selection = Selection(func,
						  init_lower_var,
						  generator,
						  linear_solver,
						  optimizer,
						  scheduler,
						  correction = options.correction,
						  dual_var_warm_start=dual_var_warm_start,
						  track_grad_for_backward = track_grad_for_backward
						  )
	return selection

class Selection(nn.Module):
	def __init__(self,
				func,
				init_lower_var,
				generator,
				linear_solver,
				optimizer,
				scheduler,
				correction = True,
				dual_var_warm_start=True,
				track_grad_for_backward=False):
		super(Selection,self).__init__()
		self.func = func
		self.generator = generator
		
		self.lower_var = tuple(init_lower_var)
		self.linear_solver = linear_solver
		self.optimizer = optimizer
		self.scheduler = scheduler
		self.correction = correction
		self.track_grad_for_backward = track_grad_for_backward
		self.dual_var_warm_start = dual_var_warm_start
		if self.correction:
			self.dual_var = [torch.zeros_like(p) for p in init_lower_var]
		else:
			self.dual_var = None
	def update_lr(self):
		if self.scheduler is not None:
			self.scheduler.step()
			self.optimizer.update_lr(self.scheduler.get_last_lr()[0])
	
	def update_dual(self,dual_var):
		if self.dual_var_warm_start:
			self.dual_var = dual_var
			# norm = 0.
			# for i,var in enumerate(dual_var):
			# 	norm += torch.mean(var**2)
			# norm = norm/i 
			# print(norm)

	# def update_var(self,opt_lower_var):
	# 	for p,new_p in zip(self.lower_var,opt_lower_var):
	# 		p.data.copy_(new_p.data)

	def forward(self,*all_params):
		len_lower = len(self.lower_var)
		with  torch.enable_grad():
			opt_lower_var =  ArgMinOp.apply(self,len_lower,*all_params)
		
		return  opt_lower_var

class ArgMinOp(torch.autograd.Function):

	@staticmethod
	def forward(ctx,selection, len_lower,*all_params):
		lower_var = all_params[:len_lower]
		upper_var = all_params[len_lower:]
		ctx.selection = selection
		ctx.len_lower = len_lower
		with  torch.enable_grad():
			iterates, val,grad, inputs = selection.optimizer.run(upper_var,lower_var)
		ctx.iterates = iterates
		ctx.grad = grad
		ctx.inputs = inputs
		ctx.save_for_backward(*all_params)
		
		return tuple( p.detach() for p in iterates[-1])

	@staticmethod
	def backward(ctx, *grad_output):
		selection = ctx.selection
		iterates = ctx.iterates
		len_lower = ctx.len_lower
		upper_var = ctx.saved_tensors[len_lower:]
		lower_var = ctx.saved_tensors[:len_lower]
		grad = ctx.grad
		inputs = ctx.inputs
		with  torch.enable_grad():
			if len(iterates)>1:
				
				val = [ torch.einsum('...i,...i->',y,g) for y,g in zip(  iterates[-1], grad_output)]
				val = torch.sum(torch.stack(val))
				all_params = iterates[0]+upper_var
				len_lower = len(lower_var)
				grad_selection = autograd.grad(
									outputs=val, 
									inputs=all_params, 
									grad_outputs=None, 
									retain_graph=selection.track_grad_for_backward,
									create_graph=False, 
									only_inputs=True,
									allow_unused=True)
				grad_selection_lower =  grad_selection[:len_lower]
				grad_selection_upper =  grad_selection[len_lower:]
				del iterates
			else:
				grad_selection_lower =  grad_output
				grad_selection_upper =  tuple([torch.zeros_like(var) for var in upper_var])
		## Solve a system Ax+b=0, 
		# A: the hessian of the lower objective, 
		# b:   grad_selection_lower
		if selection.correction:
			correction, dual_var = selection.linear_solver.run(
								upper_var,
								lower_var,
								selection.dual_var,
								grad_selection_lower,grad,inputs)		
			
			## summing the contributions of partial_x phi and correction term
			for g_upper,g_lin in zip(grad_selection_upper,correction):
				g_upper.data.add_(g_lin)
			
			## update the dual variable for next iteration 
			selection.update_dual(dual_var)
		#print(hello)

		return (None,)*(len(lower_var)+2) + grad_selection_upper






class DiffOpt(object):
	def __init__(self,func,generator,params,config_dict, 
						warm_start_iter,unrolled_iter,
						track_grad_for_backward=False):
		self.func = func
		self.generator = generator

		#scheduler = config_to_instance(**config_dict.scheduler)
		self.lr = config_dict.pop("lr", None)
		self.config_opt = config_dict
		self.optimizer = config_to_instance(**self.config_opt, lr = self.lr)
		self.opt_state = self.optimizer.init(params)		
		self.unrolled_iter = unrolled_iter
		self.warm_start_iter = warm_start_iter
		self.track_grad_for_backward = track_grad_for_backward

		assert (self.warm_start_iter + self.unrolled_iter >0) 
	
	def update_lr(self,lr):
		self.optimizer = config_to_instance(**self.config_opt,lr=lr)
		
	def init_state_opt(self):
		# Detach all tensors to avoid backbropagating twice. 
		self.opt_state = tuple([utils.detach_states(state) for state in self.opt_state])


		
	def run(self,upper_var,lower_var):
		avg_val = 0.		
		cur_lower_var = lower_var
		all_lower_var = [cur_lower_var]
		track_grad = False
		total_iter = self.warm_start_iter + self.unrolled_iter
		
		self.init_state_opt()		

		for i in range(total_iter):
			inputs = next(self.generator)
			value = self.func(inputs,upper_var,cur_lower_var)
			if i>=self.warm_start_iter:
				track_grad = True

			if i == total_iter-1:
				all_grad = torch.autograd.grad(
											outputs=value, 
											inputs=upper_var+cur_lower_var, 
											retain_graph=self.track_grad_for_backward,
											create_graph=self.track_grad_for_backward,
											only_inputs=True,
											allow_unused=True)
				grad = all_grad[len(upper_var):]
			else:
				grad = torch.autograd.grad(
							outputs=value, 
							inputs=cur_lower_var, 
							retain_graph=track_grad,
							create_graph=track_grad,
							only_inputs=True,
							allow_unused=True)


			if  i<self.warm_start_iter:
				track_grad=False

			updates, self.opt_state = self.optimizer.update(grad, self.opt_state, inplace=not track_grad)
			cur_lower_var = TorchOpt.apply_updates(cur_lower_var, updates, inplace=not track_grad)
			
			if i>=self.warm_start_iter:
				all_lower_var.append(cur_lower_var)
			
			avg_val +=value.detach()

		avg_val = avg_val/total_iter

		return all_lower_var, avg_val,all_grad,inputs


class LinearSolver(object):
	def __init__(self,func,generator, config_dict):
		self.func = func
		self.generator = generator
		self.cur_generator = generator
		self.stochastic = config_dict.pop("stochastic", None)
		self.linear_solver_alg = config_to_instance(**config_dict.algorithm)
		self.residual_op = config_to_instance(**config_dict.residual_op, 
												hvp = self.hvp)
		
	def stochastic_mode(self):
		if self.stochastic:
			self.cur_generator = self.generator
		else:
			inputs = next(self.generator)
			self.cur_generator = cycle([inputs])

	def jvp(self,upper_var, lower_var,retain_graph=True, create_graph=True):

		inputs = next(self.cur_generator)
		val = self.func(inputs,upper_var,lower_var)
		
		jac = autograd.grad(outputs=val, 
							inputs=lower_var, 
							grad_outputs=None, 
							retain_graph=retain_graph,
							create_graph=create_graph, 
							only_inputs=True,
							allow_unused=True)
		return jac
	def hvp(self,jac, diff_params, iterate, retain_graph=True):
		# if diff_params is None:
		# 	diff_params=lower_var 
		#jac = self.jvp(upper_var,lower_var)
		vhp = utils.grad_with_none(outputs=jac, 
			inputs=diff_params, 
			grad_outputs=tuple(iterate), 
			retain_graph=retain_graph,
			create_graph=False, 
			only_inputs=True,
			allow_unused=True)
		# if with_jac:
		# 	return vhp,jac
		return vhp
		
	def run(self,
			upper_var,lower_var,
			init,
			b_vector,jac,inputs):
		self.stochastic_mode()
		#res_op = partialmethod(self.residual_op.eval,upper_var=upper_var,lower_var=lower_var, b_vector=b_vector)
		with  torch.enable_grad():
			#jac = self.jvp(upper_var, lower_var)
			if isinstance(self.residual_op,FiniteDiffResidual):
				def res_op(iterate):
					return self.residual_op.eval(jac,self.func,inputs, upper_var,lower_var,b_vector, iterate)
			else:
				def res_op(iterate):
					return self.residual_op.eval(jac, upper_var,lower_var,b_vector, iterate)
			sol,out = self.linear_solver_alg(res_op,init)
		
		return out, sol

class ResidualOp(object):
	def __init__(self,hvp):
		self.hvp = hvp
	def eval(self,jac, upper_var,lower_var,b_vector, iterate):
		raise NotImplementedError


class FiniteDiffResidual(ResidualOp):
	def __init__(self,hvp,epsilon=0.01):
		super(FiniteDiffResidual,self).__init__(hvp)	
		self.epsilon = epsilon
	def eval(self,jac,func, inputs,upper_var,lower_var,b_vector, iterate):
		params = upper_var+lower_var
		norm = torch.cat([w.view(-1) for w in iterate]).norm()
		eps = self.epsilon / (norm.detach()+self.epsilon)

		## y + epsilon d
		lower_var_plus = tuple([p+eps* d if d is not None else p  for p,d in zip(lower_var,iterate)])

		val_plus =func(inputs,upper_var,lower_var_plus)
		grad_plus = torch.autograd.grad(
									outputs=val_plus, 
									inputs=params, 
									retain_graph=False,
									create_graph=False,
									only_inputs=True,
									allow_unused=True)

		# ## y - epsilon d


		grad_minus = jac
		hvp  = [(p-m)/(eps) for p,m in zip(grad_plus,grad_minus)]
		residual = hvp[len(upper_var):]
		out = hvp[:len(upper_var)]
		residual= [g+b if g is not None else b for g,b in zip(residual,b_vector)]
		return residual, out


class Residual(ResidualOp):
	# residual of the form res= Ax+b.
	def __init__(self,hvp):
		super(Residual,self).__init__(hvp)
	def eval(self,jac, upper_var,lower_var,b_vector, iterate):
		params = upper_var+lower_var
		lower_jac = jac[len(upper_var):]

		hvp  = self.hvp(lower_jac, params,iterate)
		residual = hvp[len(upper_var):]
		
		out = hvp[:len(upper_var)]
		residual= [g+b if g is not None else b for g,b in zip(residual,b_vector)]
		
		return residual, out

class NormalResidual(ResidualOp):
	# residual of the form res= A(Ax+b).
	def __init__(self,hvp):
		super(NormalResidual,self).__init__(hvp)
	def eval(self,jac,upper_var, lower_var,b_vector, iterate):
		params = upper_var+lower_var
		lower_jac = jac[len(upper_var):]
		hvp = self.hvp(lower_jac, params,iterate,retain_graph=True)
		residual = hvp[len(upper_var):]
		out = hvp[:len(upper_var)]
		residual = tuple([g+b if g is not None else b for g,b in zip(residual,b_vector)])
		residual = utils.grad_with_none(outputs=lower_jac, 
			inputs=lower_var, 
			grad_outputs=residual, 
			retain_graph=True,
			create_graph=False, 
			only_inputs=True,
			allow_unused=True)
		return residual,out

