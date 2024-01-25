import  torch 
from torch.autograd import Variable
import torch.nn as nn
from torch import autograd
import time
import numpy as np
import omegaconf

import os

import importlib


class AttrList(object):
	### Adds list of objects as attributes to a module

	def __init__(self,module,attr_list,tag):
		self.module = module
		self.tag = tag
		for i,w in enumerate(attr_list):
			setattr(module,tag+str(i),w)
		self.num_attributes = len(attr_list)
		self.len = len(attr_list)
	def __getitem__(self,key):
		return getattr(self.module,self.tag+str(key))
	def __iter__(self):
		return iter(self.__getitem__(i) for i in range(self.num_attributes))

	def __len__(self):
		return self.len

# class Functional(object):
# 	def __init__(self,module):
# 		self.module= module
# 		module.train(True)
# 		self.func, self.weights, self.buffers = make_functional_with_buffers(module)
# 		module.train(False)
# 		self.eval_func, _,_ = make_functional_with_buffers(module)
# 	def eval(self,inputs,upper_var,lower_var,train_mode=True,**kwargs):
# 		params = upper_var + lower_var 
# 		if train_mode:
# 			return self.func(params,self.buffers,inputs,**kwargs)
# 		else:
# 			return self.eval_func(params,self.buffers,inputs,**kwargs)
# 	def __call__(self,inputs,upper_var,lower_var,train_mode=True,**kwargs):
# 		return self.eval(inputs,upper_var,lower_var,train_mode=train_mode,**kwargs)



class RingGenerator:
	def __init__(self, init_generator, device, dtype):
		self.init_generator = init_generator
		self.generator = None
		self.device= device
		self.dtype = dtype
	# def make_generator(self):
	# 	return iter(self.init_generator)

	def make_generator(self):
		return (set_device_and_type(data,self.device,self.dtype) for data in self.init_generator)
	def __next__(self, *args):
		try:
			return next(self.generator)
		except:
			self.generator = self.make_generator()
			return next(self.generator)
	def __iter__(self):
		return self.make_generator()

	def __getstate__(self):
		return {'init_generator': self.init_generator,
				'generator': None}
	def __setstate__(self, d ):
		self.init_generator = d['init_generator']
		self.generator = None


def detach_states(state_tuples):
	fields = state_tuples._fields
	key_values = {}
	for field in fields:
		state_tuple = eval('state_tuples.'+field)
		try:
			if type(state_tuple[0])==torch.Tensor:
				new_state_tuple = tuple([state.data for state in state_tuple])
				key_values[field] = new_state_tuple
		except:
			pass
	return state_tuples._replace(**key_values)
			



def grad_with_none(outputs,inputs, grad_outputs=None,retain_graph=False,create_graph=False, only_inputs=False,allow_unused=False):
	# Inspired from _autograd_grad in https://pytorch.org/docs/stable/_modules/torch/autograd/functional.html#vhp
	assert isinstance(outputs, tuple)
	if grad_outputs is None:
		grad_outputs = (None,) * len(outputs)
	assert isinstance(grad_outputs, tuple)
	assert len(outputs) == len(grad_outputs)
	new_outputs: Tuple[torch.Tensor, ...] = tuple()
	new_grad_outputs: Tuple[torch.Tensor, ...] = tuple()
	for out, grad_out in zip(outputs,grad_outputs):
		if out is not None and out.requires_grad:
			new_outputs +=(out,)
			new_grad_outputs +=(grad_out,)
	if len(new_outputs) == 0:
		return (None,) * len(inputs)
	else:
		return  autograd.grad(outputs=new_outputs, 
							  inputs=inputs, 
							  grad_outputs=new_grad_outputs, 
							  retain_graph=retain_graph, 
							  create_graph=create_graph, 
							  only_inputs=only_inputs, 
							  allow_unused=allow_unused) 

def set_device_and_type(data,device, dtype):
	if isinstance(data,list):
		data = tuple(data)
	if type(data) is tuple:
		data = tuple([ set_device_and_type(d,device,dtype) for d in data ])
		return data
	elif isinstance(data,torch.Tensor): 
		if dtype==torch.double:
			data = data.double()
		else:
			data = data.float()
		return  data.to(device)
	elif isinstance(data,int):
		return torch.tensor(data).to(device)
	else:
		raise NotImplementedError('unknown type')


def grad_lower(func,upper_var,lower_var,inputs,diff_params,retain_graph=True, create_graph=True):
	val = func(inputs,upper_var,lower_var)
		
	return autograd.grad(outputs=val, 
							inputs=diff_params, 
							grad_outputs=None, 
							retain_graph=retain_graph,
							create_graph=create_graph, 
							only_inputs=True,
							allow_unused=True)

def jvp(vector,param,dual_vect,retain_graph=True):
	return grad_with_none(outputs=vector, 
		inputs=param, 
		grad_outputs=dual_vect, 
		retain_graph=retain_graph,
		create_graph=False, 
		only_inputs=True,
		allow_unused=True)


def import_module(module_name):
    module, attr = os.path.splitext(module_name)
    try:
        module = importlib.import_module(module)
        return getattr(module, attr[1:])
    except:
        try:
            module = import_module(module)
            return getattr(module, attr[1:])
        except:
            return eval(module+attr[1:])

def config_to_instance(config_module_name="name",**config):
	module_name = config.pop(config_module_name)
	attr = import_module(module_name)
	if config:
		attr = attr(**config)
	return attr







