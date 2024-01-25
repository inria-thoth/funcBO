import torch
import torch.nn as nn
import examples.multitask.autolambda.create_network as networks 
import torch.nn.functional as F
from utils import helpers as hp
from core import utils 

class Identity(nn.Module):
	def __init__(self,dim, init):
		super(Identity,self).__init__()
		self.param_x = torch.nn.parameter.Parameter(init*torch.ones([dim]))
	def forward(self,inputs):
		return self.param_x + 0.*torch.sum(inputs)


class MutiTaskLoss(nn.Module):
	def __init__(self,upper_model,lower_model,num_tasks=1,weighted=True,reg=0., apply_reg=False, device=None):
		super(MutiTaskLoss,self).__init__()
		self.upper_model = upper_model
		self.lower_model = lower_model
		upper_var = list(self.upper_model.parameters())
		lower_var = list(self.lower_model.parameters())
		self.upper_var = utils.AttrList(self,upper_var,'upper_var_')
		self.lower_var = utils.AttrList(self,lower_var,'lower_var_')
		self.device = device
		self.weighted = weighted
		self.reg= reg
		self.apply_reg= apply_reg
		self.num_tasks = num_tasks
	def format_data(self,data):
		tasks = torch.cat([d[1] for d in data],dim=0).long()
		tasks_onehot = torch.nn.functional.one_hot(tasks, num_classes=self.num_tasks)
		tasks_onehot  = hp.to_type(tasks_onehot,data[0][0][0].dtype)
		all_x = torch.cat([d[0][0] for d in data],dim=0)
		all_y = torch.cat([d[0][1].long() for d in data],dim=0)
		return all_x, all_y,tasks_onehot,tasks

	def forward(self,data,with_acc=False, all_losses=False):
		all_x, all_y,tasks_onehot,tasks = self.format_data(data)

		preds = self.lower_model(all_x,tasks_onehot,tasks)
		losses =  F.cross_entropy(preds, all_y, ignore_index=-1,reduction='none')

		sum_tasks = torch.sum(tasks_onehot,dim=0)
		inv_sum_tasks = 1./sum_tasks
		inv_sum_tasks[sum_tasks==0] = 0
		losses = torch.einsum('i,ik->k',losses,tasks_onehot)*inv_sum_tasks
		
		if self.weighted:
			loss = torch.einsum('i,i->',losses,self.upper_var[0])
		else:
			loss = torch.sum(losses)
		if with_acc:
			all_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
			all_acc = 1.*all_preds.eq(all_y.view_as(all_preds))
			acc = torch.mean(all_acc)
			all_acc = torch.einsum('id,ik->k', all_acc, tasks_onehot)*inv_sum_tasks
			
		if self.apply_reg:
			loss = loss + self.reg_lower()

		if all_losses:
			if with_acc:
				return loss,losses,acc,all_acc
			else:
				return loss,losses
		else:
			if with_acc:
				return loss,acc
			else:
				return loss

	def reg_outer(self):
		return 0.5*self.reg*torch.sum(self.upper_var[0]**2)

	def reg_lower(self):
		l2_penalty = 0.
		
		for i,p in enumerate(self.lower_var):
			l2_penalty += torch.sum(p**2)
		return 0.5*self.reg*l2_penalty
				
