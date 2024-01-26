import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import helpers as hp
from core import utils 
import torchvision

# class ModelDataset(nn.Module):
# 	def __init__(self,shape,factor=1):
# 		super(ModelDataset,self).__init__()
# 		data = torch.distributions.normal.Normal(loc=0., scale=1.).sample(shape)
# 		self.x = torch.nn.parameter.Parameter(data)
# 		#self.x = torch.nn.parameter.Parameter(torch.zeros(shape))
# 		self.reg = torch.nn.parameter.Parameter(torch.zeros(shape[1:]).view(-1))
# 		num_label = shape[0]
# 		labels = torch.tensor(factor*list(range(num_label)))
# 		labels = torch.nn.functional.one_hot(labels, num_classes=num_label)
# 		self.y = torch.nn.parameter.Parameter(labels.float())
# 	def forward(self,input):
# 		return inputs

class ModelDataset(nn.Module):
	def __init__(self,shape,factor=1,x=None,y=None):
		super(ModelDataset,self).__init__()
		num_label = shape[0]
		shape[0] *= factor 
		
		if x is not None and y is not None :

			labels = torch.nn.functional.one_hot(y.long(), num_classes=num_label)
			if x.dtype ==torch.float64:
				 labels=labels.double()
			else:
				labels = labels.float()
			inv_sums = 1./torch.sum(labels,dim=0)
			mean_x = torch.einsum('b...,bc,c->c...',x,labels,inv_sums)
			data = mean_x.repeat_interleave(factor, dim=0)
		else:
			data = torch.distributions.normal.Normal(loc=0., scale=1.).sample(shape)
		self.x = torch.nn.parameter.Parameter(data)
		#self.x = torch.nn.parameter.Parameter(torch.zeros(shape))
		self.reg = torch.nn.parameter.Parameter(torch.zeros(shape[1:]).view(-1))
		labels = torch.tensor(list(range(num_label)))
		labels = labels.repeat_interleave(factor)
		
		self.y = torch.nn.parameter.Parameter(labels.float())
	def forward(self,input):
		return inputs



class Linear(nn.Module):
	def __init__(self,n_features,n_classes,with_bias=False):
		super(Linear,self).__init__()
		#data = torch.distributions.normal.Normal(loc=0., scale=1.).sample([n_features,n_classes])
		self.weight = torch.nn.parameter.Parameter(torch.zeros([n_features,n_classes]))
		if with_bias:
			self.bias = torch.nn.parameter.Parameter(torch.zeros([n_classes]))
		else:
			self. bias = 0.
	def forward(self,inputs):
		inputs = inputs.view(inputs.shape[0], -1)
		return inputs @ self.weight + self.bias


class MLP(nn.Module):
	def __init__(self,in_channel,out_channel,hidden_channels_dim=200):
		super(MLP,self).__init__()

		self.mlp = nn.Sequential(
			nn.Flatten(),
			nn.Linear(in_channel, hidden_channels_dim),
			nn.ReLU(),
			nn.Linear(hidden_channels_dim, hidden_channels_dim),
			nn.ReLU(),
			nn.Linear(hidden_channels_dim, hidden_channels_dim),
			nn.ReLU(),
			nn.Linear(hidden_channels_dim, hidden_channels_dim),
			nn.ReLU(),
			nn.Linear(hidden_channels_dim, out_channel))
	def forward(self,inputs):
		return self.mlp(inputs)



# class LogisticDistill(nn.Module):
# 	def __init__(self,upper_model,lower_model,is_lower=False, reg= False, device=None):
# 		super(LogisticDistill,self).__init__()

# 		self.device = device
# 		self.upper_model = upper_model
# 		self.lower_model = lower_model


# 		upper_var = list(self.upper_model.parameters())
# 		lower_var = list(self.lower_model.parameters())
# 		self.upper_var = utils.AttrList(self,upper_var,'upper_var_')
# 		self.lower_var = utils.AttrList(self,lower_var,'lower_var_')


# 		self.reg = reg
# 		#self.dim ,self.n_classes  = self.lower_var[0].shape()
# 		self.is_lower = is_lower
# 	def forward(self,data,with_acc=False):
		
# 		if self.is_lower:
# 			x, targets = self.upper_model.x, self.upper_model.y.data
# 			targets = torch.nn.functional.softmax(targets, dim=1)
# 			y = y.argmax(dim=1, keepdim=False)
# 		else:
# 			x,y = data
# 			y = y.long()
# 			targets = torch.nn.functional.one_hot(y, num_classes=self.upper_model.y.shape[0])
		
# 		out_x = self.lower_model(x)
# 		#out = F.cross_entropy(out_x, y)
# 		out = F.cross_entropy(out_x, targets)
# 		if self.reg:
# 			out =  out +  self.reg_term()

# 		if with_acc:
# 			pred = out_x.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
# 			#y = targets.argmax(dim=1, keepdim=True)
# 			acc = pred.eq(y.view_as(pred)).sum() / len(y)
# 			return out,acc
# 		else:
# 			return out

# 	def reg_term(self):
# 		ones_dxc = torch.ones(self.lower_var[0].size()).to(self.lower_var[0].device)
# 		out =  0.5*((self.lower_var[0]**2)*torch.exp(self.upper_model.reg.unsqueeze(1)*ones_dxc)).mean()
# #		if len(self.upper_var)>1:
# #			out = out+ (self.lower_var[0].abs() * torch.exp(self.upper_model.reg.unsqueeze(1) * ones_dxc)).mean()
# 		return out


















class LogisticDistill(nn.Module):
	def __init__(self,upper_model,lower_model,is_lower=False,is_linear=False, reg= 1., device=None):
		super(LogisticDistill,self).__init__()

		self.device = device
		self.upper_model = upper_model
		self.lower_model = lower_model


		upper_var = list(self.upper_model.parameters())
		lower_var = list(self.lower_model.parameters())
		self.upper_var = utils.AttrList(self,upper_var,'upper_var_')
		self.lower_var = utils.AttrList(self,lower_var,'lower_var_')


		self.reg = reg
		self.is_lower = is_lower
		self.is_linear = is_linear
	def forward(self,data,with_acc=False):
		
		if self.is_lower:
			x, y = self.upper_model.x, self.upper_model.y.data
			#targets = torch.nn.functional.softmax(y, dim=1)
			#y = y.argmax(dim=1, keepdim=False)
		else:
			x,y = data
			
			#targets = torch.nn.functional.one_hot(y, num_classes=self.upper_model.y.shape[0]).float()
		#x,y = data
		y = y.long()
		out_x = self.lower_model(x)
		out = F.cross_entropy(out_x, y)
		#out = F.cross_entropy(out_x, targets)

		if self.reg>0.:
			if self.is_linear:
				out =  out +  self.reg*self.reg_term_linear()
			else:
				out =  out +  self.reg*self.reg_term()
		if with_acc:
			pred = out_x.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
			acc = pred.eq(y.view_as(pred)).sum() / len(y)
			return out,acc
		else:
			return out

	def reg_term_linear(self):
		ones_dxc = torch.ones(self.lower_var[0].size()).to(self.lower_var[0].device)
		out =  0.5*((self.lower_var[0]**2)*torch.exp(self.upper_model.reg.unsqueeze(1)*ones_dxc)).mean()
#		if len(self.upper_var)>1:
#			out = out+ (self.lower_var[0].abs() * torch.exp(self.upper_model.reg.unsqueeze(1) * ones_dxc)).mean()
		return out


	def reg_term(self):
		ones_dxc = torch.ones(self.lower_var[0].size()).to(self.lower_var[0].device)
		out = 0.
		for var in self.lower_var:
			out = out + torch.sum(var**2)
		return out





