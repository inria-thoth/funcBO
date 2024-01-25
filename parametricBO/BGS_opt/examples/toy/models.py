import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import helpers as hp
from core import utils 
import numpy as np
class Identity(nn.Module):
	def __init__(self,dim):
		super(Identity,self).__init__()
		self.param_x = torch.nn.parameter.Parameter(.00000001*torch.ones([dim]))
	def forward(self,input):
		return inputs
# class QuadyLinx(nn.Module):
# 	def __init__(self,upper_model,lower_model,device=None,cond=.1, swap = False,params_system=None):
# 		super(QuadyLinx,self).__init__()

# 		self.device = device
# 		self.upper_model = upper_model
# 		self.lower_model = lower_model


# 		upper_var = list(self.upper_model.parameters())
# 		lower_var = list(self.lower_model.parameters())
# 		self.upper_var = utils.AttrList(self,upper_var,'upper_var_')
# 		self.lower_var = utils.AttrList(self,lower_var,'lower_var_')

# 		self.cond = cond 
# 		if swap:
# 			self.y = self.upper_var[0]
# 			self.x = self.lower_var[0]
# 		else:
# 			self.x = self.upper_var[0]
# 			self.y = self.lower_var[0]
# 		N = self.x.shape[0]
# 		d = self.y.shape[0]
# 		torch.manual_seed(0)

# 		self.AA = torch.randn(d,d, dtype=self.y.dtype, device=self.y.device)
# 		A = self.AA
# 		self.A = A@A.t()
# 		U,S,V = torch.svd(self.A)
# 		S[-1] = 0.

# 		lmbda = S[0]/(self.cond-1)
		
# 		#lmbda = max(0,lmbda)
# 		S = S+lmbda
# 		S = S/S[0]
# 		self.A = torch.einsum('ij,j,kj->ik',U,S,V)

# 		if swap:
# 			self.B = torch.zeros([d,N],dtype=self.y.dtype, device=self.y.device)
# 			self.C = torch.randn([N],dtype=self.y.dtype, device=self.y.device)
# 			if params_system is None: 
# 				self.x_star = None
# 				self.D = torch.zeros([1],dtype=self.y.dtype, device=self.y.device)
# 			else:
# 				self.A_y,self.B_y,_,_ = params_system
# 				v = self.C@torch.inverse(self.A_y)@self.B_y
# 				self.x_star = torch.inverse(self.A)@v
# 				self.D = 0.5*v.t()@self.x_star
# 		else:
# 			self.B = torch.randn([d,N],dtype=self.y.dtype, device=self.y.device)
# 			self.C = torch.zeros([N],dtype=self.y.dtype, device=self.y.device)
# 			self.D = torch.zeros([1],dtype=self.y.dtype, device=self.y.device)
# 			self.x_star = None


# 	def get_param_system(self):
# 		return self.A, self.B, self.C, self.D
# 	def forward(self,inputs):

# 		loss = 0.*torch.mean(inputs)  \
# 				+.5*self.y.t()@self.A@self.y \
# 				+ self.y.t()@self.B@self.x \
# 				+ self.C@self.x +self.D

# 		#loss = 0.*torch.mean(inputs) +.5*self.y.t()@self.A@self.y + self.y.t()@self.B@self.x + self.C@self.x +self.D

# 		return loss.mean()
# 	def func(self,data,x,y,with_acc):
		
# 		if self.x_star is None:
# 			loss = .5*y.t()@self.A@y \
# 					+ y.t()@self.B@x \
# 					+ self.C@x +self.D
# 		else:
# 			vv = y-self.x_star
# 			loss = 0.5*vv.T@self.A@vv
# 			#print('x: '+str(y) + ', x_star: '+ str(self.x_star))
# 		return loss,torch.sum(torch.zeros([0])) 

class QuadyLinx(nn.Module):
	def __init__(self,upper_model,lower_model,device=None,cond=.1,with_sin=False, swap = False,params_system=None):
		super(QuadyLinx,self).__init__()
		self.with_sin = with_sin
		self.device = device
		self.upper_model = upper_model
		self.lower_model = lower_model


		upper_var = list(self.upper_model.parameters())
		lower_var = list(self.lower_model.parameters())
		self.upper_var = utils.AttrList(self,upper_var,'upper_var_')
		self.lower_var = utils.AttrList(self,lower_var,'lower_var_')

		self.cond = cond 
		if swap:
			self.y = self.upper_var[0]
			self.x = self.lower_var[0]
		else:
			self.x = self.upper_var[0]
			self.y = self.lower_var[0]
		N = self.x.shape[0]
		d = self.y.shape[0]
		self.d = d
		torch.manual_seed(0)

		self.AA = torch.randn(d,d, dtype=self.y.dtype, device=self.y.device)
		A = self.AA
		self.A = A@A.t()
		U,S,V = torch.svd(self.A)
		S[-1] = 0.

		lmbda = S[0]/(self.cond-1)
		
		#lmbda = max(0,lmbda)
		S = S+lmbda
		S = S/S[0]
		if not swap:
			S[-100:-1] = 0.
		self.A = torch.einsum('ij,j,kj->ik',U,S,V)

		if swap:
			self.B = torch.zeros([d,N],dtype=self.y.dtype, device=self.y.device)
			self.C = torch.randn([N],dtype=self.y.dtype, device=self.y.device)
			if params_system is None: 
				self.x_star = None
				self.D = torch.zeros([1],dtype=self.y.dtype, device=self.y.device)
			else:
				self.A_y,self.B_y,_,_ = params_system
				v = self.C@torch.linalg.lstsq(self.A_y,self.B_y).solution
				self.x_star = torch.inverse(self.A)@v
				self.D = 0.5*v.t()@self.x_star
			self.Q = torch.zeros([N,N],dtype=self.y.dtype, device=self.y.device)
		else:
			self.U = torch.randn([d,N],dtype=self.y.dtype, device=self.y.device)
			self.B = self.A.t()@self.U
			self.C = torch.zeros([N],dtype=self.y.dtype, device=self.y.device)
			self.D = torch.zeros([1],dtype=self.y.dtype, device=self.y.device)
			self.U = torch.linalg.lstsq(self.A,self.B).solution
			self.Q = self.B.t()@self.U #+ 1.
			
			self.x_star = None


	def get_param_system(self):
		return self.A, self.B, self.C, self.D
	def forward(self,inputs):
		
		loss = 0.*torch.mean(inputs)  \
				+.5*self.y.t()@self.A@self.y \
				+ self.y.t()@self.B@self.x \
				+ self.C@self.x +self.D\
				+ 0.5*self.x.t()@self.Q@self.x 
		loss = loss.mean()
		if self.with_sin:
			#print(loss)
			loss =torch.sqrt((loss-100)**2+100.)
			#loss = torch.cos(10.*torch.exp(-1.*(loss-.2)))*loss
			#loss = torch.cos(10.*torch.exp(-1.*(loss-.2)))+loss
			#loss = -torch.sin(2*loss)+ loss
			# eps = 0.01
			# loss = torch.sqrt((loss-1.)**2+eps) \
			# 		-0.5*torch.sqrt((loss-5.)**2+eps) \
			# 		+ 0.5*torch.sqrt((loss-7.)**2+eps) \
			# 		+ 0.5*torch.sqrt((loss-8.)**2+eps)

		# else: 
		# 	vv = self.y-self.x_star
		# 	loss = 0.5*torch.sum(vv**2)
#			loss = 0.*torch.mean(inputs) + 0*torch.mean(self.y) -self.U@self.x #+ np.sqrt(np.pi/2)/np.sqrt(self.d)
			#print( torch.norm(self.U@self.x+self.y)**2 )
		#loss = 0.*torch.mean(inputs) +.5*self.y.t()@self.A@self.y + self.y.t()@self.B@self.x + self.C@self.x +self.D

		return loss 
	def func(self,data,x,y,with_acc):
		
		if self.x_star is None:
			loss = .5*y.t()@self.A@y \
					+ y.t()@self.B@x \
					+ self.C@x +self.D\
					+ 0.5*x.t()@self.Q@x
			if self.with_sin:
				loss = -torch.sin(loss)
		else:
			vv = y-self.x_star
			loss = 0.5*vv.T@self.A@vv
			#print('x: '+str(y) + ', x_star: '+ str(self.x_star))
		return loss,torch.sum(torch.zeros([0])) 

