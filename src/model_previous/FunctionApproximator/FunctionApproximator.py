import torch
import random
from torch import nn, tensor
from torch.utils.data import Dataset, DataLoader


class FunctionApproximator():
	"""
	Approximates one of the functions h*(X) or a*(X) used in Neural Implicit Differentiation.
	"""

	def __init__(self, layer_sizes, loss_G=None, batch_size=64, function='h'):
		"""
		Init method.
			param layer_sizes: sizes of the layers of the network used to approximate functions h*(X) and a*(X)
			param loss_G: inner loss of the bilevel problem solved using Neur. Impl. Diff.
			param batch_size: size of training batches
			param function: specifies which of the two functions h*(X) or a*(X) is approximated
		"""
		self.loss_G = loss_G
		self.batch_size = batch_size
		# Setting the device to GPUs if available.
		if torch.cuda.is_available():
			dev = "cpu"#"cuda"
			print("All good, switching to GPUs.")
		else:
			dev = "cpu"
			print("No GPUs found, setting the device to CPU.")
		self.device = torch.device(dev)
		if layer_sizes is None:
			raise AttributeError("You must specify the layer sizes for the network.")
		if len(layer_sizes) != 5:
			raise ValueError("Networks have five layers, you must give a list with five integer values.")
		if function == 'h':
			if loss_G is None:
				raise AttributeError("You must specify the inner objective loss G.")
			else:
				self.NN = (NeuralNetwork_h(layer_sizes, self.device)).to(self.device)
		elif function == 'a':
			self.NN = (NeuralNetwork_a(layer_sizes, self.device)).to(self.device)
		else:
			raise AttributeError("You must specify the function that the network will approximate.")

	def load_data(self, X_inner, y_inner, X_outer=None, y_outer=None):
		"""
		Loads data into a type suitable for batch training.
			param X_inner: data of the inner objective
			param y_inner: labels of the inner objective
			param X_outer: data of the outer objective
			param y_outer: labels of the outer objective
		"""
		self.X_inner = X_inner
		self.y_inner = y_inner
		self.inner_data = Data(X_inner, y_inner)
		self.inner_dataloader = DataLoader(dataset=self.inner_data, batch_size=self.batch_size, shuffle=True)
		self.X_outer = X_outer
		self.y_outer = y_outer
		if not (self.X_outer is None and self.y_outer is None):
			self.outer_data = Data(X_outer, y_outer)
			self.outer_dataloader = DataLoader(dataset=self.outer_data, batch_size=self.batch_size, shuffle=True)

	def train(self, inner_grad22=None, outer_grad2=None, mu_k = None, h_k = None, num_epochs = 10, learning_rate = 0.001):
		"""
		Trains a neural network that approximates a function.
			param inner_grad22: hessian of the inner objective wrt h*(x)
			param outer_grad2: gradient of the outer objective wrt h*(x)
			param mu_k: current mu_k used when approximating h*(x)
			param h_k: current h*(x) used when approximating a*(x)
			param num_epochs: number of training epochs
			param learning_rate: learning rate for gradient descent
		"""
		optimizer = torch.optim.SGD(self.NN.parameters(), lr=learning_rate)
		loss_values = []
		# Approximating h_*(x)
		if not (mu_k is None and self.X_inner is None and self.y_inner is None) and (h_k is None and self.X_outer is None and self.y_outer is None):
			# Set the inner loss G with a fixed x as the objective function
			for epoch in range(num_epochs):
				for X_i, y_i in self.inner_dataloader:
					# Move to GPU
					X_i = X_i.to(self.device)
					y_i = y_i.to(self.device)
					mu_k = mu_k.to(self.device)
					# Zero all the parameter gradients
					optimizer.zero_grad()
					h_k = self.NN
					loss = self.loss_G(mu_k, h_k, X_i, y_i)
					loss_values.append(loss.float())
					loss.backward()
					optimizer.step()
			return self.NN, loss_values
		# Approximating a_*(x)
		elif not (mu_k is None and h_k is None and self.X_inner is None and self.y_inner is None and self.X_outer is None and self.y_outer is None):
			# Set the loss H with a fixed h*(x) as the objective function
			for epoch in range(num_epochs):
				for i, ((X_i, y_i), (X_o, y_o)) in enumerate(zip(self.inner_dataloader, self.outer_dataloader)):
					# Move to GPU
					X_i = X_i.to(self.device)
					y_i = y_i.to(self.device)
					X_o = X_o.to(self.device)
					y_o = y_o.to(self.device)
					mu_k = mu_k.to(self.device)
					# Zero all the parameter gradients
					a_k = self.NN
					loss = self.loss_H(mu_k, h_k, a_k, inner_grad22, outer_grad2, X_i, y_i, X_o, y_o)
					loss_values.append(loss.float())
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()
			return self.NN, loss_values
		else:
			raise AttributeError("Can only approximate h*(x) or a*(x), you must provide necessary inputs")

	def loss_H(self, mu_k, h_k, a_k, inner_grad22, outer_grad2, X_i, y_i, X_o, y_o):
		"""
		Returns a loss function to recover a*(x) that only depends on the output and the target.
		"""
		aT_in = a_k(X_i).T
		hessian = inner_grad22(mu_k, h_k, X_i, y_i)
		aT_hessian = aT_in @ hessian
		a_in = a_k(X_i)
		aT_out = a_k(X_o).T
		grad = outer_grad2(mu_k, h_k, X_o, y_o)
		return (1/2)*torch.mean(aT_hessian.double() @ a_in.double())+(1/2)*torch.mean(aT_out.double() @ grad.double())


class NeuralNetwork_a(nn.Module):
	"""
	A neural network to approximate the function a* for Neur. Imp. Diff.
	"""
	def __init__(self, layer_sizes, device):
		super(NeuralNetwork_a, self).__init__()
		self.device = device
		self.layer_1 = nn.Linear(layer_sizes[0], layer_sizes[1])
		nn.init.kaiming_uniform_(self.layer_1.weight)
		self.layer_2 = nn.Linear(layer_sizes[1], layer_sizes[2])
		nn.init.kaiming_uniform_(self.layer_2.weight)
		self.layer_3 = nn.Linear(layer_sizes[2], layer_sizes[3])
		nn.init.kaiming_uniform_(self.layer_3.weight)
		self.layer_4 = nn.Linear(layer_sizes[3], layer_sizes[4])

	def forward(self, x):
		# Move to GPU
		x = x.to(self.device)
		x = torch.relu(self.layer_1(x))
		x = torch.tanh(self.layer_2(x))
		x = torch.tanh(self.layer_3(x))
		x = self.layer_4(x)
		return x

class NeuralNetwork_h(nn.Module):
	"""
	A neural network to approximate the function h* for Neur. Imp. Diff.
	"""
	def __init__(self, layer_sizes, device):
		super(NeuralNetwork_h, self).__init__()
		self.device = device
		self.layer_1 = nn.Linear(layer_sizes[0], layer_sizes[1])
		nn.init.kaiming_uniform_(self.layer_1.weight)
		self.layer_2 = nn.Linear(layer_sizes[1], layer_sizes[2])
		nn.init.kaiming_uniform_(self.layer_2.weight)
		self.layer_3 = nn.Linear(layer_sizes[2], layer_sizes[3])
		nn.init.kaiming_uniform_(self.layer_3.weight)
		self.layer_4 = nn.Linear(layer_sizes[3], layer_sizes[4])

	def forward(self, x):
		# Move to GPU
		x = x.to(self.device)
		x = self.layer_1(x)
		x = self.layer_2(x)
		x = self.layer_3(x)
		x = self.layer_4(x)
		return x

class Data(Dataset):
	"""
	A class for input data.
	"""
	def __init__(self, X, y):
		self.X = X
		self.y = y
		self.len = len(self.y)

	def __getitem__(self, index):
		return self.X[index], self.y[index]

	def __len__(self):
		return self.len