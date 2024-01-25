import torch
import examples.multitask.autolambda.create_dataset as datasets 

import importlib
import os
from torchvision import transforms
import pickle as pkl

import numpy as np
import signal

import torchvision

from core import utils
from itertools import cycle
class ListIterator:
	def __init__(self, loader, device,dtype):
		self.device= device
		self.dtype = dtype
		self.loader = loader
		self.tensor_list = None
		self.iterator = None
	def make_tensor_list(self):
		self.tensor_list = []
		for i, data in enumerate(self.loader):
			data= utils.set_device_and_type(data,self.device,self.dtype)
			self.tensor_list.append(data)			
		self.iterator = iter(self.tensor_list)

	def __next__(self, *args):
		try:
			idx = next(self.iterator)
		except:
			self.make_tensor_list()
			return next(self.iterator)
	def __iter__(self):
		try:
		
			return iter(self.tensor_list)
		except:
			self.make_tensor_list()
			return iter(self.tensor_list)
		
	def __getitem__(self,i):
		return self.tensor_list[i]
	def __getstate__(self):
		return {'tensor_list': self.tensor_list,
				'iterator': None}
	def __setstate__(self, d ):
		self.tensor_list = d['tensor_list']
		self.iterator = None


def make_loaders(args,num_workers,dtype,device):

	download = True
	b_size = args['b_size']
	data_path = args['data_path']
	name = args['name']

	work_dir = os.getcwd()
	root = os.path.join(work_dir,data_path,name)
	path = os.path.join(work_dir,data_path,name+'.pkl')
	try:
		with open(path,'rb') as f:
			all_data = pkl.load(f)
			train_data, test_data = all_data
	except:

		if name=='MNIST':
			transform = transforms.Compose([
			# you can add other transformations in this list
				transforms.ToTensor(),
				transforms.Normalize((0.5), (0.5))
			])
			train_data = torchvision.datasets.MNIST(root, train = True,transform=transform, download = True)
			test_data = torchvision.datasets.MNIST(root, train = False,transform=transform, download = True)
			n_classes = 10
		elif name=='CIFAR10':
			transform = transforms.Compose(
				[transforms.ToTensor(),
				transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
			)

			train_data = torchvision.datasets.CIFAR10(root, train = True,transform=transform, download = True)
			test_data = torchvision.datasets.CIFAR10(root, train = False,transform=transform, download = True)
			n_classes = 10
		elif name=='FashionMNIST':
			transform = transforms.Compose([
			# you can add other transformations in this list
				transforms.ToTensor()
			])
			train_data = torchvision.datasets.FashionMNIST(root, train = True,transform=transform, download = True)
			test_data = torchvision.datasets.FashionMNIST(root, train = False,transform=transform, download = True)
			n_classes = 10
		all_data = train_data, test_data, n_classes
		with open(path,'wb') as f:
			pkl.dump(all_data,f)



	



	train_loader = torch.utils.data.DataLoader(train_data,
										  batch_size=b_size,
										  shuffle=True,
										  num_workers=1)

	test_loader = torch.utils.data.DataLoader(test_data,
										  batch_size=b_size,
										  shuffle=True,
										  num_workers=1)
	train_loader = ListIterator(train_loader,device, dtype)
	test_loader = ListIterator(test_loader,device,dtype)

	x,y= next(iter(train_loader))
	shape = list(x.shape)
	shape[0] = n_classes
	n_features = np.prod(np.array(shape[1:]))


	loaders = {'lower_loader':train_loader,
		 'upper_loader':train_loader,
		 'test_upper_loader': test_loader,
		 'test_lower_loader': None,
		}
	meta_data = {'n_features':n_features, 
				 'n_classes': n_classes, 
				 'shape':shape,
				 'total_samples': train_data.data.shape[0],
				 'b_size': b_size }



	return loaders, meta_data
