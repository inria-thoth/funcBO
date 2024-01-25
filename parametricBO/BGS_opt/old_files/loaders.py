import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
# adapted from https://github.com/JunjieYang97/stocBiO/blob/master/Hyperparameter-optimization/experimental/l2reg_on_twentynews.py

import numpy as np
import pickle as pkl
from itertools import repeat
import os
from core.utils import set_device_and_type
import learn2learn as l2l
from learn2learn.data.transforms import FusedNWaysKShots, LoadData, RemapLabels, ConsecutiveLabels, NWays, KShots
import copy
import torchvision
from torchvision import transforms
from itertools import cycle




class RingIterator:
	def __init__(self, initial_iterator,index=None):
		self.initial_iterator = initial_iterator
		self.index = index
		self.generator = None
	def make_generator(self):
		if self.index is not None:
			return zip(*[iter(self.initial_iterator),cycle([self.index])])
		else:
			return zip(*[iter(self.initial_iterator)])
	def __next__(self, *args):
		try:
			return next(self.generator)
		except:
			self.generator = self.make_generator()
			return next(self.generator)
	def __iter__(self):
		return self.make_generator()


class RingIteratorList:
	def __init__(self, ring_iterator_list):
		self.ring_iterator_list = ring_iterator_list
		self.generator = None
	def make_generator(self):
		return zip(*[iter(iterator) for iterator in self.ring_iterator_list])
	def __next__(self, *args):
		try:
			return next(self.generator)
		except:
			self.generator = self.make_generator()
			return next(self.generator)
	def __iter__(self):
		return self.make_generator()



class RingGenerator:
	def __init__(self, init_generator):
		self.init_generator = init_generator
		self.generator = None
	def make_generator(self):
		return zip(*[iter(self.initial_iterator)])
	def __next__(self, *args):
		try:
			return next(self.generator)
		except:
			self.generator = self.make_generator()
			return next(self.generator)
	def __iter__(self):
		return self.make_generator()






class CustomTensorIterator:
	def __init__(self, tensor_list, batch_size, device,dtype, **loader_kwargs):
		self.device= device
		self.dtype = dtype
		if len(tensor_list)==1:
			self.loader = DataLoader(TensorDataset(tensor_list[0]), batch_size=batch_size, **loader_kwargs)
		else:
			self.loader = DataLoader(TensorDataset(*tensor_list), batch_size=batch_size, **loader_kwargs)

		self.iterator = iter(self.loader)
		
	def __next__(self, *args):
		try:
			idx = next(self.iterator)
		except StopIteration:
			self.iterator = iter(self.loader)
			idx = next(self.iterator)
		if len(idx)==1 and isinstance(idx, list):
			return idx[0]
		else:
			idx = tuple(idx)
		return idx
	def __iter__(self):
		return iter(self.loader)


class CustomListIterator:
	def __init__(self, tensor_list, device,dtype, to_tuple=True):
		self.device= device
		self.dtype = dtype
		self.tensor_list = tensor_list
		self.iterator = iter(tensor_list)
		self.to_tuple = to_tuple
		
	def __next__(self, *args):
		try:
			idx = next(self.iterator)
		except StopIteration:
			self.iterator = iter(self.tensor_list)
			idx = next(self.iterator)
		if len(idx)==1 and isinstance(idx, list):
			return idx[0]
		#elif self.to_tuple:
		else:
			return tuple(idx)
		return idx
	def __iter__(self):
		return iter(self.tensor_list)
	def __getitem__(self,i):
		return self.tensor_list[i]
	def sample(self):
		return self.tensor_list.sample()

class CustomTaskIterator:
	def __init__(self, task_loader,adaptation_indices, device,dtype):
		self.device= device
		self.dtype = dtype
		self.task_loader = task_loader
		self.max_tasks = self.task_loader.num_tasks
		self.adaptation_indices = adaptation_indices
		self.num_tasks = 0
	def __next__(self, *args):
		if self.num_tasks >= self.max_tasks:
			raise StopIteration
		out = self.__getitem__(self.num_tasks)
		self.num_tasks +=1
		return out

	def __iter__(self):
		self.num_tasks = 0
		return self
	def __getitem__(self,i):
		data, label = self.task_loader[i]
		out = (data[self.adaptation_indices], label[self.adaptation_indices], i)
		return out
	def sample(self):
		i = np.random.randint(0,self.max_tasks,1)[0]
		return self.__getitem__(i)



class Loader(object):
	def __init__(self,num_workers,dtype,device,**args):

		self.num_workers = num_workers
		self.dtype = dtype
		self.device= device
		self.args= args
		self.name = self.args['name']
		
	def load_data(self,b_size):
		return NotImplementedError('Not implemented')
	def make_loaders(self):
		b_size = self.args['b_size']
		eval_b_size = self.args['eval_b_size']
		data, meta_data = self.load_data(b_size)
		eval_data,_ = self.load_data(eval_b_size)
		data['eval_lower_loader'] = eval_data['lower_loader']
		data['eval_upper_loader'] = eval_data['upper_loader']	
		self.data = data
		self.meta_data = meta_data

class DefaultLoader(Loader):
	def __init__(self,num_workers,dtype,device,**args):
		super(DefaultLoader,self).__init__(num_workers,dtype,device,**args)
		self.make_loaders()
	def load_data(self,b_size):
		return globals()['data_'+self.name](self.args,b_size, self.dtype, self.device, self.num_workers)

class TV_ClassifLoader(Loader):
	def __init__(self,num_workers,dtype,device,**args):
		super(TV_ClassifLoader,self).__init__(num_workers,dtype,device,**args)
		self.make_loaders()
	def load_data(self,b_size):
		return torchvision_classif_dataset(self.args,b_size, self.dtype, self.device, self.num_workers)


def torchvision_classif_dataset(args,b_size,dtype, device, num_workers):
	path = args['data_path']
	name = args['name']
	n_classes = dataset_n_classes(name)
	work_dir = os.getcwd()
	root = os.path.join(work_dir,path,name)
	klass = getattr(torchvision.datasets, name)
	train_data = klass(root, train = True,transform=augmentations(name), download = True)
	test_data = klass(root, train = False,transform=augmentations(name), download = True)

	train_loader = torch.utils.data.DataLoader(train_data,
										  batch_size=b_size,
										  shuffle=True,
										  num_workers=num_workers)

	test_loader = torch.utils.data.DataLoader(test_data,
										  batch_size=b_size,
										  shuffle=True,
										  num_workers=num_workers)
	
	train_data = []
	val_data = []
	n_val = args.val_size_ratio*len(train_loader)
	for i, data in enumerate(train_loader):
		data= set_device_and_type(data,device,dtype)
		if i<n_val:
			val_data.append(data)
		else:
			train_data.append(data)
	test_data = []
	for i, data in enumerate(test_loader):
		data=  set_device_and_type(data,device,dtype)
		test_data.append(data)
	

	data = {'lower_loader':CustomListIterator(data_train,device=device,dtype=dtype),
			 'upper_loader':CustomListIterator(data_val,device=device,dtype=dtype),
			 'test_upper_loader': CustomListIterator(test_data,device=device,dtype=dtype),
			 'test_lower_loader': CustomListIterator(test_data,device=device,dtype=dtype),
			}

	x,y= next(iterators[0])
	n_features, shape = get_n_features(x,n_classes)
	meta_data = {'n_classes':n_classes,
				'n_features':n_features,
				'shape':shape}

	return data, meta_data

def augmentations(name):
	if name == 'MNIST':
		return transforms.Compose([
							transforms.ToTensor(),
							transforms.Normalize((0.5), (0.5))
							])
	elif name =='CIFAR10':
		return transforms.Compose([
							transforms.ToTensor(),
							transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
							])
	elif name == 'FashionMNIST':
		return  transforms.Compose([
								transforms.ToTensor()
							])

def dataset_n_classes(name):
	if name in ['MNIST','CIFAR10','FashionMNIST']:
		return 10
	elif name in ['CIFAR100']:
		return 100

def get_n_features(x):
	shape = list(x.shape)
	shape[0] = n_classes
	n_features = np.prod(np.array(shape[1:]))
	return n_features, shape

def data_Zeros(args,b_size, device, dtype):
	lower_dim = args['lower']['dim']
	upper_dim = args['upper']['dim']
	lower_data = [torch.zeros([args['lower']['size'],lower_dim])]
	upper_data = [torch.zeros([args['upper']['size'],upper_dim])]
	meta_data = {'lower_dim':lower_dim,
				 'upper_dim':upper_dim}
	data = {'lower_loader':CustomTensorIterator(lower_data,batch_size=1,device=device,dtype=dtype),
		 'upper_loader':CustomTensorIterator(upper_data,batch_size=1,device=device,dtype=dtype),
		 'test_upper_loader': CustomTensorIterator(upper_data,batch_size=1,device=device,dtype=dtype),
		 'test_lower_loader': CustomTensorIterator(lower_data,batch_size=1,device=device,dtype=dtype),
		}
	return data, meta_data

def data_20newsgroups(args,b_size,dtype, device, num_workers):

	val_size_ratio= args['val_size_ratio']
	data_path = args['data_path']
	work_dir = os.getcwd()
	path = os.path.join(work_dir,data_path,'20newsgroups_'+str(b_size)+'.pkl')
	try:
		with open(path,'rb') as f:
			all_data = pkl.load(f)
			data, meta_data = all_data
	except:
		
		X, y = fetch_20newsgroups_vectorized(subset='train', return_X_y=True)
		x_test, y_test = fetch_20newsgroups_vectorized(subset='test', return_X_y=True)
		x_train, x_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=val_size_ratio)

		train_samples, n_features = x_train.shape
		test_samples, n_features = x_test.shape
		val_samples, n_features = x_val.shape
		n_classes = np.unique(y_train).shape[0]

		y_test = torch.from_numpy(y_test).long()
		y_train = torch.from_numpy(y_train).long()
		y_val = torch.from_numpy(y_val).long()
		
		x_train = from_sparse(x_train)
		x_test = from_sparse(x_test)
		x_val = from_sparse(x_val)

		dataset = [(x_train,y_train),(x_val,y_val),(x_test,y_test)]

		cuda = (device=='cuda')
		cuda = False
		kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
		iterators = []
		for x,y in dataset[:2]:
			iterators.append(CustomTensorIterator([x, y], batch_size=b_size,device='cpu',dtype=dtype, shuffle=True, **kwargs))
		train_iterator, val_iterator = iterators
		data_train, data_val= [], []
		for _ in range(train_samples // b_size+1):
			data_train.append(next(train_iterator))
		for _ in range(val_samples // b_size+1):
			data_val.append(next(val_iterator))


		data = {'lower_loader':CustomListIterator(data_train,device=device,dtype=dtype),
				 'upper_loader':CustomListIterator(data_val,device=device,dtype=dtype),
				 'test_upper_loader': CustomListIterator([(x_test,y_test)],device=device,dtype=dtype),
				 'test_lower_loader': CustomListIterator([(x_test,y_test)],device=device,dtype=dtype),
				}

		meta_data = {'n_classes':n_classes,
					'n_features':n_features}
		all_data = data,meta_data
		with open(path,'wb') as f:
			pkl.dump(all_data,f)	
	for key, data_loader in data.items():
		data_loader.device = device
		data_loader.dtype = dtype
	return data, meta_data



def from_sparse(x):
	x = x.tocoo()
	values = x.data
	indices = np.vstack((x.row, x.col))

	i = torch.LongTensor(indices)
	v = torch.FloatTensor(values)
	shape = x.shape

	return torch.sparse.FloatTensor(i, v, torch.Size(shape))


