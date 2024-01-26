import torch
import examples.multitask.autolambda.create_dataset as datasets 

import importlib
import os
from torchvision import transforms
import pickle as pkl

import numpy as np
import signal

class ConcatIndexedDatasets(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.len = np.max(np.array([len(d)for d, ind in self.datasets]))
    def __getitem__(self, i):
        return tuple([(d[i%len(d)],ind) for d, ind in self.datasets])

    def __len__(self):
        return self.len

def augmentations(name, dataset_type='train'):
	if name == 'CIFAR100MTL':
		if dataset_type=='train':
			return transforms.Compose([
				    transforms.RandomCrop(32, padding=4),
				    transforms.RandomHorizontalFlip(),
				    transforms.ToTensor(),
				    transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
					])
		else:
			return transforms.Compose([
			    	transforms.ToTensor(),
			    	transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
					])


def make_loaders(args,num_workers,dtype,device):
	
	download = False
	b_size = args['b_size']
	num_tasks = args['num_tasks']
	subset_id = args['subset_id']
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


		module, attr = os.path.splitext(name)
		name = attr[1:]
		try: 
			module = importlib.import_module(module)
		except:
			module = globals()[module]
		klass = getattr(module, name)
		train_data = 	[(klass(root, train = True,transform=augmentations(name, dataset_type='train'), subset_id=i, download = download),i) for i in range(num_tasks)]
		if subset_id>=0:
			test_data = [(klass(root, train = False,transform=augmentations(name, dataset_type='test'), subset_id=subset_id, download = download),subset_id)]
		else:
			test_data = [(klass(root, train = False,transform=augmentations(name, dataset_type='test'), subset_id=i, download = download),i) for i in range(num_tasks)]
		all_data = train_data, test_data
		with open(path,'wb') as f:
			pkl.dump(all_data,f)


	Cat_dataset_train = ConcatIndexedDatasets(train_data)
	Cat_dataset_test  = ConcatIndexedDatasets(test_data)

	if subset_id >=0:
		Cat_dataset_val   =  ConcatIndexedDatasets([train_data[subset_id]])
	else:
		Cat_dataset_val   =  ConcatIndexedDatasets(train_data)

	def worker_init(x):
		signal.signal(signal.SIGINT, signal.SIG_IGN)

	loader_kwargs = {'shuffle':True, 'num_workers':4, 'pin_memory': True, 'worker_init_fn':worker_init}



	train_loaders = torch.utils.data.DataLoader(dataset=Cat_dataset_train, batch_size=b_size, **loader_kwargs)
	test_loaders = torch.utils.data.DataLoader(dataset=Cat_dataset_test, batch_size=b_size, **loader_kwargs)
	val_loaders = torch.utils.data.DataLoader(dataset=Cat_dataset_val, batch_size=b_size, **loader_kwargs)

	loaders = {'lower_loader':train_loaders,
		 'upper_loader':val_loaders,
		 'test_upper_loader': test_loaders,
		 'test_lower_loader': None,
		}
	meta_data = {'num_tasks': num_tasks, 
				 'subset_id': subset_id, 
				 'total_samples': len(train_data[0][0]),
				 'b_size': b_size }

	return loaders, meta_data
