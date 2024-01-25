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


def make_loaders(args,num_workers,dtype,device):


	loaders = {'lower_loader':[torch.zeros([11,1000])] ,
		 'upper_loader':[torch.zeros([20,1000])],
		 'test_upper_loader': cycle([torch.zeros([1])]),
		 'test_lower_loader': None,
		}
	meta_data = {}



	return loaders, meta_data
