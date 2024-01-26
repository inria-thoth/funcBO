
import numpy as np


import matplotlib.pyplot as plt
from matplotlib import pyplot
import matplotlib 
import matplotlib.cm as cm
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches
import seaborn as sns
import matplotlib as mpl
import pickle as pkl
import os
#from kcrf.estimator import simple_estimator as np_simple_est
import time 
import csv
import torch
from torch.autograd import Variable
import torch.optim as optim
from copy import deepcopy
import pandas as pd
import hydra
from helpers import *
from matplotlib.colors import hsv_to_rgb
import matplotlib.colors as mpl_colors
from math import log10

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf
from Experimentalist.reader import Reader
import json
from Experimentalist.utils import load_dict_from_json
from sympy import latex, sympify, Symbol


def safeargmin(a):
	try:
		return np.nanargmin(a)
	except:
		return -1
def safeargmax(a):
	try:
		return np.nanargmax(a)
	except:
		return -1
def reduced_dict_to_list_dicts(out_dict, index, key):
	out = []
	for method in out_dict.keys():
		if len(out_dict[method][key])>index:
			out.append(out_dict[method][key][index])
	return out

def aggregate_paths_in_config_dicts(out_dict, key):
	out = []
	for method in out_dict.keys():
			cur_dict = copy.deepcopy(out_dict[method][key][0])
			cur_dict['logs']['log_id'] = [ p['logs']['log_id'] for p in out_dict[method][key]]
			out.append(cur_dict)
	return out

def make_paths_to_seeds(aggregate_dict, res_key):
	out = []
	for method, p in aggregate_dict.items():
		for keys, values in p.items():
			
			
			cur_dict = copy.deepcopy(values['refs'][res_key][0])
			cur_dict['logs']['path'] = [ p['logs']['path'] for p in values['refs'][res_key]]
			out.append(cur_dict)  
	return out
def make_plot_dicts(out):
	methods = list(set([p['training']['tag'] for p in out]))
	labels = {m:m  for m in methods}
	colors = sns.color_palette("colorblind", n_colors=len(methods), desat=.7)
	sns.palplot(colors)
	color_dict_index = {m:i for i,m in enumerate(methods)}
	color_dict = {key:colors[value] for key,value in color_dict_index.items()}
	linestyles = {m:'--' for m in methods}
	return color_dict,labels,linestyles, colors


def make_paths_to_seeds(aggregate_dict, res_key):
	out = []
	for method, p in aggregate_dict.items():
		for keys, values in p.items():
			
			
			cur_dict = copy.deepcopy(values['refs'][res_key][0])
			cur_dict['logs']['path'] = [ p['logs']['path'] for p in values['refs'][res_key]]
			out.append(cur_dict)  
	return out

def op_aggregate_paths(res_dict):

	if res_dict:
		out = copy.deepcopy(res_dict[0])
		out['logs']['path'] = [p['logs']['path'] for p in res_dict]
		return [out]
	else:
		return res_dict






log_name = 'multitask_cifar100'
out_dir = 'data/outputs'
reader = load_method(log_name,out_dir, reload=False)





query = {'selection.name' : ["BGS"]}






out = reader.search(query)


val_keys = ['train_upper_loss','train_upper_acc','test_upper_loss_all','test_upper_acc_all']

failed_exps = []
for i,p in enumerate(out):
	try:
		add_res_to_data(p, val_keys, mode='last')
	except:
		failed_exps.append(i)

out = [p  for i,p in enumerate(out) if i not in  failed_exps]

group_keys = [('selection','name')]
group_keys = [('selection','linear_op','compute_new_grad')]
#variable_keys = [('data','b_size')]
variable_keys = [('training','total_epoch')]
#variable_keys = [('solver','inner_forward','n_iter')]

value_keys = [('results','train_upper_loss'),('results','train_upper_acc')]
aggregate_dict = aggregate_res_dict(out,group_keys, variable_keys, value_keys)
reduced_dict = reduce_res_dict(aggregate_dict, [safeargmin,safeargmax],variable_keys,sort=True, is_index=True)
list_config_dicts = reduced_dict_to_list_dicts(reduced_dict, 0, 'index_results_train_upper_loss')

num_x =2
num_y = 1
fig, ax = plt.subplots(num_y,num_x, figsize=(9*num_x,6*num_y))
avg_time_dict = None
xlabel = ''
ylabel = ''
title  = ''
yname = 'test_acc'
xlim= [0,200]
ylim = [0,100.]
xlabel = 'Gradient evaluations'
gen = extract_xy_data_from_configs(
		list_config_dicts,
		yname= 'test_upper_acc_all',
		xname='epoch',
		key_name=('training','tag'), 
		relative_error= False,
		avg_time_dict=None
		)

for data in gen:
	print(len(data['x']))
	print(len(data['y']))











