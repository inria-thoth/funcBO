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
#from Experimentalist.utils import load_dict_from_json
from sympy import latex, sympify, Symbol
import Experimentalist as Exp
from Experimentalist.utils import group_by, aggregate, reduce_collectionDict_to_list,  Map, extract_data_from_config
from Experimentalist.helper import safe_argmin, safe_argmax, get_scalar_data,compute_mean_and_std


def make_plot_dicts(methods,key_name):
    labels = {m: key_name +' ' +m  for m in methods}
    colors = sns.color_palette("colorblind", n_colors=len(methods), desat=.7)
    sns.palplot(colors)
    color_dict_index = {m:i for i,m in enumerate(methods)}
    color_dict = {key:colors[value] for key,value in color_dict_index.items()}
    linestyles = {m:'-' for m in methods}
    return color_dict,labels,linestyles, colors



def make_gens(list_config_dicts,ynames,xname='epoch'):
    gen_list= []
    for yname in ynames:
        gen = get_xy_data_from_collection_list(
        list_config_dicts,
        yname= yname,
        xname=xname)
        gen_list.append(list(gen))
    return gen_list
    

    

def get_xy_data_from_collection(
                collection,
                xname, 
                yname):
    xname = 'metrics.'+xname
    yname = 'metrics.'+yname
    key_values = [xname,yname]
    data_list = []
    for config_dict in collection['config']:
        data = extract_data_from_config(config_dict,key_values)
        data_list.append(data)
    data,_ = compute_mean_and_std(data_list)
    m = min(len(data[xname]),len(data[yname])) 
    
    data[xname] = data[xname][:m]
    data[yname] = data[yname][:m]
    group_keys = collection['group_keys']
    key_name = '-'.join(['-'.join([el.split('.')[-1] for el in key]) for key in group_keys])
    
    return {'x':data[xname],
            'y': data[yname],
            'key':'_'.join(collection['group_keys_val']),
            'key_name': key_name}

def _get_xy_data_from_collection(
                config_dict,
                xname, 
                yname,
                group_key, 
                relative_error= True,
                avg_time_dict=None):
    for i,cur_dict in enumerate(config_dict):
        #try:
        all_data = load_multiple_dicts_from_json(cur_dict['logs']['path'])
        data = compute_mean_and_std(all_data)
        #data = load_dict_from_json(cur_dict['logs']['path'])
        if xname=='idealized_time':
            try:
                b_size = cur_dict['data']['b_size']
            except:
                b_size=1
            compute_idealized_time(data,avg_time_dict, b_size)
        #import pdb
        #pdb.set_trace()
        if relative_error:
            data[yname] = [d/data[yname][0] for d in data[yname] ]
        if not len(data[xname])==len(data[yname]):
            m = min(len(data[xname]),len(data[yname])) 
            data[xname] = data[xname][:m]
            data[yname] = data[yname][:m] 
        key_list = [str(reduce(dict.get,key,cur_dict)) for key in group_key]
        key_val = '_'.join(key_list)

        key_name = '_'.join(['.'.join(el) for el in group_key])
        out = {
                'x':data[xname],
                'y': data[yname],
                'key':key_val,
                'key_name': key_name
        }
        yield out



def get_xy_data_from_collection_list(collection_list, xname,yname):
    return [ get_xy_data_from_collection(collection,xname,yname) for collection in collection_list]



log_name = 'multitask_cifar100'
out_dir = '../data/outputs'
reader = Reader(os.path.join(out_dir,log_name), reload=False)



query = {'metadata.selection.name' : ["BGS"],
        'metadata.selection.linear_solver.lr': [0.001],
        'metadata.selection.optimizer.lr': [0.1],
        'metadata.selection.linear_op.stochastic':[True],
        'metadata.selection.linear_op.compute_new_grad':[True],
        }



out = reader.search(query)

query ={'metadata.selection.name' : ["BGS"],
        'metadata.selection.linear_solver.lr': [0.001],
        'metadata.selection.optimizer.lr': [0.1],
        'metadata.selection.linear_op.stochastic':[False],
        }


out += reader.search(query)


len(out)
def get_path(data, value_key, exp_dir):
    path= os.path.join(exp_dir,str(data[value_key[0]]))
    return {'path':path}
out.add_to_config(Map(get_path,value_keys=['metadata.logs.log_id'],args={'exp_dir': reader.root_dir}),
                parent_key='metadata.logs',
                group_key='metadata.logs.log_id')


out.add_to_config(Map(get_scalar_data,value_keys=['metrics.train_upper_loss'],args={'mode': 'last'}),
                parent_key='metadata',suffix='_last')

out.add_to_config(Map(get_scalar_data,value_keys=['metrics.train_upper_acc'],args={'mode': 'last'}),
                parent_key='metadata', suffix='_last')


group_keys = [['metadata.selection.linear_op.compute_new_grad',
              'metadata.selection.compute_latest_correction',
             'metadata.selection.linear_op.stochastic'],
             ['metadata.training.total_epoch']]


maps = [Map(safe_argmin,['metadata.metrics.train_upper_loss_last']),
        Map(safe_argmax,['metadata.metrics.train_upper_acc_last'])]

collection = group_by(out,group_keys)
reduced_dict = aggregate(collection,maps)
list_config_dicts = reduce_collectionDict_to_list(reduced_dict['metadata.metrics.train_upper_loss_last'])

#list_config_dicts = out.aggregate(grqoup_keys,maps)['metrics.train_upper_loss']


ynames = ['train_upper_acc','test_upper_acc_all']
gen_list = make_gens(list_config_dicts,ynames,group_keys[0])


methods = [data['key'] for data in gen_list[0]]

key_name = '-'.join(['-'.join([el.split('.')[-1] for el in key]) for key in group_keys])+':'

color_dict, labels_dict, linestyles_dict,colors = make_plot_dicts(methods,key_name)





