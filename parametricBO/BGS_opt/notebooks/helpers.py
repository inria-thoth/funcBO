import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
import numpy as np
import matplotlib
from matplotlib import cm
import os
import json
from functools import reduce
import matplotlib.colors as mpl_colors
from math import log10
import matplotlib as mpl

def search_query(readers, query):
	out = []
	for r in readers:
		out += r.search(query)
	return out

def path_dict(out,colors,labels, exp_dir):
	out_dict= {}
	for p, c, l in zip(out,colors,labels):
		path=os.path.join(exp_dir,str(p['logs']['log_id']),'metrics.json')
		out_dict[p['solver']['solver_name']] = {'path':path, 'color':c,'label':l }
	
	return out_dict

def paths_grouped_by(out,group_key,variable_key,exp_dir,sort_ascend=True):
	# group_key: tuple of hierarchical keys
	# group_key: tuple of hierarchical keys
	out_dict= {}
	for p in out:
		key = reduce(dict.get,group_key,p)
		path=os.path.join(exp_dir,str(p['logs']['log_id']),'metrics.json')
		var = reduce(dict.get,variable_key,p)
		if key in out_dict.keys():
			out_dict[key]['path'].append(path)
			out_dict[key]['var'].append(var)
		else:
			out_dict[key] = {'path': [path], 'var': [var] }
	if sort_ascend:
		for key,value in out_dict.items():
			sorted_list = sorted(zip(value['var'],value['path']))
			value['var'] = [x for x,y in sorted_list]
			value['path'] = [y for x,y in sorted_list]
	return out_dict


def dicts_grouped_by(out,group_keys,variable_keys,value_keys):
	# group_key: tuple of hierarchical keys
	# group_key: tuple of hierarchical keys
	out_dict= {}
	for p in out:
		#try:
		key_list = [reduce(dict.get,key,p) for key in group_keys]
		key_name = '_'.join(key_list)
		var_list = [reduce(dict.get,key,p) for key in variable_keys]
		val_list = [reduce(dict.get,key,p) for key in value_keys]

		var_names = ['_'.join(key) for key in variable_keys]
		val_names   = ['_'.join(key) for key in value_keys]
		#import pdb
		#pdb.set_trace()

		if key_name not in out_dict.keys():
			out_dict[key_name] = {}
			for key_val,val in zip(val_names,val_list):
				out_dict[key_name][key_val] = []
			for key_var,var in zip(var_names,var_list):
				if var is not None:
					out_dict[key_name][key_var] = []
			
			
		for key_val,val in zip(val_names,val_list):
			out_dict[key_name][key_val].append(val)
		for key_var,var in zip(var_names,var_list):
			if var is not None:
				out_dict[key_name][key_var].append(var)
		#except: 
		#    pass

	return out_dict


def reduce(dict,group_keys, variable_keys,reduction_keys, value_keys, reduction_map, sort_ascend=True):
   # group_key: tuple of hierarchical keys
	# group_key: tuple of hierarchical keys
	reduce_dict= {}
	for p in out:
		#try:
		key_list = [reduce(dict.get,key,p) for key in group_keys]
		key_name = '_'.join(key_list)
		var_list = [reduce(dict.get,key,p) for key in variable_keys if key not in reduction_keys]
		var_list = [v for v in var_list if v is not None]
		var_list_key = '_'.join(var_list)

		val_list = [reduce(dict.get,key,p) for key in value_keys]
		
		var_names = ['_'.join(key) for key in variable_keys]
		val_names   = ['_'.join(key) for key in value_keys]
		#import pdb
		#pdb.set_trace()

		if key_name not in reduce_dict.keys():
			reduce_dict[key_name] = {}

			collection = {} 
			collection['values'] = {}
			collection['vars'] = {}
			for key_val,val in zip(val_names,val_list):
				collection['values'][key_val] = []
			for key_var,var in zip(var_names,var_list):
				collection['vars'][key_var] = var                    

			reduce_dict[key_name][var_list_key] = collection
		
		
			
		for key_val,val in zip(val_names,val_list):
			reduce_dict[key_name][var_list_key]['values'][key_val].append(val)
	key_names = list(reduce_dict.keys())
	out_dict = {}
	for key_name in key_names:
		var_keys = list(reduce_dict[key_name].keys())
		if key_name not in out_dict.keys():
			out_dict[key_name] = {}
		for var_key in var_keys:
			keys=   list( reduce_dict[key_name][var_key]['values'].keys())
			for key in keys:
				val = reduction_map(np.array(reduce_dict[key_name][var_key]['values'][key]))
				reduce_dict[key_name][var_key]['values'][key] = val
			out_dict['var'] = reduce_dict[key_name][var_key]['var']
			out_dict['values'] = reduce_dict[key_name][var_key]['values']
	return out_dict



def load_dict_from_json(json_file_name):
	out_dict = {}
	try:
		with open(json_file_name) as f:
			for line in f:
				cur_dict = json.loads(line)
				keys = cur_dict.keys()
				for key in keys:
					if key in out_dict:
						out_dict[key].append(cur_dict[key])
					else:
						out_dict[key] = [cur_dict[key]]
	except Exception as e:
		print(str(e))

	return out_dict







# def compute_time(data,index,avg_time_dict,prefix=''):
#     if index is None:
#         return {'real_time': np.inf,
#                 'idealized_time': np.inf}
#     else:
#         real_time = data[prefix+'time'][index]
#         idealized_time = data[prefix+'outer_grad'][index]*avg_time_dict['cost_outer_grad'][0]\
#                         + data[prefix+'inner_grad'][index]*avg_time_dict['cost_inner_grad'][0]\
#                         + data[prefix+'inner_hess'][index]*avg_time_dict['cost_inner_hess'][0]\
#                         + data[prefix+'inner_jac'][index]*avg_time_dict['cost_inner_jac'][0]
#         return {'real_time': real_time,
#                 'idealized_time': idealized_time}



def compute_idealized_time(data, avg_time_dict,b_size, prefix=''):
	O,G,H,J = data[prefix+'outer_grad'],\
				data[prefix+'inner_grad'],\
				data[prefix+'inner_hess'],\
				data[prefix+'inner_jac']
	if avg_time_dict is not None:
		t1,t2,t3,t4 = avg_time_dict['cost_outer_grad'][0],\
					avg_time_dict['cost_inner_grad'][0],\
					avg_time_dict['cost_inner_hess'][0],\
					avg_time_dict['cost_inner_jac'][0]
	else:
		t1,t2,t3,t4 = 1.,1.,1.,1.
		avg_time_dict = {}
		avg_time_dict['cost_outer_grad']  =[1.]
		avg_time_dict['cost_inner_grad']  =[1.]
		avg_time_dict['cost_inner_hess']  =[1.]
		avg_time_dict['cost_inner_jac']  =[1.]


	idealized_time = [b_size*(o*t1+g*t2+h*t3+j*t4) for o,g,h,j in zip(O,G,H,J) ]
	data['idealized_time'] = idealized_time




from functools import reduce
def dicts_grouped_by(out,group_keys,variable_keys,value_keys):
	# group_key: tuple of hierarchical keys
	# group_key: tuple of hierarchical keys
	out_dict= {}
	for p in out:
		#try:
		key_list = [reduce(dict.get,key,p) for key in group_keys]
		key_name = '_'.join(key_list)
		var_list = [reduce(dict.get,key,p) for key in variable_keys]
		val_list = [reduce(dict.get,key,p) for key in value_keys]

		var_names = ['_'.join(key) for key in variable_keys]
		val_names   = ['_'.join(key) for key in value_keys]
		#import pdb
		#pdb.set_trace()

		if key_name not in out_dict.keys():
			out_dict[key_name] = {}
			for key_val,val in zip(val_names,val_list):
				out_dict[key_name][key_val] = []
			for key_var,var in zip(var_names,var_list):
				if var is not None:
					out_dict[key_name][key_var] = []
			
			
		for key_val,val in zip(val_names,val_list):
			out_dict[key_name][key_val].append(val)
		for key_var,var in zip(var_names,var_list):
			if var is not None:
				out_dict[key_name][key_var].append(var)
		#except: 
		#    pass

	return out_dict

def add_time_to_eps(config_dict, epsilon,val_key,avg_time_dict=None):
	# takes dict with keys 'path' and 'var'
	path= config_dict['logs']['path']# os.path.join(config_dict['logs']['path'],'metrics.json')
	data = load_dict_from_json(path)  
	values = np.array(data[val_key])
	#print(values[0])
	values = values/values[0]
	index = next((i for i, x in enumerate(values<epsilon) if x), None)
	avg_time_keys = ['cost_outer_grad','cost_inner_grad','cost_inner_hess','cost_inner_jac']
	#if avg_time_dict is None:
	#    avg_time_dict = {k:data[k] for k in avg_time_keys if k in data}
	time_to_eps_dict = compute_time(data,index,avg_time_dict)
	if 'results' in config_dict.keys():
		config_dict['results'].update(time_to_eps_dict)
	else:
		config_dict['results'] = time_to_eps_dict
	return avg_time_dict


def add_error_at_time(config_dict, time,val_key,avg_time_dict=None):
	# takes dict with keys 'path' and 'var' 
	path= config_dict['logs']['path']#os.path.join(exp_dir,str(config_dict['logs']['log_id']),'metrics.json')
	data = load_dict_from_json(path) 
	try:
		b_size = config_dict['data']['b_size'] 
	except:
		b_size = 1
	compute_idealized_time(data,avg_time_dict, b_size) 
	values = np.array(data[val_key])
	values = values/values[0]
	values_at_time_dict = {}
	for time_key,value in time.items():
		time_array =  np.array(data[time_key])
		index = next((i for i, x in enumerate(time_array>value) if x), len(time_array)-1)
		#import pdb
		#pdb.set_trace()
		values_at_time_dict['err_at_'+time_key] = values[index]

	if 'results' in config_dict.keys():
		config_dict['results'].update(values_at_time_dict)
	else:
		config_dict['results'] = values_at_time_dict

def add_res_to_data(config_dict, val_keys, mode='last'):

	path=config_dict['logs']['path']

	data = compute_mean_and_std(load_multiple_dicts_from_json(path), mean_keys= val_keys)
	if mode=='last':
		def operation(a):
			return a[-1]
	elif mode=='first':
		def operation(a):
			try:
				return a[1]
			except:
				return 'Not existing'
	elif mode=='smallest':
		def operation(a):
			return min(a)
	elif mode=='largest':
		def operation(a):
			return max(a)
	else:
		raise NotImplementedError
	values = {val_key: operation( data[val_key]) for val_key in val_keys}
	if 'results' in config_dict.keys():
		config_dict['results'].update(values)
	else:
		config_dict['results'] = values

	

def add_path_to_data(config_dict,exp_dir):
	path= os.path.join(exp_dir,str(config_dict['logs']['log_id']),'metrics.json')
	config_dict['logs']['path'] = path
	
def compute_time(data,index,avg_time_dict,prefix=''):
	if index is None:
		return {'real_time': np.inf,
				'idealized_time': np.inf}
	else:
		if avg_time_dict is not None:
			t1,t2,t3,t4 = avg_time_dict['cost_outer_grad'][0],\
						avg_time_dict['cost_inner_grad'][0],\
						avg_time_dict['cost_inner_hess'][0],\
						avg_time_dict['cost_inner_jac'][0]
		else:
			t1,t2,t3,t4 = 1.,1.,1.,1.

		real_time = data[prefix+'time'][index]
		idealized_time = data[prefix+'outer_grad'][index]*t1\
						+ data[prefix+'inner_grad'][index]*t2\
						+ data[prefix+'inner_hess'][index]*t3\
						+ data[prefix+'inner_jac'][index]*t4
		return {'real_time': real_time,
				'idealized_time': idealized_time}

def aggregate_res_dict(input_dict,group_keys, variable_keys, value_keys):
   # group_key: tuple of hierarchical keys
	# group_key: tuple of hierarchical keys
	reduce_dict= {}
	for i,p in enumerate(input_dict):
		#print( 'iter: ' + str(i))
		key_list = [str(reduce(dict.get,key,p)) for key in group_keys]
		key_name = '_'.join(key_list)
		var_list = [reduce(dict.get,key,p) for key in variable_keys]
		var_list = [str(v) for v in var_list if v is not None]
		var_list_key = '_'.join(var_list)

		val_list = [reduce(dict.get,key,p) for key in value_keys]
		
		var_names = ['_'.join(key) for key in variable_keys]
		val_names   = ['_'.join(key) for key in value_keys]
		
		if key_name not in reduce_dict.keys():
			reduce_dict[key_name] = {}
		if var_list_key not in reduce_dict[key_name].keys():
			collection = {} 
			collection['values'] = {}
			collection['vars'] = {}
			collection['refs'] = {}
			for key_val,val in zip(val_names,val_list):
				collection['values'][key_val] = []
				collection['refs'][key_val] = []
			for key_var,var in zip(var_names,var_list):
				collection['vars'][key_var] = var                    
			reduce_dict[key_name][var_list_key] = collection
			
		for key_val,val in zip(val_names,val_list):
			reduce_dict[key_name][var_list_key]['values'][key_val].append(val)
			reduce_dict[key_name][var_list_key]['refs'][key_val].append(p)
	return reduce_dict    

def safe_float(potential_float):
	try:
		return float(potential_float)

	except ValueError:
		return potential_float
def reduce_res_dict(reduce_dict,reduction_maps,variable_keys, sort=True, is_index=False):
		
	key_names = list(reduce_dict.keys())
	out_dict = {}
	for key_name in key_names:
		var_keys = list(reduce_dict[key_name].keys())
		if key_name not in out_dict.keys():
			out_dict[key_name] = {}
			keys=  list( reduce_dict[key_name][var_keys[0]]['values'].keys())\
					+ list( reduce_dict[key_name][var_keys[0]]['vars'].keys())
			for key in keys:
				out_dict[key_name][key] =[]
			keys= list( reduce_dict[key_name][var_keys[0]]['values'].keys())
			for key in keys:
				out_dict[key_name]['index_'+key] =[]      
		for var_key in var_keys:
			keys=  list( reduce_dict[key_name][var_key]['values'].keys())
			for key,red_map in zip(keys,reduction_maps):
				val_array = np.array(reduce_dict[key_name][var_key]['values'][key])
				if is_index:
					index = red_map(val_array)
					p = reduce_dict[key_name][var_key]['refs'][key][index]
					out_dict[key_name]['index_'+key].append(p)
					val = val_array[index]
				else:
					val = red_map(val_array)
				out_dict[key_name][key].append(val)

			keys=  list( reduce_dict[key_name][var_key]['vars'].keys())
			for key in keys:
				val = safe_float(reduce_dict[key_name][var_key]['vars'][key])
				out_dict[key_name][key].append(val)
	if sort:
		var_names = ['_'.join(key) for key in variable_keys]
		sort_ascend(out_dict,var_names[0])
	return out_dict

def sort_ascend(input_dict,sort_key):
	for key,value in input_dict.items():
		keys = [key for key in value.keys() if key not in [sort_key]]
		vals =  [ value[key] for key in keys] 
		#import pdb
		#pdb.set_trace()
		sorted_list = sorted(zip(value[sort_key],*vals))
		value[sort_key] = [x[0] for x in sorted_list]
		for i,key in enumerate(keys):
			value[key] =  [ x[i+1] for x in  sorted_list]
def plot_helper(ax,
				res_dict,
				colors_dict,
				labels_dict, 
				linestyles =None,  
				linestyle='--',
				linewidth=3., 
				xname= 'time', 
				yname= 'outer_loss', 
				xlabel='Time/s', 
				ylabel='Loss value', 
				xlim=None, 
				ylim=None, 
				lw=5.,
				xscale= 'log',
				yscale= 'log',
				title='',
				with_legend=True): 
	for key,value in res_dict.items():
		#data = load_dict_from_json(value['path'])
		#import pdb
		#pdb.set_trace()
		if linestyles is None:
			linestyles = {v:linestyle for v in colors_dict.keys()}
		ax.plot(value[xname],
				value[yname],
				color=colors_dict[key],
				label=labels_dict[key],
				lw=lw, 
				linestyle=linestyles[key])
	ax.set_xscale(xscale)
	ax.set_yscale(yscale)
	ax.set_xlabel(xlabel, fontsize=25)
	ax.set_ylabel(ylabel, fontsize=25)
	ax.set_title(title,fontsize=25)
	if not ylim is None:
		ax.set_ylim(ylim)
	if not xlim is None:
		ax.set_xlim(xlim)
	if with_legend:
		ax.legend()

def load_multiple_dicts_from_json(paths):
	if not isinstance(paths,list):
		paths = [paths]
	return [load_dict_from_json(path) for path in paths]

def compute_mean_and_std(all_data, mean_keys=None):

	if len(all_data)==1:
		return all_data[0]
	else:
		keys = list(all_data[0].keys())
		if mean_keys is None:
			mean_keys= keys
		out = {key:0. for key in keys }
		out.update({'std_'+key: 0. for key in mean_keys})
		for i,p in enumerate(all_data):
			for key in mean_keys:
				#import pdb
				#pdb.set_trace()
				if i==0:
					index = len(p[key])
					out[key] = out[key] + np.asarray(p[key])[:index]
					out['std_'+key] = out['std_'+key] + (np.asarray(p[key])[:index])**2

				else:
					index = min(out[key].size,len(p[key]))
					out[key] = out[key][:index] + np.asarray(p[key])[:index]
					out['std_'+key] = out['std_'+key][:index] + (np.asarray(p[key])[:index])**2
		for key in mean_keys:
			out[key] = out[key]/(i+1)
			out['std_'+key] = out['std_'+key]/(i+1) - (out[key])**2
		return out


def plot_from_config(ax,
				data_gen,
				x,y,
				colors_dict,
				labels_dict,
				linestyles_dict=None,  
				linewidth=3.,
				xlabel='', 
				ylabel='',
				title ='',
				xlim=None, 
				ylim=None, 
				lw=5.,
				xscale= 'log',
				yscale= 'log',
				fontsize=25,
				ticks_fontsize=20,
				with_legend=True):

	for data in data_gen:
		key_value = data['group_keys_val']
		m = min(len(data[x]),len(data[y])) 
		data[x] = data[x][:m]
		data[y] = data[y][:m]
		if xscale=='log':
			data[x][0] +=0.0001

		im = ax.plot(data[x],
					data[y],
					color=colors_dict[key_value],
					label=labels_dict[key_value],
					lw=lw, 
					linestyle=linestyles_dict[key_value])
	ax.set_yscale(yscale)
	ax.set_xscale(xscale)   
	ax.set_xlabel(xlabel, fontsize=fontsize)
	ax.set_ylabel(ylabel, fontsize=fontsize)
	ax.set_title(title, fontsize=fontsize)
	mpl.rc('ytick', labelsize=ticks_fontsize)
	mpl.rc('xtick', labelsize=ticks_fontsize)
	if not ylim is None:
		ax.set_ylim(ylim)
	if not xlim is None:
		ax.set_xlim(xlim)
	if with_legend:
		ax.legend()
	#return im


def make_plot_dicts(methods,key_name):
    labels = {m: key_name +' ' +m  for m in methods}
    colors = sns.color_palette("colorblind", n_colors=len(methods), desat=.7)
    sns.palplot(colors)
    color_dict_index = {m:i for i,m in enumerate(methods)}
    color_dict = {key:colors[value] for key,value in color_dict_index.items()}
    linestyles = {m:'-' for m in methods}
    return color_dict,labels,linestyles, colors



def extract_xy_data_from_configs(
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



def colormap_2D( base_colors):
	base_colors = [np.asarray(c) for c in base_colors]
	end, b_1, b_2, start = base_colors
	#cmap = lambda p1,p2 : tuple(start + p1*(1-p2)*(b_1-start) + p2*(1-p2)*(b_2-start) + p1*p2*(end-start))
	cmap = lambda p1,p2 : tuple(start + p1*(1-p2)*(b_1-start) + p2*(1-p2)*(b_2-start) + p1*p2*(end-start))
	return cmap

def colormap_2D(base_colors, alpha=3):
	base_colors = [np.asarray(c) for c in base_colors]
	size_b = len(base_colors)-alpha
	#cmap = lambda p1,p2 : tuple(start + p1*(1-p2)*(b_1-start) + p2*(1-p2)*(b_2-start) + p1*p2*(end-start))
	def cmap(p1,p2):
		index = int(p1*size_b) if p1<1. else size_b-alpha
		return tuple(base_colors[index] + p2*(base_colors[index+alpha+1]-base_colors[index]))
		
	return cmap


def colormap_2D(base_colors,T,alpha=3):
	#cmap = lambda p1,p2 : tuple(start + p1*(1-p2)*(b_1-start) + p2*(1-p2)*(b_2-start) + p1*p2*(end-start))
	def cmap(p1,p2):
		#index_1 = int(p1*N) if p1<1. else N-1
		#index_2 = int(p2*T) if p2<1. else T-1
		index = p1*(T+alpha) + p2
		return base_colors[index]
		
	return cmap

def colormap_1D(base_colors):
	def cmap(p1):
 #       index = int(p1*N) if p1<1. else N-1
		return base_colors[p1]
		
	return cmap  

def colormap_2D_dict(res_dict, keys, alpha=3, colorname="rocket"):
	
	#colors_2 = sns.color_palette("mako", n_colors=len(res_dict), desat=.7)
	#colors[0],colors[-1]
	#base_colors = [colors[0],colors[int(len(colors)/2)],colors_2[int(len(colors_2)/2)],colors[-1]]
   
	
	
	keys_dict = {'_'.join(key): [] for key in keys} 
	for p in res_dict:
		for key in keys:
			keys_dict['_'.join(key)].append(reduce(dict.get,key,p))
	#import pdb
	#pdb.set_trace()
	
	sorted_vals = {'_'.join(key):list(set(keys_dict['_'.join(key)])) for key in keys}
	#sorted_vals = {'_'.join(key):sorted_vals['_'.join(key)].sort() for key in keys}
	dict_vals = {'_'.join(key):{} for key in keys}
	for key,val in sorted_vals.items():
		val.sort()
		dict_vals[key] = dict(zip(val, [*range(len(val))]))
	
	#min_vals = {'_'.join(key):min(keys_dict['_'.join(key)] for key in keys}

	
	#len_first_key = alpha*(len(keys_dict['_'.join(keys[0])])+1)
	#base_colors = list(sns.color_palette("rocket", n_colors=len_first_key, desat=.7))
   
	#cmap = colormap_2D(base_colors, alpha=alpha)
	colormap = []
	cmap, base_colors = make_color_palette(keys_dict, alpha, colorname=colorname)
	labels = {'left':[],'right':[]}
	if len(keys)>1:
		for key_1,key_2 in zip(keys_dict['_'.join(keys[0])],keys_dict['_'.join(keys[1])]):
			val_1 = dict_vals['_'.join(keys[0])][key_1]
			val_2 = dict_vals['_'.join(keys[1])][key_2]
#            if max_vals['_'.join(keys[0])]>min_vals['_'.join(keys[0])]:
#                val_1 = (key_1-min_vals['_'.join(keys[0])])/(max_vals['_'.join(keys[0])]-min_vals['_'.join(keys[0])])
#            else:
#                val_1 = 1.
#            if max_vals['_'.join(keys[1])]>min_vals['_'.join(keys[1])]:
#                val_2 = (key_2-min_vals['_'.join(keys[1])])/(max_vals['_'.join(keys[1])]-min_vals['_'.join(keys[1])])
#            else:
#                val_2 =1.
			colormap.append(cmap(val_1,val_2))
			labels['left'].append(key_1)
			labels['right'].append(key_2)
	else:
		for key_1 in keys_dict['_'.join(keys[0])]:
			val = dict_vals['_'.join(keys[0])][key_1]
#            if max_vals['_'.join(keys[0])]>min_vals['_'.join(keys[0])]:
#                val = (key_1-min_vals['_'.join(keys[0])])/(max_vals['_'.join(keys[0])]-min_vals['_'.join(keys[0])])
#            else:
#                val=1.
			colormap.append(cmap(val))
			labels['right'].append(key_1)
	# make RGB image, p1 to red channel, p2 to blue channel
	#Legend = np.dstack((Cp1, C0, Cp2))
	return colormap,labels, base_colors

def make_color_palette(var_dict, alpha, colorname="rocket"):
	keys = list(var_dict.keys())
	lens = [len(list(set(var_dict[key]))) for key in keys]
	if len(lens)>1:
		colors = sns.color_palette(colorname, n_colors=lens[0]*(lens[1]+alpha), desat=.7)
		base_colors = list(colors)
		base_colors = base_colors[::-1]
		cmap = colormap_2D(base_colors,lens[1],alpha )
	else:
		colors = sns.color_palette(colorname, n_colors=lens[0], desat=.7)
		base_colors = list(colors)
		base_colors = base_colors[::-1]
		cmap = colormap_1D(base_colors)
	return cmap, colors

def make_colorbar():

	# parameters range between 0 and 1
	plt.subplots_adjust(left=0.1, right=0.65, top=0.85)
	cax = fig.add_axes([0.7,0.55,0.3,0.3])
	cax.imshow(Legend, origin="lower", extent=[0,1,0,1])
	cax.set_xlabel(name_1)
	cax.set_ylabel(name_2)
	#cax.set_title("2D cmap legend", fontsize=10)
import copy
def make_colorbar(colormap, labels,names, pos,fontsize, spacing=0.08, labelpad=20): 
	ticks_labels_left = list(set(labels['left']))
	ticks_labels_left.sort()
	ticks_labels_right = list(set(labels['right']))
	ticks_labels_right.sort()
	if len(ticks_labels_left)>0:
		ticks_labels_right = [''.join([names[1],r"$10^{%01d}$" % (int(log10(p))) ]) for  p in ticks_labels_right]
	else:
		ticks_labels_right = [r"$10^{%01d}$" % int(log10(p)) for  p in ticks_labels_right]
	ticks_labels_left = [r"$10^{%01d}$" % int(log10(p)) for  p in ticks_labels_left]
	colorbar_index(colormap,ticks_labels_left,ticks_labels_right, pos=pos,fontsize=fontsize, label=names[0], spacing=spacing, labelpad=labelpad)

def eformat(f, prec, exp_digits):
	s = "%.*e"%(prec, f)
	#s =  ""
	mantissa, exp = s.split('e')
	#mantissa = int(mantissa)
	mantissa= 1
	# add 1 to digits as 1 is taken by sign +/-
	return "%se%+0*d"%(mantissa, exp_digits+1, int(exp))    
	
def colorbar_index( colors_list, ticks_labels_left,ticks_labels_right, pos=[0.85, 0.1, 0.075, 0.8],fontsize=20, label='T',spacing=0.08 , labelpad=20):
	#ncolors = len(colors)

	n_ticks_left = len(ticks_labels_left)
	n_ticks_right = len(ticks_labels_right)
	if n_ticks_left>0:
		colors_list = list(set(colors_list))
		colors_list.sort(reverse=True)
		ncolors = n_ticks_left *n_ticks_right
		new_ticks_labels_right = ticks_labels_right*n_ticks_right
		new_ticks_labels_left = ['']*n_ticks_left
		new_ticks_labels_left[0] = ticks_labels_left[0]
		new_ticks_labels_left[-1] = ticks_labels_left[-1]
		for j, el in enumerate(ticks_labels_right):
			index = j*n_ticks_left+int(n_ticks_left/2)
			new_ticks_labels_right[index] = el 
	else:
		#import pdb
		#pdb.set_trace()
		colors_list = list(set(colors_list))
		colors_list.sort(reverse=True)
		ncolors= n_ticks_right
		new_ticks_labels_right = ticks_labels_right
	

	if n_ticks_left>0:
		for i, el_right in enumerate(ticks_labels_right):
			new_colors_list = colors_list[i*n_ticks_left:(i+1)*n_ticks_left]
			cmap = mpl_colors.ListedColormap(new_colors_list)
			mappable = cm.ScalarMappable(cmap=cmap)
			mappable.set_array([])
			mappable.set_clim(-0.5, n_ticks_left+0.5)
			new_pos = [i for i in pos]
			
			new_pos[3] = pos[3]/len(ticks_labels_right)
			new_pos[1] = pos[1] + i*(new_pos[3]+spacing) 
			
			cax12 = plt.axes(new_pos)
			cax3 = cax12.twinx()
			
			colorbar = plt.colorbar(mappable, cax=cax12)
			colorbar.set_ticks(np.linspace(1, 4, 4))
			colorbar.set_ticklabels(['','','',el_right])
			colorbar.ax.tick_params(labelsize=fontsize, rotation=90)
			colorbar.ax.tick_params(size=0)
			colorbar.ax.yaxis.set_ticks_position('left')
			colorbar.ax.yaxis.set_label_position('left')
			cax3.set_ylabel(ylabel=label, rotation=0, fontsize=20,labelpad=labelpad)
			cax3.set_yticks(np.linspace(0, n_ticks_left, n_ticks_left+2)) 
			#ticks = [ ticks_labels_left[0],'','',ticks_labels_left[-1]]
			cax3.set_yticklabels(['']+new_ticks_labels_left+[''],fontsize=fontsize)
	else:
		cmap = mpl_colors.ListedColormap(colors_list)
		mappable = cm.ScalarMappable(cmap=cmap)
		mappable.set_array([])
		mappable.set_clim(-0.5, n_ticks_right+0.5)
		cax12 = plt.axes(pos)
		colorbar = plt.colorbar(mappable, cax=cax12)
		colorbar.set_ticks(np.linspace(0, n_ticks_right, n_ticks_right))
		colorbar.set_ticklabels(new_ticks_labels_right)
		colorbar.set_label(label, rotation=0,fontsize=20,labelpad=20)
		#colorbar.ax.tick_params(labelsize=fontsize, size=0)  


	