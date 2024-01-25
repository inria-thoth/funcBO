import torch
import inspect
from core import utils


class Metrics(object):
	def __init__(self,args, device, dtype):
		self.args = args
		self.device = device
		self.dtype = dtype	
		self.metrics = []
		self.values = {}
		self.count_values = {}
	def register_metric(self,func, loader, max_iters,prefix,func_args = {}, condition=None, metric='value'):
		if condition is None:
			condition = lambda counter: True
		metric = {	'func': func,
					'loader': loader,
					'max_iters':max_iters,
					'func_args':  func_args,
					'prefix': prefix,
					'condition': condition,
					'metric':metric}
		self.metrics.append(metric)
	def eval_metrics(self, counter, values):
		out_dicts = [values]
		for metric_dict in self.metrics:
			if metric_dict['condition'](counter):
				metric = globals()[metric_dict['metric']]
				out_dict =  metric(metric_dict['func'], 
									  metric_dict['loader'],
									  metric_dict['func_args'],
									  metric_dict['max_iters'], 
									  metric_dict['prefix'], self.device, self.dtype) 
				out_dicts.append(out_dict)	
		out_dicts = {k:v for d in out_dicts for k, v in d.items()}

		for k,val in out_dicts.items():
			if k in self.values:
				self.values[k]+=val
				self.count_values[k] +=1
			else:
				self.values[k] = val
				self.count_values[k] =1
		#self.values.append(out_dicts)
		
	def avg_metrics(self):
		avg_dict = {key:value/self.count_values[key] for key,value in self.values.items()}
		self.values = {}
		return avg_dict




def value(func,loader,func_args, max_iter, prefix, device, dtype):
	value = 0
	accuracy = 0
	counter = 0
	args = inspect.getfullargspec(func)[0]

	with torch.no_grad():
		for data in loader:
			if counter > max_iter and max_iter>0:
				break
			#if len(data)==1 and isinstance(data, list):
			#	data = data[0]
			data = utils.set_device_and_type(data,device,dtype)
			counter += 1
			loss,acc = func(data,with_acc=True,**func_args)
			value = value + loss
			accuracy = accuracy + acc

		
	accuracy = accuracy/counter
	value = value/counter
	return {prefix+'_loss': value.item(), prefix+'_acc': 100*accuracy.item()}





def multivalue(func,loader,func_args, max_iter, prefix, device, dtype):
	value = 0
	accuracy = 0
	counter = 0
	all_values = 0
	all_accuracy = 0
	with torch.no_grad():
		for data in loader:
			if counter > max_iter and max_iter>0:
				break
			#if len(data)==1 and isinstance(data, list):
			#	data = data[0]
			data = utils.set_device_and_type(data,device,dtype)
			counter += 1
			loss,losses,acc,all_acc = func(data,with_acc=True,all_losses=True,**func_args)
			value = value + loss
			all_values = all_values+ losses
			accuracy = accuracy + acc
			all_accuracy = all_accuracy + all_acc
	accuracy = accuracy/counter
	value = value/counter
	all_accuracy = all_accuracy/counter
	all_values = all_values/counter
	
	all_values = torch.chunk(all_values,all_values.shape[0],dim=0)
	out_dict = {prefix+'_loss_all': value.item()}
	out_dict.update({prefix+'_loss_task_'+str(i): val.item()
						for i,val in enumerate(all_values)})
	out_dict.update({prefix+'_acc_all': 100*accuracy.item()})
	all_accuracy = torch.chunk(all_accuracy,all_accuracy.shape[0],dim=0)
	out_dict.update({prefix+'_acc_task_'+str(i): 100*acc.item() for i,acc in enumerate(all_accuracy)})
	return out_dict

















