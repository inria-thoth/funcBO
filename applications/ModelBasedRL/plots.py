import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import mlxp as mlxpy
import pdb

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator, NullFormatter, FixedLocator, ScalarFormatter
import numpy as np
import os
import json

def save(n, ext='.pdf', save_figs=True,dirname = 'figures', **kwargs):
    if save_figs==True:
        kwargs.setdefault('bbox_inches', 'tight')
        kwargs.setdefault('pad_inches', 0.1)
        kwargs.setdefault('transparent', True)
        plt.savefig(os.path.join(dirname, n + ext), **kwargs)

def make_plot_dicts(methods, key_name):
    labels = {m: key_name +' ' +m  for m in methods}
    colors = sns.color_palette("colorblind", n_colors=len(methods), desat=.7)
    sns.palplot(colors)
    color_dict_index = {m:i for i,m in enumerate(methods)}
    color_dict = {key:colors[value] for key,value in color_dict_index.items()}
    linestyles = {m:'-' for m in methods}
    return color_dict, labels, linestyles, colors

root = '../data/outputs'
parent_log_dir = os.path.join(root,'cartpole_4')
reader = mlxpy.Reader(parent_log_dir, reload=False)

query = "info.status == 'COMPLETE'"

results = reader.filter(query_string=query)

from mlxp.data_structures.contrib.aggregation_maps import AvgStd

group_keys = ['config.agent_type', 'config.inner_lr', 'config.tau']
grouped_results = results.groupBy(group_keys)
agg_maps = [AvgStd('eval.episode_return'),AvgStd('eval.step')]
agg_results = grouped_results.aggregate(agg_maps)

def mean_std_array(arrays, keys, end_index=0):
    out = {}
    first = True
    for el in arrays:
        for key in keys:
            import pdb 
            
            data = np.array(el[key][end_index:])
            
            if first:
                out[key+'_avg'] = data
                out[key+'_std'] = data**2
            else:
                out[key+'_avg'] += data
                out[key+'_std'] += data**2                
        first=False
    for key in keys:
        out[key+'_avg'] = out[key+'_avg']/len(arrays)
        out[key+'_std'] = out[key+'_std']/len(arrays)
        out[key+'_std'] = out[key+'_std'] - out[key+'_avg']**2
        out[key+'_std'] = np.sqrt(out[key+'_std'])
    return out
    
def mean_std(results,keys, end_index=0):
    out = {}
    for key, value in results.items():
        new_values = mean_std_array(value, keys, end_index=end_index)
        out[key] = new_values
    return out

def find_best(results, field, groups_idx):
    best_key = {}
    best_value = {}
    
    for key,value in results.items():
        group = key[groups_idx]
        cur_value = np.mean(value[field])
        if group not in best_value or cur_value > best_value[group]:
            best_value[group]= cur_value
            best_key[group] = key
    return best_key

keys = ['eval.step', 'eval.episode_return']
res = mean_std(grouped_results, keys, end_index=-201)

train_keys = ['train_returns.step', 'train_returns.episode_return']
train_res = mean_std(grouped_results, train_keys, end_index=-200)

best_keys = find_best(train_res, field='train_returns.episode_return_avg', groups_idx=0)

# Colors used in the paper:
# brownish yellow '#bdb76b', 
# blue '#00b0f6', 
# green '#00bf7d', 
# pink '#e76bf3', 
# red '#f8766d'

colors  ={'omd': '#f8766d',
          'funcBO':'#00b0f6',
          'vep':'#00bf7d',
          'mle':'#e76bf3'}

labels  ={'omd': 'OMD',
          'funcBO':'funcID',
          'vep':'VEP',
          'mle':'MLE'}

# Plotting the results
figure_name = 'cartpole_4'
plt.clf()
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

best_keys_list = [value for key,value in best_keys.items()]
for key, value in res.items():
    if key in best_keys_list:
        steps = value['eval.step_avg']
        returns = value['eval.episode_return_avg']
        std_returns = value['eval.episode_return_std']  # Standard deviation values
        # Plot mean
        ax.plot(steps, returns, label=labels[key[0]], color=colors[key[0]], linewidth=1.5)
        # Fill standard deviation
        ax.fill_between(steps, returns - std_returns, returns + std_returns, color=colors[key[0]], alpha=0.1)

plt.legend(title='Algorithms', loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, title_fontsize='14')
plt.ylabel('Reward', fontsize=14, labelpad=10)
plt.xlabel('Steps', fontsize=14)
plt.grid(True, linewidth=0.5)
# Vertical axis scale
plt.yscale('linear')
ax = plt.gca()
ax.yaxis.set_major_formatter(ScalarFormatter())
plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax.grid(which='minor', linestyle=':', linewidth=0.2)
# Set the color of the edges of the main plot
for spine in ax.spines.values():
    spine.set_edgecolor('grey')
# Set the color of the ticks
ax.tick_params(axis='x', which='both', color='darkgrey')
ax.tick_params(axis='y', which='both', color='darkgrey')
plt.tight_layout()
plt.savefig("figures/"+figure_name+".png", dpi=400)