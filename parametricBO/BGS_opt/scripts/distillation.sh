#!/bin/bash


log_name=distillation_linear
total_epoch=2000


cpus=1

cluster_name=''
device=-1
dtype=64
b_size=1024
name=''

launcher='besteffort'
app='/scratch/clear/marbel/anaconda3/bin/python'
gpumem=2
hours=1
cluster_name=''

training='dataset_distillation'
resume=True

name='MNIST'
n_features=784

upper_lr=.001
lower_lr=.001
solver_lr=.0001
correction=True
selection='BGS'

warm_start_iter=0
unrolled_iter=1
solver_iter=1
disp_freq=1




python launch_jobs.py --multirun \
		system.dtype=$dtype\
		system.device=$device\
		launcher=$launcher\
		launcher.app=$app\
		launcher.gpumem=$gpumem\
		launcher.hours=$hours\
		cluster.name=$cluster_name\
		logs.log_name=$log_name\
		training=$training \
		training.loader.name=$name\
		training.loader.n_features=$n_features\
		training.resume=$resume\
		training.loader.b_size=$b_size\
		training.total_epoch=$total_epoch\
		selection=$selection\
		training.upper.scheduler.use_scheduler=False\
		selection.scheduler.use_scheduler=False\
		training.upper.optimizer.lr=$upper_lr\
		selection.optimizer.lr=$lower_lr\
		selection.linear_solver.lr=$solver_lr\
		selection.warm_start_iter=$warm_start_iter\
		selection.unrolled_iter=$unrolled_iter\
		selection.linear_solver.n_iter=$solver_iter\
		selection.linear_solver.name='core.linear_solvers.GD'\
		selection.correction=False\
		selection.optimizer.momentum=0.\
		training.lower.objective.reg=0.\
		training.metrics.disp_freq=$disp_freq\




python launch_jobs.py --multirun \
		system.dtype=$dtype\
		system.device=$device\
		launcher=$launcher\
		launcher.app=$app\
		launcher.gpumem=$gpumem\
		launcher.hours=$hours\
		cluster.name=$cluster_name\
		logs.log_name=$log_name\
		training=$training \
		training.loader.name=$name\
		training.loader.n_features=$n_features\
		training.resume=$resume\
		training.loader.b_size=$b_size\
		training.total_epoch=$total_epoch\
		selection=$selection\
		training.upper.scheduler.use_scheduler=False\
		selection.scheduler.use_scheduler=False\
		training.upper.optimizer.lr=$upper_lr\
		selection.optimizer.lr=$lower_lr\
		selection.linear_solver.lr=0.0001,0.001,0.000000001,0.0000001,0.00001\
		selection.warm_start_iter=$warm_start_iter\
		selection.unrolled_iter=$unrolled_iter\
		selection.linear_solver.n_iter=1,2,10\
		selection.linear_solver.name='core.linear_solvers.GD'\
		selection.correction=True\
		selection.optimizer.momentum=0.\
		training.lower.objective.reg=0.\
		training.metrics.disp_freq=$disp_freq\


