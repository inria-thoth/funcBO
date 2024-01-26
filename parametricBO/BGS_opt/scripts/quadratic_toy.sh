#!/bin/bash


log_name=debug
total_epoch=2000


cpus=1

cluster_name=''
device=-2
dtype=64
b_size=1
name=''

launcher='besteffort'
app='/scratch/clear/marbel/anaconda3/bin/python'
gpumem=2
hours=1
cluster_name=''

training='quadratic_toy'
resume=True
upper_lr=1.
lower_lr=0.9
solver_lr=0.9



upper_lr=1.
lower_lr=0.9
solver_lr=0.9

correction=True
selection='BGS'

warm_start_iter=1
unrolled_iter=0
solver_iter=1

outer_cond=10
lower_cond=10
lower_dim=1000
upper_dim=2000



HYDRA_FULL_ERROR=1   OC_CAUSE=1 python -m ipdb run.py --multirun \
		system.dtype=$dtype\
		system.device=$device\
		launcher=$launcher\
		launcher.app=$app\
		launcher.gpumem=$gpumem\
		launcher.hours=$hours\
		cluster.name=$cluster_name\
		logs.log_name=$log_name\
		training=$training \
		training.resume=$resume\
		training.loader.b_size=$b_size\
		selection=$selection\
		selection.optimizer.lr=$lower_lr\
		selection.correction=True\
		training.upper.scheduler.use_scheduler=False\
		training.upper.optimizer.lr=$upper_lr\
		selection.warm_start_iter=$warm_start_iter\
		selection.unrolled_iter=$unrolled_iter\
		training.total_epoch=$total_epoch\
		selection.linear_solver.name='core.linear_solvers.GD'\
		selection.linear_solver.n_iter=$solver_iter\
		selection.linear_solver.lr=$solver_lr\
		training.upper.objective.cond=$outer_cond\
		training.lower.objective.cond=$lower_cond\
		training.lower.model.dim=$lower_dim\
		training.upper.model.dim=$upper_dim\
		selection.optimizer.momentum=0.\
		selection.scheduler.use_scheduler=False\
		 +training.lower.objective.with_sin=True\


warm_start_iter=10
unrolled_iter=0
solver_iter=10


