#!/bin/bash


log_name=multitask_cifar100_scheduling_weight_decay
dtype=32
training=multitask_cifar100
trainer_name='examples.multitask.trainer.Trainer'
resume=True

gpumem=14
cluster_name=''
device=-1
hours=20
launcher='besteffort'
app='/scratch/clear/marbel/anaconda3/bin/python'

# correction=True
# python launch_jobs.py --multirun   \
# 		system.dtype=$dtype\
# 		system.device=$device\
# 		launcher=$launcher\
# 		launcher.app=$app\
# 		launcher.gpumem=$gpumem\
# 		launcher.hours=$hours\
# 		cluster.name=$cluster_name\
# 		logs.log_name=$log_name\
# 		training=$training \
# 		training.resume=$resume\
# 		training.trainer_name=$trainer_name\
# 		training.lower.selection.correction=$correction\

# selection='BGS'
# python launch_jobs.py --multirun  \
# 		system.dtype=$dtype\
# 		system.device=$device\
# 		launcher=$launcher\
# 		launcher.app=$app\
# 		launcher.gpumem=$gpumem\
# 		launcher.hours=$hours\
# 		cluster.name=$cluster_name\
# 		logs.log_name=$log_name\
# 		training=$training \
# 		training.resume=$resume\
# 		training.trainer_name=$trainer_name\
# 		selection=$selection\
# 		selection.optimizer.lr=0.1,0.01\
# 		selection.linear_solver.lr=0.001,0.0001\
# 		selection.linear_op.stochastic=True,False\
# 		selection.linear_op.compute_new_grad=True,False\
# 		selection.compute_latest_correction=True,False\
# 		selection.linear_op.use_new_input=True,False\


selection='Unrolled'
python launch_jobs.py --multirun  \
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
		training.trainer_name=$trainer_name\
		selection='Unrolled'\
		selection.optimizer.lr=0.1\
		training.upper.scheduler.use_scheduler=True\
		training.upper.optimizer.weight_decay=0.1,1.,0.01\


# selection='BGS'
# python launch_jobs.py --multirun  \
# 		system.dtype=$dtype\
# 		system.device=$device\
# 		launcher=$launcher\
# 		launcher.app=$app\
# 		launcher.gpumem=$gpumem\
# 		launcher.hours=$hours\
# 		cluster.name=$cluster_name\
# 		logs.log_name=$log_name\
# 		training=$training \
# 		training.resume=$resume\
# 		training.trainer_name=$trainer_name\
# 		training.upper.optimizer.weight_decay=0.1,1.\
# 		selection=$selection\
# 		selection.optimizer.lr=0.1\
# 		selection.linear_solver.lr=0.0001\
# 		training.upper.scheduler.use_scheduler=True\

# hours=24
# gpumem=16
# log_name=multitask_cifar100_correction
# selection='BGS'
# python launch_jobs.py --multirun  \
# 		system.dtype=$dtype\
# 		system.device=$device\
# 		launcher=$launcher\
# 		launcher.app=$app\
# 		launcher.gpumem=$gpumem\
# 		launcher.hours=$hours\
# 		cluster.name=$cluster_name\
# 		logs.log_name=$log_name\
# 		training=$training \
# 		training.resume=$resume\
# 		training.trainer_name=$trainer_name\
# 		training.upper.optimizer.weight_decay=0.01,0.0\
# 		selection=$selection\
# 		selection.optimizer.lr=0.1\
# 		selection.linear_solver.lr=0.0001\
# 		training.upper.scheduler.use_scheduler=True\
# 		selection.warm_start_iter=0\
# 		selection.unrolled_iter=1\



