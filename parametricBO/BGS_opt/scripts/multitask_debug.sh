#!/bin/bash


log_name=debug
dtype=32
training=multitask_cifar100
trainer_name='examples.multitask.trainer.Trainer'
resume=True
 
gpumem=14
cluster_name=''
device=-1
hours=17
launcher='besteffort'
app='/scratch/clear/marbel/anaconda3/bin/python'
warm_start_iter=0
unrolled_iter=1
correction=True
selection='Unrolled'
# HYDRA_FULL_ERROR=1   OC_CAUSE=1  python -m ipdb run.py --multirun   \
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
# 		selection.optimizer.lr=0.01\


selection='Unrolled'
HYDRA_FULL_ERROR=1   OC_CAUSE=1  python -m ipdb run.py --multirun   \
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
		training.upper.optimizer.weight_decay=0.0\
		selection=$selection\
		selection.optimizer.lr=0.1\
		training.upper.scheduler.use_scheduler=False\
		selection.warm_start_iter=0\
		selection.unrolled_iter=1\




