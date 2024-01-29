#!/bin/bash

# Define parent directories for working and log directories
parent_work_dir="../../../data/.workdir"


#Run parametric implicit differentiation
#parent_log_dir="../../../data/outputs/cartpole"
parent_log_dir="../../../data/outputs/cartpole_3"

# HYDRA_FULL_ERROR=1 OC_CAUSE=1 python -m ipdb applications/ModelBasedRL/funcBO/main.py \
#                 agent_type='vep'\
#                 seed=0\
#                 inner_lr=0.0003\
#                 tau=0.01\
#                 +mlxp.use_scheduler=False\
#                 +mlxp.use_version_manager=False\
#                 +mlxp.interactive_mode=Flase\
#                 +mlxp.use_logger=True\
#                 +mlxp.logger.parent_log_dir=$parent_log_dir\
#                 +mlxp.version_manager.parent_work_dir=$parent_work_dir\





HYDRA_FULL_ERROR=1 OC_CAUSE=1 python applications/ModelBasedRL/funcBO/main.py \
                agent_type='vep','mle','omd','funcBO'\
                seed=0,1,2,3,4,5,6,7,8,9\
                inner_lr=0.0003,0.001,0.003\
                tau=0.01,0.005\
                +mlxp.use_scheduler=True\
                +mlxp.use_version_manager=True\
                +mlxp.interactive_mode=True\
                +mlxp.use_logger=True\
                +mlxp.logger.parent_log_dir=$parent_log_dir\
                +mlxp.version_manager.parent_work_dir=$parent_work_dir\


 









