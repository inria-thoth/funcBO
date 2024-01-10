#!/bin/bash


parent_work_dir="../../data/.workdir" 
parent_log_dir="../../data/outputs/debug" 

HYDRA_FULL_ERROR=1 OC_CAUSE=1 python -m ipdb main.py \
                lam_u=0.1\
                lam_V=0.1\
                +mlxp.use_scheduler=False\
                +mlxp.use_version_manager=False\
                +mlxp.use_logger=True\
                +mlxp.logger.parent_log_dir=$parent_log_dir\
                +mlxp.version_manager.parent_work_dir=$parent_work_dir\