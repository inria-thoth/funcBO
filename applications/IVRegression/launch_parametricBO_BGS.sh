#!/bin/bash

# Define parent directories for working and log directories
parent_work_dir="../../data/.workdir"

#Run parametric implicit differentiation
parent_log_dir="../../../data/outputs/debug"
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python applications/IVRegression/parametricBO/main.py \
                selection=BGS\
                lam_V=0.1,0.01,0.001\
                optimizer.lr=0.01,0.001,0.0001\
                linear_solver.lr=0.01,0.001,0.0001\
                linear_solver.n_iter=2,20\
                +mlxp.use_scheduler=False\
                +mlxp.use_version_manager=False\
                +mlxp.interactive_mode=False\
                +mlxp.use_logger=True\
                +mlxp.logger.parent_log_dir=$parent_log_dir\
                +mlxp.version_manager.parent_work_dir=$parent_work_dir\