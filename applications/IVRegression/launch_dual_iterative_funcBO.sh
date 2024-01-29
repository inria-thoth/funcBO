#!/bin/bash

# Define parent directories for working and log directories
parent_work_dir="../../data/.workdir"

# Grid search over hyperparameters
#dual_solver.optimizer.lr=0.01,0.001,0.0001,0.00001,0.000001\

# Run the main script Functional Bilevel Optimization with linear closed form gradient
parent_log_dir="../../data/outputs/funcBO_dual_iterative"
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python applications/IVRegression/funcBO_dual_iterative/main.py \
                data_size=10000\
                batch_size=5000\
                seed=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20\
                +mlxp.use_scheduler=True\
                +mlxp.use_version_manager=False\
                +mlxp.interactive_mode=False\
                +mlxp.use_logger=True\
                +mlxp.logger.parent_log_dir=$parent_log_dir\
                +mlxp.version_manager.parent_work_dir=$parent_work_dir\