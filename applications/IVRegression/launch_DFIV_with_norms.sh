#!/bin/bash

# Define parent directories for working and log directories
parent_work_dir="../../data/.workdir"

# Run the main script Functional Bilevel Optimization
parent_log_dir="../../data/outputs/DFIV_with_norms"
# Try setting version manager to True
#dual_solver.optimizer.lr=0.0001,0.0001\
#dual_solver.num_iter=100,10,1\
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python applications/IVRegression/DFIV_with_norms/main.py \
                +mlxp.use_scheduler=True\
                +mlxp.use_version_manager=False\
                +mlxp.interactive_mode=False\
                +mlxp.use_logger=True\
                +mlxp.logger.parent_log_dir=$parent_log_dir\
                +mlxp.version_manager.parent_work_dir=$parent_work_dir\
