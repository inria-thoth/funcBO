#!/bin/bash

# Define parent directories for working and log directories
parent_work_dir="../../data/.workdir"

parent_log_dir="../../data/outputs/Penalty5k_grid_search"
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python applications/IVRegression/PenaltyBased/main.py \
                seed=1,2,3,4\
                data_size=5000\
                max_epochs=2000\
                val_size=200\
                penalty=1,0.1,0.01,0.001\
                optimizer.lr=0.01,0.001,0.0001\
                optimizer.wd=0.1,0.01,0.001\
                +mlxp.use_scheduler=False\
                +mlxp.use_version_manager=False\
                +mlxp.interactive_mode=False\
                +mlxp.use_logger=True\
                +mlxp.logger.parent_log_dir=$parent_log_dir\
                +mlxp.version_manager.parent_work_dir=$parent_work_dir\
