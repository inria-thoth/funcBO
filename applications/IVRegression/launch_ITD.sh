#!/bin/bash

# Define parent directories for working and log directories
parent_work_dir="../../data/.workdir"

# Grid search over hyperparameters
# seed=1,2,3,4\
# val_size=200\
# selection.unrolled_iter=2,5\
# selection.warm_start_iter=18,15\
# selection.optimizer.lr=0.01,0.001,0.0001\
# selection.optimizer.weight_decay=0.1,0.01,0.001\
# outer_optimizer.outer_lr=0.01,0.001,0.0001\

parent_log_dir="../../data/outputs/parametricBO_ITD5k"
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python applications/IVRegression/parametricBO/main.py \
                selection=ITD\
                seed=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20\
                data_size=5000\
                batch_size=2500\
                max_epochs=100\
                selection.unrolled_iter=2\
                selection.warm_start_iter=18\
                selection.optimizer.lr=0.001\
                selection.optimizer.weight_decay=0.01\
                outer_optimizer.outer_lr=0.01\
                +mlxp.use_scheduler=True\
                +mlxp.use_version_manager=False\
                +mlxp.interactive_mode=False\
                +mlxp.use_logger=True\
                +mlxp.logger.parent_log_dir=$parent_log_dir\
                +mlxp.version_manager.parent_work_dir=$parent_work_dir\

parent_log_dir="../../data/outputs/parametricBO_ITD10k"
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python applications/IVRegression/parametricBO/main.py \
                selection=ITD\
                seed=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20\
                data_size=10000\
                batch_size=5000\
                max_epochs=100\
                selection.unrolled_iter=2\
                selection.warm_start_iter=18\
                selection.optimizer.lr=0.001\
                selection.optimizer.weight_decay=0.01\
                outer_optimizer.outer_lr=0.01\
                +mlxp.use_scheduler=True\
                +mlxp.use_version_manager=False\
                +mlxp.interactive_mode=False\
                +mlxp.use_logger=True\
                +mlxp.logger.parent_log_dir=$parent_log_dir\
                +mlxp.version_manager.parent_work_dir=$parent_work_dir\