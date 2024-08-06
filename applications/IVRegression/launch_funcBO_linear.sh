#!/bin/bash

# Define parent directories for working and log directories
parent_work_dir="../../data/.workdir"

parent_log_dir="../../data/outputs/funcBO_linear5k"
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python applications/IVRegression/funcBO/main.py \
                seed=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20\
                data_size=5000\
                batch_size=2500\
                max_epochs=100\
                +mlxp.use_scheduler=True\
                +mlxp.use_version_manager=False\
                +mlxp.interactive_mode=False\
                +mlxp.use_logger=True\
                +mlxp.logger.parent_log_dir=$parent_log_dir\
                +mlxp.version_manager.parent_work_dir=$parent_work_dir\

parent_log_dir="../../data/outputs/funcBO_linear10k"
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python applications/IVRegression/funcBO/main.py \
                seed=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20\
                data_size=10000\
                batch_size=5000\
                max_epochs=100\
                +mlxp.use_scheduler=True\
                +mlxp.use_version_manager=False\
                +mlxp.interactive_mode=False\
                +mlxp.use_logger=True\
                +mlxp.logger.parent_log_dir=$parent_log_dir\
                +mlxp.version_manager.parent_work_dir=$parent_work_dir\