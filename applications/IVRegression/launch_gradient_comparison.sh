#!/bin/bash

# Define parent directories for working and log directories
parent_work_dir="../../data/.workdir"

# Run the main script for the DFIV method
parent_log_dir="../../data/outputs/dfiv"
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python applications/IVRegression/DFIV/main.py \
                +mlxp.use_scheduler=True\
                +mlxp.use_version_manager=False\
                +mlxp.interactive_mode=False\
                +mlxp.use_logger=True\
                +mlxp.logger.parent_log_dir=$parent_log_dir\
                +mlxp.version_manager.parent_work_dir=$parent_work_dir\

# Run the main script Functional Bilevel Optimization with linear closed form gradient
parent_log_dir="../../data/outputs/funcBO"
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python applications/IVRegression/funcBO/main.py \
                +mlxp.use_scheduler=True\
                +mlxp.use_version_manager=False\
                +mlxp.interactive_mode=False\
                +mlxp.use_logger=True\
                +mlxp.logger.parent_log_dir=$parent_log_dir\
                +mlxp.version_manager.parent_work_dir=$parent_work_dir\