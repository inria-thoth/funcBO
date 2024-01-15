#!/bin/bash

# Define parent directories for working and log directories
parent_work_dir="../../data/.workdir" 
parent_log_dir="../../data/outputs/funcBO" 

# Run the main script Functional Bilevel Optimization
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python applications/IVRegression/funcBO/main.py \
                +mlxp.use_scheduler=False\
                +mlxp.use_version_manager=False\
                +mlxp.use_logger=True\
                +mlxp.logger.parent_log_dir=$parent_log_dir\
                +mlxp.version_manager.parent_work_dir=$parent_work_dir\

parent_log_dir="../../data/outputs/dfiv" 

# Run the main script for the DFIV method
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python applications/IVRegression/DFIV/main.py \
                +mlxp.use_scheduler=False\
                +mlxp.use_version_manager=False\
                +mlxp.use_logger=True\
                +mlxp.logger.parent_log_dir=$parent_log_dir\
                +mlxp.version_manager.parent_work_dir=$parent_work_dir\