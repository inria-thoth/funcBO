#!/bin/bash

# Define parent directories for working and log directories
parent_work_dir="../../data/.workdir"

 Run truncated iterative differentiation
 parent_log_dir="../../../data/outputs/debug"
 HYDRA_FULL_ERROR=1 OC_CAUSE=1 python applications/IVRegression/parametricBO/main.py \
                 seed=1\
                 selection=ITD\
                 +mlxp.use_scheduler=False\
                 +mlxp.use_version_manager=False\
                 +mlxp.interactive_mode=False\
                 +mlxp.use_logger=True\
                 +mlxp.logger.parent_log_dir=$parent_log_dir\
                 +mlxp.version_manager.parent_work_dir=$parent_work_dir\