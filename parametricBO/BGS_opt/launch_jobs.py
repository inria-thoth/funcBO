# script for submitting jobs to cluster

import os
import logging
import hydra
 
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf
from Experimentalist.launcher import create

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@hydra.main(config_name='config.yaml',config_path='./configs' )
def main(cfg: DictConfig) -> None:
    logger.info(f"Current working directory: {os.getcwd()}")
    try:
        create(cfg)
    except Exception as e:
        print('No job launched')
        logger.critical(e, exc_info=True)


if __name__ == "__main__":
    main()
