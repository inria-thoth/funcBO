import time
import mlxp
from DFIVtraining import Trainer
from funcBO.utils import set_seed 




@mlxp.launch(config_path='configs/DFIV',
             seeding_function=set_seed)
def train(ctx: mlxp.Context)->None:

    try:
        training  = ctx.logger.load_checkpoint(log_name='last_ckpt') 
        print("Loading from latest checkpoint")
     
    except:
        print("Failed to load checkpoint, Starting from scratch")

        training = Trainer(ctx.config, ctx.logger)

    training.train()


if __name__ == "__main__":
    
    train()
