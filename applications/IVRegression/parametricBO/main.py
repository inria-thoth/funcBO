import mlxp
from funcBO.utils import set_seed 
from applications.IVRegression.parametricBO.trainer import Trainer
from omegaconf import OmegaConf

def resolve_tuple(*args):
    return tuple(args)

OmegaConf.register_new_resolver('as_tuple', resolve_tuple)

@mlxp.launch(config_path='./configs',
             seeding_function=set_seed)

def train(ctx: mlxp.Context) -> None:
    try:
        # Attempt to load the latest checkpoint using the logger from the MLXP context
        trainer = ctx.logger.load_checkpoint(log_name='last_ckpt')
        print("Loading from the latest checkpoint")
    except:
        # If loading the checkpoint fails, print a message and start training from scratch
        print("Failed to load the checkpoint, starting from scratch")
        # Create a new instance of the Trainer class with the configuration and logger from the MLXP context
        # Check if the run logs (in ctx.logger) already exist, if so, delete them here (only .json files)
        
        from parametricBO.BGS_opt.core.linear_solvers import Normal_CG
        trainer = Trainer(ctx.config, ctx.logger)

    # Call the train method of the Trainer instance
    trainer.train()

# Entry point of the script, executing the train function if the script is run as the main module
if __name__ == "__main__":
    train()