import mlxp
import os

def clear_dir(log_dir):
    for filename in os.listdir(log_dir):
        file_path = os.path.join(log_dir, filename)
        try:
            if os.path.isfile(file_path) and file_path.endswith('json'):
                os.remove(file_path)

        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))



@mlxp.launch(config_path='./configs')
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
        log_dir = ctx.logger.log_dir 
        clear_dir(os.path.join(log_dir,'metrics'))

        from applications.ModelBasedRL.funcBO.trainer import Trainer
        trainer = Trainer(ctx.config, ctx.logger)

    # Call the train method of the Trainer instance
    trainer.train()

# Entry point of the script, executing the train function if the script is run as the main module
if __name__ == "__main__":
    train()