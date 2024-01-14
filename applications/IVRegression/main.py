import time
time_0 = time.time()

import mlxp

time_00 = time.time()


from training import Trainer

time_01 = time.time()

from funcBO.utils import set_seed 


time_1 = time.time()


@mlxp.launch(config_path='configs',
             seeding_function=set_seed)
def train(ctx: mlxp.Context)->None:

    try:
        training  = ctx.logger.load_checkpoint(log_name='last_ckpt') 
        print("Loading from latest checkpoint")
     
    except:
        print("Failed to load checkpoint, Starting from scratch")
        training = Trainer(ctx.config, ctx.logger)


    training.train()


time_2 = time.time()

print("import mlxp time: "+str(time_00-time_0))
print("import trainer time: "+str(time_01-time_00))
print("import time: "+str(time_1-time_01))
print("mlxp time: "+str(time_2-time_1))



if __name__ == "__main__":
    
    train()
