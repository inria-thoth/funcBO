import haiku as hk
import os
import gym
import numpy as np
import time

import jax
from jax.config import config
config.update("jax_enable_x64", True)


import sys
from os.path import dirname
sys.path.append('../external/')


from applications.ModelBasedRL.external.omd.cartpole.omd import Agent
from applications.ModelBasedRL.external.omd.cartpole.replay_buffer import ReplayBuffer
from applications.ModelBasedRL.external.omd.cartpole.utils import *

class Trainer:
    """
    Solves an instrumental regression problem using the bilevel functional method.
    """
    def __init__(self, config, logger):
        """
        Initializes the Trainer class with the provided configuration and logger.

        Parameters:
        - config (object): Configuration object containing various settings.
        - logger (object): Logger object for logging metrics and information.
        """
        self.logger = logger
        self.args = config
        #self.device = assign_device(self.args.system.device)
        #self.dtype = get_dtype(self.args.system.dtype)
        #torch.set_default_dtype(self.dtype)
        self.build_trainer()
        self.iters = 0

    def log(self,dico, log_name='metrics'):
        self.logger.log_metrics(dico, log_name=log_name)

    def log_metrics_list(self, dico_list, iteration, prefix="", log_name='metrics'):
        total_iter = len(dico_list)
        for dico in dico_list:
            dico['outer_iter'] = iteration
            dico['iter'] = iteration*total_iter + dico['inner_iter']
            dico = {prefix+key:value for key,value in dico.items()}
            self.log(dico, log_name=log_name)

    def build_trainer(self):

        self.env = gym.make(self.args.env_name)
        self.eval_env = gym.make(self.args.env_name)

        for e in [self.eval_env, self.env]:
          observation = e.reset(seed=self.args.seed, options={})
          #e.seed(self.args.seed)
          e.action_space.seed(self.args.seed)
          e.observation_space.seed(self.args.seed)
        np.random.seed(self.args.seed)
        self.rngs = hk.PRNGSequence(self.args.seed)
        
        self.agent = Agent(self.args,self.env.observation_space, self.env.action_space)
        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape, self.args.num_train_steps)


        self.step, self.episode = 0, 0
        self.episode_return, self.episode_step = 0, 0
        self.cum_return = 0.0

    def train(self):
        """
        The main optimization loop for the bilevel functional method.
        """
        # This is an atribute to be able to start in the middle if starting from a checkpoint
        obs = self.env.reset()
        action = self.agent.act(self.agent.params_Q, obs, next(self.rngs))
        start_time = time.time()
        cur_eval_time = start_time
        done = False

        while self.step < self.args.num_train_steps:
            
            # evaluate agent periodically
            if self.step % self.args.eval_frequency == 0:
                eval_return = evaluate(self.agent, self.eval_env, next(self.rngs))
                eval_time  = time.time()
                eval_dict = {'episode': self.episode,
                              'episode_return': eval_return,
                              'step': self.step,
                              'time': eval_time-cur_eval_time}
                cur_eval_time = eval_time
                self.log(eval_dict, log_name='eval')
                print(eval_dict)

            # with epsilon exploration
            action = self.env.action_space.sample() if (np.random.rand() < self.args.eps or self.step < self.args.init_steps) else action.item()
            
            ## next state and reward
            next_obs, reward, done, _ = self.env.step(action)

            done = float(done)
            # allow infinite bootstrap
            done_no_max = 0 if self.episode_step + 1 == self.env._max_episode_steps else done
            self.episode_return += reward

            self.replay_buffer.add(obs, action, reward, 
                                  next_obs, done, done_no_max)
            
            obs = next_obs
            self.episode_step += 1
            self.step += 1

            if done:
              train_dict = {'episode': self.episode,
                            'episode_return': self.episode_return,
                            'duration': time.time() - start_time,
                            'step': self.step}
              
              self.log(train_dict, log_name='train_returns')
              #print(train_dict)
              obs = self.env.reset()
              done = False
              self.episode_return = 0
              self.episode_step = 0
              self.episode += 1

            action = self.agent.act(self.agent.params_Q, obs, next(self.rngs))
            
            if self.step >= self.args.init_steps:
              losses_dict = self.agent.update(self.replay_buffer)
              losses_dict['step'] = self.step


              if (self.step % self.args.log_frequency) == 0:
                self.log(losses_dict, log_name='train')
                #print(losses_dict)
              if (self.step % self.args.ckpt_frequency) == 0:
                self.logger.log_checkpoint(self,log_name='last_ckpt')

          # final eval after training is done
        eval_return = evaluate(self.agent, self.eval_env, next(self.rngs))
        eval_time  = time.time()
        eval_dict = {'episode': self.episode,
                      'episode_return': eval_return,
                      'step': self.step,
                      'time': eval_time-cur_eval_time}
        cur_eval_time = eval_time
        self.log(eval_dict, log_name='eval')


        #print("Done in {:.1f} minutes".format((time.time() - start_time)/60))










