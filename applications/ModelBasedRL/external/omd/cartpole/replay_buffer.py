import numpy as np
import jax
import jax.numpy as jnp
import pickle

class ReplayBuffer:
  """Buffer to store environment transitions. Discrete actions only"""
  def __init__(self, obs_shape, capacity):
    self.capacity = capacity

    self.obses = jnp.empty((capacity, *obs_shape), dtype=jnp.float32)
    self.next_obses = jnp.empty((capacity, *obs_shape), dtype=jnp.float32)
    self.actions = jnp.empty((capacity, 1), dtype=jnp.float32)
    self.rewards = jnp.empty((capacity, 1), dtype=jnp.float32)
    self.not_dones = jnp.empty((capacity, 1), dtype=jnp.float32)
    self.not_dones_no_max = jnp.empty((capacity, 1), dtype=jnp.float32)

    self.idx = 0
    self.full = False

  def __len__(self):
    return self.capacity if self.full else self.idx

  def add(self, obs, action, reward, next_obs, done, done_no_max):
    
    self.obses = self.obses.at[self.idx].set(obs)
    self.actions = self.actions.at[self.idx].set(action)
    self.rewards = self.rewards.at[self.idx].set(reward)
    self.next_obses = self.next_obses.at[self.idx].set(next_obs)
    self.not_dones = self.not_dones.at[self.idx].set(not done)
    self.not_dones_no_max = self.not_dones_no_max.at[self.idx].set(not done_no_max)

    self.idx = (self.idx + 1) % self.capacity
    self.full = self.full or self.idx == 0

  def sample(self, batch_size, replace=False):
    idxs = np.random.choice(len(self), size=batch_size, replace=replace)

    obses = self.obses[idxs]
    actions = self.actions[idxs]
    rewards = self.rewards[idxs]
    next_obses = self.next_obses[idxs]
    not_dones = self.not_dones[idxs]
    not_dones_no_max = self.not_dones_no_max[idxs]

    return obses, actions, rewards, next_obses, not_dones, not_dones_no_max

  def save(self, data_path):
    all_data = [
      self.obses,
      self.actions,
      self.rewards,
      self.next_obses,
      self.not_dones,
      self.not_dones_no_max
    ]
    pickle.dump(all_data, open(data_path, 'wb'))
  
  def load(self, data_path):
    self.obses, self.actions, self.rewards, self.next_obses, \
      self.not_dones, self.not_dones_no_max = pickle.load(open(data_path, "rb"))
    self.capacity = len(self.obses)
    self.full = True
