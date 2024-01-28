from functools import partial
from collections import namedtuple

import jax
import jax.numpy as jnp
from jax import value_and_grad
from jax.scipy import stats
import optax
from jax.scipy.special import logsumexp
from jax.lax import stop_gradient

import sys
from os.path import dirname
sys.path.append('applications/ModelBasedRL/external/omd/cartpole/')



from utils import *
import chex

AuxP = namedtuple('AuxP', 'params_Q target_params_Q opt_state_Q rng')
AuxOut = namedtuple('AuxOut', 'vals_Q entropy_Q next_obs_nll')
AuxPdual = namedtuple('AuxPdual', 'params_Q target_params_Q params_dual_Q opt_state_Q opt_state_dual_Q rng')



nll_loss = lambda x, m, ls: -stats.norm.logpdf(x, m, jnp.exp(ls)).sum(-1).mean()
mse_loss = lambda x, xp: ((x - xp) ** 2).sum(-1).mean()


class Agent:
  def __init__(self, args, obs_space, action_space):
    self.args = args
    self.obs_dim = obs_space.shape[0]
    self.action_dim = action_space.n
    self.obs_range = (obs_space.low, obs_space.high)
    demo_obs = jnp.ones((1, self.obs_dim))
    demo_obs_action = jnp.ones((1, self.obs_dim + self.action_dim))
    self.rngs = hk.PRNGSequence(self.args.seed)
    
    if self.args.agent_type == 'vep':
      self.V = init_net_opt(self.args,'V', (self.obs_dim, self.action_dim, self.args.hidden_dim))
      self.params_V = self.V.net.init(next(self.rngs), demo_obs)

    self.T = init_net_opt(self.args,'T', (self.obs_dim, self.action_dim, self.args.model_hidden_dim))
    self.params_T = self.T.net.init(next(self.rngs), demo_obs_action)
    self.opt_state_T = self.T.opt.init(self.params_T)

    self.Q = init_net_opt(self.args,'Q', (self.obs_dim, self.action_dim, self.args.hidden_dim))
    self.params_Q = self.target_params_Q = self.Q.net.init(next(self.rngs), demo_obs)
    self.opt_state_Q = self.Q.opt.init(self.params_Q)

    if self.args.agent_type == 'funcBO':
      self.dual_Q = init_net_opt(self.args,'dual_Q', (self.obs_dim, self.action_dim, self.args.hidden_dim))
      self.params_dual_Q =  self.dual_Q.net.init(next(self.rngs), demo_obs)
      self.opt_state_dual_Q = self.dual_Q.opt.init(self.params_dual_Q)



  @partial(jax.jit, static_argnums=(0,))
  def act(self, params_Q, obs, rng):
    obs = jnp.array(obs[0]) if isinstance(obs, tuple) else obs[None, ...] 
    current_Q = self.Q.net.apply(params_Q, obs[None, ...])
    if not self.args.no_double:
      current_Q = 0.5 * (current_Q[0] + current_Q[1])
    if self.args.hard:
      action = jnp.argmax(current_Q, axis=-1)
    else:
      action = jax.random.categorical(rng, current_Q / self.args.alpha)
    return action if isinstance(obs, list) else action[0]

  @partial(jax.jit, static_argnums=(0,))
  def model_pred(self, params_T, obs, action, rng):

    ### Generate prediction of the model T from current state and action 
    if not isinstance(action, int):
      action = action[:, 0]
    a = jax.nn.one_hot(action, self.action_dim)
    x = jnp.concatenate((obs, a), axis=-1)
    if self.args.prob_model:
      if self.args.no_learn_reward:
        next_obs_pred = self.T.net.apply(params_T, x)
        reward_pred = None
      else:
        next_obs_pred, reward_pred = self.T.net.apply(params_T, x)
      means, logstds = next_obs_pred.split(2, axis=1)
      logstds = jnp.clip(logstds, -5.0, 2.0)
      noise = jax.random.normal(rng, shape=means.shape)
      samples = noise * jnp.exp(logstds) + means
      return samples, means, logstds, reward_pred
    else:
      if self.args.no_learn_reward:
        next_obs_pred = self.T.net.apply(params_T, x)
        reward_pred = None
      else:
        next_obs_pred, reward_pred = self.T.net.apply(params_T, x)
      return next_obs_pred, next_obs_pred, None, reward_pred
    
  def batch_real_to_model(self, params_T, batch, rng):
    ### Compares T model's prediction with actual outcome. 

    obs, action, reward, next_obs, not_done, not_done_no_max = batch
    next_obs_pred, means, logstds, reward_pred = self.model_pred(
        params_T, obs, action, rng)
    if self.args.no_learn_reward:
      reward_pred = reward
    batch_model = obs, action, reward_pred, next_obs_pred, not_done, not_done_no_max
    if self.args.prob_model:
      nll = nll_loss(next_obs, means, logstds)
    else:
      nll = mse_loss(next_obs_pred, next_obs)
    return batch_model, nll
  
  @partial(jax.jit, static_argnums=(0,))
  #@chex.assert_max_traces(n=1)
  def loss_Q(self, params_Q, target_params_Q, batch):

    ### Inner_level loss (need to make functional)
    ### Dependence on outer level comes from batch (when using model's pred)
    



    obs, action, reward, next_obs, not_done, not_done_no_max = batch
    target_Q = self.Q.net.apply(stop_gradient(target_params_Q), next_obs)
    Q_s = self.Q.net.apply(params_Q, obs)

    return self.func_loss_Q(Q_s,target_Q,batch)

  @partial(jax.jit, static_argnums=(0,))
  #@chex.assert_max_traces(n=1)
  def func_loss_Q(self,Q_s,target_Q,batch):

    obs, action, reward, next_obs, not_done, not_done_no_max = batch

    if self.args.hard:
      if self.args.no_double:
        target_V = jnp.max(target_Q, axis=-1, keepdims=True)
      else:
        target_Q = jnp.minimum(target_Q[0], target_Q[1])
        target_V = jnp.max(target_Q, axis=-1, keepdims=True)
    else:
      if self.args.no_double:
        target_V = self.args.alpha * logsumexp(target_Q / self.args.alpha, 
                                          axis=-1, keepdims=True)
      else:
        target_Q = jnp.minimum(target_Q[0], target_Q[1])
        target_V = self.args.alpha * logsumexp(target_Q / self.args.alpha, 
                                          axis=-1, keepdims=True)
    
    target_Q = (reward + (not_done_no_max * self.args.discount * target_V))[:, 0]

    if self.args.no_double:
      current_Q = Q_s[jnp.arange(obs.shape[0]), action.astype(int)[:, 0]]
      vals_Q = current_Q.mean()
      entropy_Q = (-jax.nn.log_softmax(Q_s) * jax.nn.softmax(Q_s)).sum(-1).mean()
      mse_Q = jnp.mean((current_Q - target_Q)**2)
    else:
      current_Q1 = Q_s[0][jnp.arange(obs.shape[0]), action.astype(int)[:, 0]]
      current_Q2 = Q_s[1][jnp.arange(obs.shape[0]), action.astype(int)[:, 0]]
      entropy_Q  = (-jax.nn.log_softmax(Q_s[0]) * jax.nn.softmax(Q_s[0])).sum(-1).mean()
      entropy_Q += (-jax.nn.log_softmax(Q_s[1]) * jax.nn.softmax(Q_s[1])).sum(-1).mean()
      vals_Q = 0.5 * (current_Q1.mean() + current_Q2.mean())
      mse_Q = 0.5 * (jnp.mean((current_Q1 - target_Q)**2) + jnp.mean((current_Q2 - target_Q)**2))

    aux_out = AuxOut(vals_Q, entropy_Q, None)
    return mse_Q, aux_out

  @partial(jax.jit, static_argnums=(0,))
  def loss_dual_Q(self,params_dual_Q, replay, outer_grad):
    
    #_, vdp_fun = jax.vjp(lambda val_Q: self.grad_inner_loss(params_T, val_Q, replay, rng, target_val_Q), val_Q)

    obs, action, reward, next_obs, not_done, not_done_no_max = replay
    val_dual_Q = self.dual_Q.net.apply(params_dual_Q,obs)
    if self.args.no_double:
      current_Q = val_dual_Q[jnp.arange(obs.shape[0]), action.astype(int)[:, 0]]
      target_Q = outer_grad[jnp.arange(obs.shape[0]), action.astype(int)[:, 0]]
#      vals_Q = current_Q.mean()
#      entropy_Q = (-jax.nn.log_softmax(Q_s) * jax.nn.softmax(Q_s)).sum(-1).mean()
      mse_Q = jnp.mean((current_Q - target_Q)**2)
    else:
      current_Q1 = val_dual_Q[0][jnp.arange(obs.shape[0]), action.astype(int)[:, 0]]
      current_Q2 = val_dual_Q[1][jnp.arange(obs.shape[0]), action.astype(int)[:, 0]]
      
      target_Q1 = outer_grad[0][jnp.arange(obs.shape[0]), action.astype(int)[:, 0]]
      target_Q2 = outer_grad[1][jnp.arange(obs.shape[0]), action.astype(int)[:, 0]]
      mse_Q = 0.5 * (jnp.mean((current_Q1 - target_Q1)**2) + jnp.mean((current_Q2 - target_Q2)**2))

    return mse_Q


  def constraint_func(self, params_T, params_Q, replay, rng, target_params_Q):
    '''Parameterized by T function giving grad_Q (Bellman-error) = 0 constraint.
    '''
    ### Evaluates the inner loss using outer model T 
    ### and computes gradient wrt inputs

    replay_model, _ = self.batch_real_to_model(params_T, replay, rng)
    grads, aux_out = jax.grad(self.loss_Q, has_aux=True)(
      params_Q, target_params_Q, replay_model)
    return grads


  def grad_inner_loss(self,params_T, val_Q, replay, rng, target_val_Q):
    '''Parameterized by T function giving grad_Q (Bellman-error) = 0 constraint.
    '''
    ### Evaluates the inner loss using outer model T 
    ### and computes gradient wrt inputs

    replay_model, _ = self.batch_real_to_model(params_T, replay, rng)
    grads, aux_out = jax.grad(self.func_loss_Q, has_aux=True)(
      val_Q, target_val_Q, replay_model)
    return grads

  def fwd_solver(self, 
    params_Q, target_params_Q, opt_state_Q, params_T, replay, rng):
    """Get Q_* satisfying the constraint (approximately). 
    """
    replay_model, nll = self.batch_real_to_model(params_T, replay, rng)
    
    if self.args.no_warm:
      target_params_Q = params_Q
    if not self.args.warm_opt:
      opt_state_Q = self.Q.opt.init(params_Q)
    
    for i in range(self.args.num_Q_steps):
      updout = self.update_step_inner(
        params_Q, target_params_Q, opt_state_Q, None, replay_model)
      params_Q, opt_state_Q = updout.params_Q, updout.opt_state_Q
      target_params_Q = soft_update_params(self.args.tau, params_Q, target_params_Q)

    Sol = namedtuple('Sol', 
      'params_Q loss_Q vals_Q grad_norm_Q entropy_Q target_params_Q opt_state_Q next_obs_nll')
    return Sol(params_Q, updout.loss_Q, updout.vals_Q, updout.grad_norm_Q, 
      updout.entropy_Q, target_params_Q, opt_state_Q, nll)

  @partial(jax.jit, static_argnums=(0,))
  @chex.assert_max_traces(n=1)
  def update_step_inner(self, params, aux_params, opt_state, batch, replay):
    (value, aux_out), grads = value_and_grad(self.loss_Q, has_aux=True)(
      params, aux_params, replay)
    updates, opt_state = self.Q.opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    UpdOut = namedtuple('Upd_inner', 
      'loss_Q params_Q opt_state_Q grads_Q grad_norm_Q vals_Q entropy_Q')
    return UpdOut(value, new_params, opt_state, grads, tree_norm(grads), 
      aux_out.vals_Q, aux_out.entropy_Q)



  def bwd_solver(self,
    params_dual_Q, opt_state_dual_Q, replay,rng, outer_grad):

    if not self.args.warm_opt:
      opt_state_dual_Q = self.dual_Q.opt.init(params_dual_Q)
    
    for i in range(self.args.num_dual_Q_steps):
      updout = self.update_step_dual(
        params_dual_Q, opt_state_dual_Q, 
        replay, outer_grad)

      params_dual_Q, opt_state_dual_Q = updout.params_dual_Q, updout.opt_state_dual_Q
      #target_params_Q = soft_update_params(self.args.tau, params_Q, target_params_Q)
    
    obs, action, reward, next_obs, not_done, not_done_no_max = replay
    val_dual_Q = self.dual_Q.net.apply(params_dual_Q,obs)

    DualSol = namedtuple('DualSol', 
      'params_dual_Q val_dual_Q loss_dual_Q opt_state_dual_Q grad_norm_dual_Q')
    return DualSol(params_dual_Q, val_dual_Q, updout.loss_dual_Q, opt_state_dual_Q, updout.grad_norm_dual_Q)

  @partial(jax.jit, static_argnums=(0,))
  @chex.assert_max_traces(n=1)
  def update_step_dual(self, params_dual_Q, opt_state, outer_replay, outer_grad):
    value, grads = value_and_grad(self.loss_dual_Q, has_aux=False)(
      params_dual_Q, outer_replay, outer_grad)
    updates, opt_state = self.dual_Q.opt.update(grads, opt_state)
    new_params = optax.apply_updates(params_dual_Q, updates)
    UpdOut = namedtuple('Upd_dual', 
      'loss_dual_Q params_dual_Q opt_state_dual_Q grads_dual_Q grad_norm_dual_Q')
    return UpdOut(value, new_params, opt_state, grads, tree_norm(grads))


  def loss_funcBO(self, params_T, aux_params, batch, replay):

    if self.args.no_warm:
      params_Q = self.Q.net.init(aux_params.rng, replay[0])
      params_dual_Q = self.dual_Q.net.init(aux_params.rng, replay[0])
    else:
      params_Q = aux_params.params_Q
      params_dual_Q = aux_params.params_dual_Q


    fwd_solver = lambda  params_Q, params_T, replay, rng: self.fwd_solver(
       params_Q, aux_params.target_params_Q, aux_params.opt_state_Q, params_T, replay, rng)
    
    bwd_solver = lambda  params_dual_Q, replay, rng, outer_grad: self.bwd_solver(
      params_dual_Q, aux_params.opt_state_dual_Q, 
      replay, rng, outer_grad)



    # Update params_Q on a batch w.r.t. loss yielded by params_T
    sol = inner_solution(self.grad_inner_loss, self.Q.net, params_Q, 
      params_T, replay, aux_params.rng, params_dual_Q,(fwd_solver,bwd_solver))

    #sol = inner_solution(self.grad_inner_loss, self.Q.net, params_Q, 
    #  params_T, replay, aux_params.rng, params_dual_Q,(fwd_solver,))

    
    #sol = root_solve(self.constraint_func, params_Q, 
    #  params_T, replay, aux_params.rng, (fwd_solver,))    


    return self.func_loss_Q(sol.val_Q, sol.val_target_Q, replay)[0], sol
    #return self.loss_Q(sol.params_Q, sol.target_params_Q, replay)[0], sol

  def loss_omd(self, params_T, aux_params, batch, replay):
    fwd_solver = lambda params_Q, params_T, replay, rng: self.fwd_solver(
                  params_Q, aux_params.target_params_Q, aux_params.opt_state_Q, params_T, replay, rng)
    if self.args.no_warm:
      params_Q = self.Q.net.init(aux_params.rng, replay[0])
    else:
      params_Q = aux_params.params_Q

    # Update params_Q on a batch w.r.t. loss yielded by params_T
    sol = root_solve(self.constraint_func, params_Q, 
      params_T, replay, aux_params.rng, (fwd_solver,))
    
    return self.loss_Q(sol.params_Q, sol.target_params_Q, replay)[0], sol

  def loss_mle(self, params_T, batch, rng):
    obs, action, reward, next_obs, not_done, not_done_no_max = batch
    pred, means, logstds, reward_pred = self.model_pred(params_T, obs, action, rng)
    assert next_obs.ndim == pred.ndim  # no undesired broadcasting
    
    nll = nll_loss(next_obs, means, logstds) if self.args.prob_model else mse_loss(pred, next_obs)
    if not self.args.no_learn_reward:
      assert reward_pred.ndim == reward.ndim  # no undesired broadcasting
      nll += ((reward_pred - reward) ** 2).mean()
    return nll

  def loss_vep(self, params_T, aux_params, batch, rng):
    assert not self.args.prob_model
    obs, action, reward, next_obs, not_done, not_done_no_max = batch
    pred, means, logstds, reward_pred = self.model_pred(params_T, obs, action, rng)
    assert next_obs.ndim == pred.ndim  # no undesired broadcasting
    nll = nll_loss(next_obs, means, logstds) if self.args.prob_model else mse_loss(pred, next_obs)
    
    # note that VFs are random
    params_V = stop_gradient(aux_params)
    next_V = self.V.net.apply(params_V, next_obs)
    pred_V = self.V.net.apply(params_V, pred)
    l = 0
    for i in range(self.args.num_ensemble_vep):
      l += jnp.mean((next_V[i] - pred_V[i])**2)

    if not self.args.no_learn_reward:
      assert reward_pred.ndim == reward.ndim  # no undesired broadcasting
      l += ((reward_pred - reward) ** 2).mean()
    aux_out = AuxOut(None, None, nll)
    return l, aux_out



  @partial(jax.jit, static_argnums=(0,6))
  @chex.assert_max_traces(n=1) 
  def update_step_outer(self, params_T, aux_params, opt_state_T, batch, replay,loss):
      (value, aux_out), grads = value_and_grad(loss, has_aux=True)(
        params_T, aux_params, batch, replay)
      updates, opt_state_T = self.T.opt.update(grads, opt_state_T)
      new_params = optax.apply_updates(params_T, updates)
      UpdOut = namedtuple('Upd_outer', 
        'loss_T params_T opt_state_T loss_Q vals_Q grad_norm_Q entropy_Q params_Q target_params_Q opt_state_Q next_obs_nll')
      return UpdOut(value, new_params, opt_state_T, aux_out.loss_Q, 
        aux_out.vals_Q, aux_out.grad_norm_Q, aux_out.entropy_Q, aux_out.params_Q, 
        aux_out.target_params_Q, aux_out.opt_state_Q, aux_out.next_obs_nll)




  @partial(jax.jit, static_argnums=(0,6))
  #@chex.assert_max_traces(n=1)
  def update_step(self, params, aux_params, opt_state, batch, replay, loss_type):

    if loss_type == 'sql':
      (value, aux_out), grads = value_and_grad(self.loss_Q, has_aux=True)(
        params, aux_params, replay)
      updates, opt_state = self.Q.opt.update(grads, opt_state)
      new_params = optax.apply_updates(params, updates)
      UpdOut = namedtuple('Upd_{}'.format(loss_type), 
        'loss_Q params_Q opt_state_Q grads_Q grad_norm_Q vals_Q entropy_Q')
      return UpdOut(value, new_params, opt_state, grads, tree_norm(grads), 
        aux_out.vals_Q, aux_out.entropy_Q)

    elif loss_type == "mle":
      value, grads = value_and_grad(self.loss_mle)(params, replay, aux_params.rng)
      updates, opt_state = self.T.opt.update(grads, opt_state)
      new_params = optax.apply_updates(params, updates)
      UpdOut = namedtuple('Upd_{}'.format(loss_type), 
        'loss_T params_T opt_state_T')
      return UpdOut(value, new_params, opt_state)

    elif loss_type == "vep":
      (value, aux_out), grads = value_and_grad(self.loss_vep, has_aux=True)(
          params, aux_params.params_Q, replay, aux_params.rng)
      updates, opt_state = self.T.opt.update(grads, opt_state)
      new_params = optax.apply_updates(params, updates)
      UpdOut = namedtuple('Upd_{}'.format(loss_type), 
        'loss_T params_T opt_state_T next_obs_nll')
      return UpdOut(value, new_params, opt_state, aux_out.next_obs_nll)





  def update(self, replay_buffer):
    replay = replay_buffer.sample(self.args.batch_size)
    
    if self.args.agent_type in ['omd', 'funcBO']:
      if self.args.agent_type=='omd':
        outer_loss = self.loss_omd 
        if self.args.no_warm:
          aux_params = AuxP(None, None, None, next(self.rngs))
        else:
          aux_params = AuxP(self.params_Q, self.target_params_Q, self.opt_state_Q, 
            next(self.rngs))
      else:
        outer_loss = self.loss_funcBO
        if self.args.no_warm:
          aux_params = AuxPdual(None, None, None, None, None,next(self.rngs))
        else:
          aux_params = AuxPdual(self.params_Q, self.target_params_Q, self.params_dual_Q, self.opt_state_Q,
            self.opt_state_dual_Q, 
            next(self.rngs))          

      batch = None

      updout = self.update_step_outer(self.params_T, aux_params, self.opt_state_T, 
                                        batch, replay, outer_loss)
      self.params_Q, self.opt_state_Q = updout.params_Q, updout.opt_state_Q
      self.target_params_Q = updout.target_params_Q
      self.params_T, self.opt_state_T = updout.params_T, updout.opt_state_T

      return {'loss_T': updout.loss_T.item(), 
              'vals_Q': updout.vals_Q.item(), 
              'loss_Q': updout.loss_Q.item(), 
              'grad_norm_Q': updout.grad_norm_Q.item(), 
              'entropy_Q': updout.entropy_Q.item(),
              'next_obs_nll': updout.next_obs_nll.item()}


    elif self.args.agent_type == 'mle':
      for i in range(self.args.num_T_steps):
        aux_params = AuxP(None, None, None, next(self.rngs))
        replay = replay_buffer.sample(self.args.batch_size)
        updout_T = self.update_step(self.params_T, aux_params, self.opt_state_T, 
          None, replay, 'mle')
        self.params_T, self.opt_state_T = updout_T.params_T, updout_T.opt_state_T

      for i in range(self.args.num_Q_steps):
        replay = replay_buffer.sample(self.args.batch_size)
        replay_model, nll = self.batch_real_to_model(self.params_T, replay, next(self.rngs))
        updout_Q = self.update_step(self.params_Q, self.target_params_Q, 
          self.opt_state_Q, None, replay_model, 'sql')    
        self.params_Q, self.opt_state_Q = updout_Q.params_Q, updout_Q.opt_state_Q
        self.target_params_Q = soft_update_params(
          self.args.tau, self.params_Q, self.target_params_Q)

      return {'loss_T': updout_T.loss_T.item(), 
              'vals_Q': updout_Q.vals_Q.item(), 
              'loss_Q': updout_Q.loss_Q.item(), 
              'grad_norm_Q': updout_Q.grad_norm_Q.item(), 
              'entropy_Q': updout_Q.entropy_Q.item()}

    elif self.args.agent_type == 'vep':
      for i in range(self.args.num_T_steps):
        aux_params = AuxP(self.params_V, None, None, next(self.rngs))
        replay = replay_buffer.sample(self.args.batch_size)
        updout_T = self.update_step(self.params_T, aux_params, self.opt_state_T, 
          None, replay, 'vep')
        self.params_T, self.opt_state_T = updout_T.params_T, updout_T.opt_state_T

      for i in range(self.args.num_Q_steps):
        replay = replay_buffer.sample(self.args.batch_size)
        replay_model, nll = self.batch_real_to_model(self.params_T, replay, next(self.rngs))
        updout_Q = self.update_step(self.params_Q, self.target_params_Q, 
          self.opt_state_Q, None, replay_model, 'sql')    
        self.params_Q, self.opt_state_Q = updout_Q.params_Q, updout_Q.opt_state_Q
        self.target_params_Q = soft_update_params(
          self.args.tau, self.params_Q, self.target_params_Q)

      return {'loss_T': updout_T.loss_T.item(), 
              'vals_Q': updout_Q.vals_Q.item(), 
              'loss_Q': updout_Q.loss_Q.item(), 
              'grad_norm_Q': updout_Q.grad_norm_Q.item(), 
              'entropy_Q': updout_Q.entropy_Q.item(),
              'next_obs_nll': updout_T.next_obs_nll.item()}






  # def save(self, agent_path):
  #   pickle.dump([self.params_Q, self.target_params_Q, self.params_T], 
  #               open(agent_path, 'wb'))
  
  # def load(self, agent_path):
  #   self.params_Q, self.target_params_Q, self.params_T = pickle.load(
  #     open(agent_path, "rb"))
